# mini-llm 项目训练链路与 bf16 改造指南

这份文档的目标不是直接给你一个黑盒补丁，而是帮你把当前项目从数据到训练完整过一遍，并说明如果要支持 `bf16` 训练，应该改哪些地方、为什么这么改、改完怎么确认是对的。

## 1. 当前项目怎么理解

当前项目的主要边界可以按这几块看：

```text
cs336_basics/
  get_tokens.py      原始文本 -> token ids -> .npy
  tokenizer.py       tokenizer 实现
  model.py           TransformerLM 模型结构
  train.py           训练入口、batch、loss、optimizer、评估、checkpoint

scripts/
  train_0.5b.sh      0.5B 模型启动脚本

data/
  *.txt              原始文本
  *.npy              tokenized 数据

runs/
  0.5b/train.log     训练日志

checkpoints/
  0.5b/latest.pt     模型 checkpoint
```

整体训练链路是：

```text
文本数据
  -> get_tokens.py
  -> token ids .npy
  -> train.py 读入 token ids
  -> get_batch 随机采样 batch
  -> TransformerLM 前向
  -> cross_entropy_loss
  -> backward
  -> gradient clipping
  -> AdamW step
  -> evaluate
  -> checkpoint
```

你平时运行项目，推荐从项目根目录用模块方式：

```bash
cd /root/mini-llm
uv run python -m cs336_basics.get_tokens
uv run python -m cs336_basics.train ...
```

不要写成：

```bash
uv run python -m cs336_basics.get_tokens.py
```

`-m` 后面是模块名，不是文件名，所以不能带 `.py`。

## 2. 当前 0.5B 配置在做什么

当前 0.5B 模型大致是：

```text
vocab_size      50257
context_length  1024
d_model         1152
num_layers      24
num_heads       18
d_ff            3072
```

参数量大约：

```text
498M
```

训练日志里能看到类似：

```text
params=498054528
train_tokens=...
valid_tokens=...
tokens_per_iter=...
```

这些指标可以判断三件事：

```text
params           模型规模是否符合预期
tokens_per_iter  每步实际吃多少 token
loss / ppl       训练是否真的在下降
```

## 3. 为什么要做 bf16

现在如果不加混合精度，训练基本是 fp32。

fp32 的特点：

```text
稳定
显存占用大
速度通常不如 bf16/fp16
```

bf16 的特点：

```text
比 fp32 省显存
通常更快
数值范围接近 fp32
比 fp16 稳
不需要 GradScaler
```

所以对你这个项目，优先级应该是：

```text
先 bf16
再考虑 fp16
```

不要一开始就上 fp16，因为 fp16 通常需要 `GradScaler`，否则更容易出现梯度 underflow 或 NaN。

## 4. bf16 的正确使用方式

推荐的策略是：

```text
模型参数仍然保持 fp32
optimizer state 仍然保持 fp32
前向计算用 bf16 autocast
loss 计算转回 fp32
反向传播正常 backward
optimizer 正常 step
```

也就是：

```text
不是把整个模型 model.to(torch.bfloat16)
而是在前向区域使用 torch.autocast
```

这样更稳。

核心形态是：

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(batch)

loss = cross_entropy_loss(logits.float(), targets)
```

关键点：

```text
logits.float()
```

你的 `cross_entropy_loss` 是自己写的，里面有：

```text
exp
log
sum
```

这些操作用 fp32 更稳，所以建议前向可以 bf16，但 loss 用 fp32。

## 5. 应该怎么改 TrainConfig

在 `cs336_basics/train.py` 的 `TrainConfig` 里增加一个字段：

```python
dtype: str = "fp32"
```

含义：

```text
fp32  不启用 autocast
bf16  使用 torch.bfloat16 autocast
```

暂时不建议急着支持 fp16。等 bf16 稳了，再加 fp16 和 GradScaler。

## 6. 应该怎么改命令行参数

在 `parse_args()` 里增加：

```python
parser.add_argument("--dtype", type=str, choices=["fp32", "bf16"], default="fp32")
```

这样训练时可以通过脚本控制：

```bash
--dtype bf16
```

默认仍然是 `fp32`，这样不会影响已有运行方式。

## 7. 建议新增一个 autocast 工具入口

为了不把训练循环写乱，可以单独加几个小入口。

建议新增：

```python
def get_autocast_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "fp32":
        return None
    if dtype == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")
```

再加：

```python
def should_use_autocast(dtype: str) -> bool:
    return dtype != "fp32"
```

这样 `train()` 和 `evaluate()` 里不用到处判断字符串。

## 8. 建议把前向和 loss 收敛成一个入口

当前训练里大致是：

```text
batch, targets
  -> model(batch)
  -> reshape logits
  -> reshape targets
  -> cross_entropy_loss
```

支持 bf16 后，这一段最好收敛成一个入口，比如：

```python
def forward_loss(
    model: TransformerLM,
    batch: torch.Tensor,
    targets: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    autocast_dtype = get_autocast_dtype(cfg.dtype)

    with torch.autocast(
        device_type="cuda",
        dtype=autocast_dtype or torch.float32,
        enabled=should_use_autocast(cfg.dtype),
    ):
        logits = model(batch)

    logits = logits.float().reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    return cross_entropy_loss(logits, targets)
```

这样训练和验证都能复用这条路径。

## 9. train loop 应该怎么改

原来的训练逻辑大概是：

```text
batch, targets = get_batch(...)
logits = model(batch)
logits reshape
targets reshape
loss = cross_entropy_loss(logits, targets)
loss.backward()
grad_norm = clip_gradients(...)
opt.step()
```

改完后应该变成：

```text
batch, targets = get_batch(...)
loss = forward_loss(model, batch, targets, cfg)
loss.backward()
grad_norm = clip_gradients(...)
opt.step()
```

也就是把前向、autocast、loss dtype 处理都收到 `forward_loss` 里。

## 10. evaluate 也要改

这是很容易漏的地方。

如果只改训练，不改验证，那么：

```text
训练前向是 bf16
验证前向还是 fp32
```

这不是不能跑，但指标速度和显存状态不一致。

建议验证里也走：

```python
loss = forward_loss(model, x, y, cfg)
```

也就是 `evaluate()` 改成：

```python
def evaluate(
    model: TransformerLM,
    valid_data: np.ndarray,
    cfg: TrainConfig,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = get_batch(valid_data, cfg.batch_size, cfg.context_length, cfg.device)
            loss = forward_loss(model, x, y, cfg)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))
```

这样训练 loss 和验证 loss 的计算路径一致。

## 11. 启动脚本怎么改

在 `scripts/train_0.5b.sh` 的训练参数里加一行：

```bash
--dtype bf16 \
```

例如放在：

```bash
--device cuda \
--dtype bf16 \
--checkpoint_path "${CHECKPOINT_DIR}/latest.pt"
```

如果你想继续用 fp32，就不传，或者写：

```bash
--dtype fp32
```

## 12. batch size 怎么配

你有 48GB 显存，可以这样试：

```text
第一轮：bf16 + batch_size=4
第二轮：bf16 + batch_size=8
```

不要同时改太多变量。

建议第一轮只改：

```text
dtype: fp32 -> bf16
batch_size: 4 保持不变或从 1 调到 4
```

观察：

```text
tokens_per_iter
tokens_per_sec
train_loss
val_loss
val_ppl
grad_norm
是否出现 NaN
是否 OOM
```

## 13. 日志里应该新增 dtype

训练启动时最好把 dtype 也打印出来。

当前启动信息里已经有：

```text
params
train_tokens
valid_tokens
tokens_per_iter
```

建议加上：

```text
dtype
```

也就是启动时能看到：

```text
params=498054528 train_tokens=... valid_tokens=... tokens_per_iter=4096 dtype=bf16
```

这样以后看 `train.log` 就知道这次实验到底是 fp32 还是 bf16。

## 14. 改完怎么验证

先不要直接跑 100000 步。

建议先短跑：

```bash
uv run python -m cs336_basics.train \
  --train_tokens_path data/ts-train.npy \
  --valid_tokens_path data/ts-val.npy \
  --vocab_size 50257 \
  --context_length 128 \
  --d_model 256 \
  --num_layers 4 \
  --num_heads 4 \
  --d_ff 768 \
  --batch_size 8 \
  --total_iters 20 \
  --eval_interval 5 \
  --eval_batches 2 \
  --checkpoint_interval 20 \
  --max_learning_rate 3e-4 \
  --min_learning_rate 3e-5 \
  --warmup_iters 5 \
  --cosine_cycle_iters 20 \
  --device cuda \
  --dtype bf16 \
  --checkpoint_path checkpoints/debug-bf16/latest.pt
```

这个测试的目标不是训好模型，而是确认：

```text
能启动
能前向
能 backward
能 eval
能 save checkpoint
loss 不是 NaN
tokens_per_sec 正常打印
```

## 15. 正常日志应该长什么样

正常的话，你会看到类似：

```text
params=... train_tokens=... valid_tokens=... tokens_per_iter=... dtype=bf16
it=0 tokens_seen=... train_loss=... train_ppl=... val_loss=... val_ppl=... lr=... grad_norm=... tokens_per_sec=...
it=5 tokens_seen=... train_loss=... val_loss=... val_ppl=...
```

重点看：

```text
loss 是有限数字
ppl 是有限数字或初期很大但不是 NaN
grad_norm 是有限数字
tokens_per_sec 比 fp32 更好
```

## 16. 常见问题

### 16.1 出现 NaN

优先检查：

```text
loss 是否用了 logits.float()
learning rate 是否太大
batch 是否有非法 token id
```

先把学习率降到：

```text
1e-4
```

再试。

### 16.2 CUDA out of memory

按这个顺序降：

```text
batch_size 8 -> 4 -> 2 -> 1
context_length 1024 -> 512
```

不要先降模型参数，因为你现在目标是训练 0.5B。

### 16.3 bf16 没有变快

可能原因：

```text
GPU 对 bf16 支持一般
瓶颈在自定义 Python 代码或数据采样
batch_size 太小，GPU 没吃满
```

可以对比：

```text
fp32 batch_size=4 tokens_per_sec
bf16 batch_size=4 tokens_per_sec
bf16 batch_size=8 tokens_per_sec
```

## 17. 推荐最终改造后的运行方式

小模型验证：

```bash
uv run python -m cs336_basics.train ... --dtype bf16
```

0.5B 正式训练：

```bash
RUN_NAME=0.5b_bf16_bs4 ./scripts/train_0.5b.sh
```

脚本内建议包含：

```bash
--batch_size 4 \
--dtype bf16 \
```

如果 batch size 8 稳定，可以新开一个实验：

```bash
RUN_NAME=0.5b_bf16_bs8 ./scripts/train_0.5b.sh
```

## 18. 复习重点

这次改造其实是在复习训练系统的几个关键边界：

```text
配置层：TrainConfig / argparse
运行层：scripts/*.sh
数据层：.txt -> .npy -> get_batch
模型层：TransformerLM
数值层：fp32 / bf16 / autocast
优化层：loss -> backward -> clip -> AdamW step
评估层：evaluate / val_loss / val_ppl
记录层：train.log / tokens_per_sec / grad_norm
产物层：checkpoints/latest.pt
```

最重要的理解是：

```text
bf16 不是换数据，也不是换模型结构。
bf16 是改变前向计算的数值精度边界。
```

推荐改法是：

```text
参数保留 fp32
前向 autocast 到 bf16
loss 回到 fp32
backward 和 optimizer 正常走
```

这样最稳，也最适合你现在这个 mini-llm 项目。