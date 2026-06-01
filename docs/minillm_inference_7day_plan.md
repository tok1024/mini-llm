# MiniLLM Inference Lab 七天冲刺计划

目标：在一周内把现有 CS336 from-scratch Transformer 项目扩展成一个可展示、可 benchmark、可写进保研简历的 LLM 推理优化项目。

项目定位：

> MiniLLM Inference Lab：基于自研 TransformerLM，实现 KV Cache、prefill/decode、GQA/MQA、SDPA/FlashAttention backend、KV cache quantization、prefix caching / chunked prefill，并构建轻量级 Eval Harness 做量化对比。

核心原则：

- 先跑通，再优化。
- 每天必须有可运行产物。
- 每个优化都必须有数字，否则不写进简历。
- 你自己写核心逻辑：attention cache、GQA、quantization、prefix cache、benchmark。
- 胶水代码可以后面快速补，但 benchmark 记录格式从第一天就固定。
- 不写“生产级”，写“from-scratch / mini / experimental inference engine”。

## 最终交付物

### 代码交付

建议新增或改造这些文件：

```text
cs336-proj1/
  cs336_basics/
    inference.py          # sampling、generate、prefill/decode 入口
    kv_cache.py           # KV cache 数据结构
    eval.py               # perplexity、distinct-n、repetition
    benchmark.py          # 计时、显存、tokens/sec 工具
  scripts/
    generate.py           # 从 checkpoint 生成
    evaluate_lm.py        # 跑 PPL / generation metrics
    bench_inference.py    # 跑 naive/cache/GQA/SDPA/quant/prefix 对比
  configs/
    inference_tiny.yaml   # 可选，没时间可先不用 YAML
```

会改到的老文件：

```text
cs336-proj1/cs336_basics/model.py
```

### 文档交付

```text
portfolio/results/minillm-inference/
  README.md
  metrics.md
  ablation_table.csv
  samples.md
  notes.md
  figures/
    speedup_by_context.png
    memory_by_context.png
    backend_comparison.png
```

### 简历交付

最终简历 bullet 不要提前写死数字。先写模板：

- 基于自研 decoder-only TransformerLM 实现 KV Cache 增量解码，将生成流程拆分为 prefill/decode 两阶段，复用各层历史 Key/Value，并在不同 context length 下 benchmark tokens/sec、latency 与显存占用。
- 构建轻量级 LLM Eval Harness，支持 perplexity、distinct-n、重复率、多 checkpoint 横向对比和 Markdown/CSV 实验报告生成，用于量化推理优化效果。
- 扩展 attention 模块支持 GQA/MQA、SDPA backend、KV cache INT8 量化与 prefix caching，系统分析速度、显存、PPL 之间的 trade-off。

等跑出真实数据后，再替换成：

```text
tokens/sec 提升 X.Xx，peak memory 下降 Y%，PPL 变化 < Z%
```

## 每天固定流程

每天都按这个节奏：

1. 开始前创建当天 TODO。
2. 只做当天目标，不额外开坑。
3. 核心代码写完后先用 toy tensor / tiny model 测。
4. 跑一次 benchmark，哪怕数据很粗。
5. 记录到 `portfolio/results/minillm-inference/metrics.md`。
6. commit 或至少保存 patch。
7. 写当天 notes：做了什么、卡在哪里、明天改什么。

每日记录模板：

```markdown
## Day N - YYYY-MM-DD

### Done
- 

### Commands
```bash

```

### Results
| Variant | Context | New Tokens | tok/s | Latency ms/token | Peak Mem MB | PPL |
|---|---:|---:|---:|---:|---:|---:|

### Notes
-
```

## Benchmark 统一标准

固定 benchmark 条件，否则数字不可比。

基础配置：

- batch size：先用 1，后面有余力再测 4/8。
- prompt length / context：128、256、512、1024。
- new tokens：64 或 128。
- seed：固定 42。
- dtype：先 fp32 跑通；如果 CUDA 支持，再测 fp16/bf16。
- device：记录 CPU / CUDA / MPS。
- warmup：至少 3 次。
- measured runs：至少 5 次取平均。

核心指标：

- `tokens_per_second = new_tokens / elapsed_seconds`
- `latency_ms_per_token = elapsed_seconds * 1000 / new_tokens`
- `peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2`
- `ppl`
- `distinct_1 / distinct_2`
- `repetition_4gram`

最重要的 ablation：

| Variant | 必须做吗 | 说明 |
|---|---|---|
| naive generate | 必做 | baseline |
| KV cache torch.cat | 必做 | 先跑通 |
| KV cache preallocated | 必做 | 正式版本 |
| SDPA backend | 必做 | 和手写 attention 对比 |
| GQA | 建议做 | 架构改动，注意要说明 trade-off |
| KV INT8 | 建议做 | 重点看 memory 和 PPL |
| prefix cache | 建议做 | 做 exact prefix 版本即可 |
| chunked prefill | 有时间做 | 简化版 |
| FlashAttention | 有环境再做 | 没环境就只做 fallback |

## Day 0：准备和边界确认，1-2 小时

这一天不算正式开发日，但建议今晚就做。

### 目标

确认当前模型、训练 checkpoint、tokenizer、测试方式，避免 Day 1 才发现没有可用输入。

### 任务

1. 跑通当前单元测试：

```bash
cd /Users/toki/Code/Courses/cs336/toki-cs336/cs336-proj1
uv run pytest
```

2. 找到当前生成 notebook / checkpoint 路径：

```bash
rg -n "generate|checkpoint|load_checkpoint|decode" .
```

3. 确认 `TransformerLM.forward(input_ids)` 输入输出：

```text
input_ids: [B, T]
logits: [B, T, vocab_size]
```

4. 创建结果目录：

```bash
mkdir -p /Users/toki/Code/Courses/cs336/toki-cs336/portfolio/results/minillm-inference/figures
```

5. 写 `portfolio/results/minillm-inference/notes.md` 初始说明。

### 验收

- 你知道当前模型 forward 路径在哪里。
- 你知道当前 tokenizer 怎么 encode/decode。
- 你有一个 tiny prompt 可以生成。
- 你有一个地方记录 benchmark。

### 降级方案

如果没有训练好的 checkpoint，就先用随机 tiny model 做推理 benchmark。KV cache 的速度对比仍然可以展示工程逻辑；PPL 等有 checkpoint 后再补。

## Day 1：KV Cache 最小可用版

### 今日目标

实现最小可用 KV cache：prefill 一次 prompt，decode 阶段每次只输入最后一个 token，并复用每层过去的 K/V。

今天先用 `torch.cat` 版本，不追求最优性能，只追求正确。

### 需要理解的关键点

naive generate：

```text
step 1: model([x1 ... xt])
step 2: model([x1 ... xt y1])
step 3: model([x1 ... xt y1 y2])
```

KV cache generate：

```text
prefill: model([x1 ... xt]) -> cache K/V for all layers
decode 1: model([y1], cache)
decode 2: model([y2], cache)
```

### 代码任务

#### 1. 新建 `cs336_basics/kv_cache.py`

先写一个简单 dataclass：

```python
from dataclasses import dataclass
import torch

@dataclass
class LayerKVCache:
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k = k_new
            self.v = v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        return self.k, self.v

    def reset(self) -> None:
        self.k = None
        self.v = None
```

约定 shape：

```text
k/v: [B, H, T, D]
沿 dim=2 追加序列长度
```

#### 2. 改 `MultiHeadSelfAttention.forward`

建议签名：

```python
def forward(
    self,
    x: torch.Tensor,
    kv_cache: LayerKVCache | None = None,
    use_cache: bool = False,
    start_pos: int = 0,
) -> torch.Tensor:
```

注意：

- q/k/v 投影后转成 `[B, H, T, D]`。
- RoPE 的 `token_positions` 不能总从 0 开始。decode 时应该是 `start_pos:start_pos+T`。
- prefill 时 `start_pos=0`。
- decode 第一步如果 prompt 长度是 `P`，新 token 的 `start_pos=P`。

#### 3. 改 `TransformerBlock.forward`

签名类似：

```python
def forward(self, x, kv_cache=None, use_cache=False, start_pos=0):
    x = self.attn(self.ln1(x), kv_cache=kv_cache, use_cache=use_cache, start_pos=start_pos) + x
    x = self.ffn(self.ln2(x)) + x
    return x
```

#### 4. 改 `TransformerLM.forward`

签名：

```python
def forward(self, input_ids, kv_caches=None, use_cache=False, start_pos=0):
```

如果 `use_cache=True`，要求 `kv_caches` 长度等于 `num_layers`。

#### 5. 新建 `cs336_basics/inference.py`

实现：

```python
def init_kv_caches(num_layers: int):
    return [LayerKVCache() for _ in range(num_layers)]

@torch.no_grad()
def generate_naive(...):
    ...

@torch.no_grad()
def generate_with_cache(...):
    ...
```

`generate_with_cache` 流程：

1. reset caches。
2. prefill prompt：`model(input_ids, kv_caches=caches, use_cache=True, start_pos=0)`。
3. sample next token。
4. decode loop：每次输入 `[B, 1]`，`start_pos=current_length`。

### 正确性测试

先别管随机采样，用 greedy：

```python
next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
```

测试：

- naive greedy 和 cache greedy 在同一模型、同一 prompt 下输出应该一致。
- 至少测 5 个 new tokens。

### 今日 benchmark

先跑：

| Variant | Context | New Tokens |
|---|---:|---:|
| naive | 128 | 32 |
| cache_cat | 128 | 32 |
| naive | 256 | 32 |
| cache_cat | 256 | 32 |

今天数据可能不好看，因为 `torch.cat` 会慢，没关系。

### 验收标准

- `generate_with_cache` 能跑。
- greedy 输出和 naive 一致。
- `metrics.md` 有第一版 benchmark。

### 降级方案

如果模型类改起来太乱，今天只在 `MultiHeadSelfAttention` 做 standalone test：

- 构造随机 x。
- full attention 输出最后一个 token。
- prefill+decode 输出最后一个 token。
- 先证明 attention cache 逻辑正确。

## Day 2：预分配 KV Cache + Benchmark Harness

### 今日目标

把 Day 1 的 `torch.cat` cache 改成 preallocated buffer，并写正式 benchmark 工具。

### 为什么要做

`torch.cat` 每步都会分配新 tensor，长文本生成会有额外内存拷贝。真实 KV cache 都是预分配或分页管理。

### 代码任务

#### 1. 扩展 `kv_cache.py`

新增：

```python
class StaticKVCache:
    def __init__(self, batch_size, num_heads, max_seq_len, head_dim, device, dtype):
        self.k = torch.empty(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.v = torch.empty(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.length = 0

    def append(self, k_new, v_new, start_pos: int):
        T = k_new.shape[2]
        self.k[:, :, start_pos:start_pos + T, :] = k_new
        self.v[:, :, start_pos:start_pos + T, :] = v_new
        self.length = max(self.length, start_pos + T)
        return self.k[:, :, :self.length, :], self.v[:, :, :self.length, :]

    def reset(self):
        self.length = 0
```

注意不要用 `zeros`，`empty` 更符合性能测试；但调试时可以先用 `zeros`。

#### 2. 实现 cache factory

因为每层 head_dim 一样，你可以在 `TransformerLM` 里或 `inference.py` 里创建：

```python
def init_static_kv_caches(model, batch_size, max_seq_len, device, dtype):
    ...
```

需要知道：

- `num_layers`
- `num_heads`
- `head_dim = d_model // num_heads`

如果模型类没存这些属性，就在 `TransformerLM.__init__` 里补：

```python
self.d_model = d_model
self.num_heads = num_heads
self.num_layers = num_layers
self.context_length = context_length
```

#### 3. 新建 `benchmark.py`

工具函数：

```python
def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    return None

def benchmark_generate(fn, warmup=3, repeats=5):
    ...
```

#### 4. 新建 `scripts/bench_inference.py`

参数建议：

```bash
uv run python scripts/bench_inference.py \
  --variant naive,cache_cat,cache_static \
  --context_lengths 128,256,512 \
  --new_tokens 64 \
  --device cpu
```

没时间可以先硬编码参数，后面再 argparse。

输出：

- 终端表格。
- `portfolio/results/minillm-inference/ablation_table.csv`。
- 追加到 `metrics.md`。

### 今日 benchmark

必须跑：

| Variant | Context | New Tokens |
|---|---:|---:|
| naive | 128 | 64 |
| cache_cat | 128 | 64 |
| cache_static | 128 | 64 |
| naive | 256 | 64 |
| cache_static | 256 | 64 |
| naive | 512 | 64 |
| cache_static | 512 | 64 |

### 验收标准

- static cache 版本能跑。
- greedy 输出仍然和 naive 一致。
- 有 CSV benchmark。
- 能看到 context 越长 cache 越有优势的趋势。如果 CPU 上趋势不明显，记录原因。

### 降级方案

如果 static cache 写不完，保留 `cache_cat`，但 benchmark harness 必须完成。Day 3 早上补 static。

## Day 3：Eval Harness + 生成质量指标

### 今日目标

构建轻量级评测工具，让项目从“代码优化”变成“实验系统”。

### 代码任务

#### 1. 新建 `cs336_basics/eval.py`

实现：

```python
def compute_nll(model, token_ids, context_length, device):
    ...

def compute_perplexity(model, token_ids, context_length, device):
    ...

def distinct_n(tokens: list[int], n: int) -> float:
    ...

def repetition_rate(tokens: list[int], n: int = 4) -> float:
    ...
```

Perplexity 注意：

- 输入 token stream。
- 每个 chunk 做 next-token prediction。
- `inputs = chunk[:-1]`
- `targets = chunk[1:]`
- 用 `F.cross_entropy(..., reduction="sum")` 累加，再除 token 数。

#### 2. 新建 `scripts/evaluate_lm.py`

功能：

- 加载模型 / checkpoint。
- 加载 token `.npy` 或 `.bin`。
- 计算 PPL。
- 可选生成若干 samples。
- 计算 distinct / repetition。
- 保存 JSONL。

输出建议：

```json
{"variant": "baseline", "ppl": 18.2, "distinct_1": 0.31, "distinct_2": 0.62, "repetition_4": 0.08}
```

#### 3. 统一 metrics 写入

新增小工具：

```python
def append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
```

### 今日实验

如果有 checkpoint：

- baseline PPL。
- naive generate samples。
- cache generate samples。
- 检查 cache 不改变输出质量。

如果没有 checkpoint：

- 先只做指标函数单元测试。
- 用 tiny random model 跑通流程。

### 验收标准

- `evaluate_lm.py` 可以运行。
- `metrics.jsonl` 至少有一条 PPL 或 generation metric。
- `samples.md` 有 naive/cache 对比样例。

### 降级方案

如果 checkpoint 加载太麻烦，先把 `eval.py` 写完并用 toy logits/token 测试；Day 4 再接真实 checkpoint。

## Day 4：SDPA Backend + FlashAttention Fallback

### 今日目标

让 attention 支持 backend 切换：

- `manual`：你原来的 attention。
- `sdpa`：`torch.nn.functional.scaled_dot_product_attention`。
- `flash`：如果环境支持就用，否则 fallback。

### 为什么有价值

这能自然展示你从手写 attention 走到现代 PyTorch kernel，对比不同 backend 的速度、显存和数值一致性。

### 代码任务

#### 1. 增加 attention backend 参数

在 `MultiHeadSelfAttention.__init__`：

```python
def __init__(self, d_model, num_heads, rope=None, attention_backend="manual"):
    self.attention_backend = attention_backend
```

在 `TransformerLM.__init__` 里向下传。

#### 2. 写 backend dispatch

```python
def attention_forward(q, k, v, mask=None, backend="manual", is_causal=True):
    if backend == "manual":
        return scaled_dot_product_attention(q, k, v, mask)
    if backend == "sdpa":
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
        )
    if backend == "flash":
        ...
```

注意：

- 你的 q/k/v 建议统一 `[B, H, T, D]`。
- SDPA 支持这个 shape。
- cache decode 阶段如果 `q.shape[-2] == 1` 且 `k` 是历史合法 cache，可以考虑 `is_causal=False`。

#### 3. 数值一致性测试

用同一个 q/k/v：

```python
manual_out = attention_forward(..., backend="manual")
sdpa_out = attention_forward(..., backend="sdpa")
torch.testing.assert_close(manual_out, sdpa_out, atol=1e-5, rtol=1e-5)
```

如果 fp16，容忍度放宽。

#### 4. FlashAttention fallback

只做可选：

```python
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
```

如果没装，不要卡住。README 写“环境未安装 flash-attn，本实验保留 backend 接口并对比 manual/SDPA”。

### 今日 benchmark

| Variant | Backend | Context | New Tokens |
|---|---|---:|---:|
| naive | manual | 512 | 64 |
| naive | sdpa | 512 | 64 |
| cache_static | manual | 512 | 64 |
| cache_static | sdpa | 512 | 64 |

### 验收标准

- backend 参数能切换。
- manual 和 SDPA 输出 close。
- 有 backend comparison 表。

### 降级方案

如果改模型参数传递太麻烦，就先在 attention 函数层面提供 `attention_forward`，脚本里 monkey patch 或单独 benchmark attention kernel。Day 5 再接模型。

## Day 5：GQA / MQA

### 今日目标

把 MHA 扩展成 GQA/MQA，并分析 KV cache memory trade-off。

### 先说清楚风险

GQA 是架构改动。如果你拿原 MHA checkpoint 直接改成 GQA，权重 shape 对不上，PPL 比较不公平。

所以今天的展示重点：

- 结构实现。
- KV cache memory 理论和实测。
- tiny model 可重新初始化跑通。
- 如果有时间，训练一个很小 GQA 模型做 PPL 对比。

### 代码任务

#### 1. 增加参数

```python
num_query_heads = num_heads
num_kv_heads = num_heads  # 默认等于 MHA
```

MHA：

```text
num_kv_heads == num_query_heads
```

GQA：

```text
num_kv_heads < num_query_heads
```

MQA：

```text
num_kv_heads == 1
```

#### 2. 修改 K/V projection 输出维度

现在可能是：

```python
self.Wk = Linear(d_model, d_model)
self.Wv = Linear(d_model, d_model)
```

GQA 版本：

```python
self.Wk = Linear(d_model, num_kv_heads * head_dim)
self.Wv = Linear(d_model, num_kv_heads * head_dim)
```

Q 仍然：

```python
self.Wq = Linear(d_model, num_query_heads * head_dim)
```

#### 3. repeat K/V

如果 q shape `[B, Hq, T, D]`，k/v shape `[B, Hkv, T, D]`：

```python
if self.num_kv_heads != self.num_query_heads:
    repeat_factor = self.num_query_heads // self.num_kv_heads
    k_for_attn = k.repeat_interleave(repeat_factor, dim=1)
    v_for_attn = v.repeat_interleave(repeat_factor, dim=1)
else:
    k_for_attn, v_for_attn = k, v
```

KV cache 里保存的是未 repeat 的 k/v：

```text
cache shape: [B, Hkv, T, D]
```

这才有显存收益。

#### 4. Memory 公式

写进报告：

```text
MHA KV cache elements = 2 * B * Hq * T * D
GQA KV cache elements = 2 * B * Hkv * T * D
memory ratio = Hkv / Hq
```

例如 Hq=8, Hkv=2，KV cache 理论下降 75%。

### 今日 benchmark

用随机 tiny model 也可以：

| Variant | Hq | Hkv | Context | KV elements | Peak Mem |
|---|---:|---:|---:|---:|---:|
| MHA | 8 | 8 | 512 | | |
| GQA | 8 | 4 | 512 | | |
| GQA | 8 | 2 | 512 | | |
| MQA | 8 | 1 | 512 | | |

如果能训练小模型，再加 PPL。

### 验收标准

- MHA/GQA/MQA forward 都能跑。
- cache 保存未 repeat 的 K/V。
- 报告里有理论 memory ratio 和实测 peak memory。

### 降级方案

如果模型整体 GQA 改不完，单独写 `GQAAttention` 模块和 standalone benchmark，也可以作为项目模块展示。

## Day 6：KV Cache Quantization + Prefix Cache

### 今日目标

做两个 serving 味道很强的小优化：

1. KV cache INT8 quantization。
2. Exact prefix cache。

这两个都做简化版，重点是能跑、能解释 trade-off。

## Part A：KV Cache Quantization

### 代码任务

#### 1. 新增量化函数

放在 `kv_cache.py` 或 `quantization.py`：

```python
def quantize_int8(x: torch.Tensor):
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 127
    x_q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
    return x_q, scale

def dequantize_int8(x_q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    return (x_q.to(dtype=torch.float32) * scale).to(dtype)
```

#### 2. Quantized cache 简化版

先别一开始就写复杂类。可以这样：

- append 时把 `k_new/v_new` quantize 后存。
- attention 前 dequantize 当前完整 cache。

这不会一定加速，因为 dequant 有开销，但能展示 memory trade-off。

#### 3. 记录质量损失

测：

- fp cache PPL。
- int8 kv cache PPL。
- greedy sample 是否明显崩。

### 今日量化 benchmark

| Variant | KV dtype | Context | New Tokens | tok/s | Peak Mem | PPL |
|---|---|---:|---:|---:|---:|---:|
| cache_static | fp32/fp16 | 512 | 64 | | | |
| cache_int8 | int8 | 512 | 64 | | | |

注意：如果没有 CUDA，peak memory 不明显，就报告理论 KV memory：

```text
fp16 KV bytes = num_elements * 2
int8 KV bytes = num_elements * 1 + scale_bytes
```

## Part B：Prefix Cache

### 简化目标

只做 exact prompt prefix cache：

- 如果两个请求 prompt 完全相同，第二次复用第一次 prefill 的 KV。
- 如果 prompt 前 N tokens 相同，可以先不做最长前缀，后面有时间再做 block prefix。

### 代码任务

#### 1. hash tokens

```python
def hash_token_ids(input_ids: torch.Tensor) -> str:
    arr = input_ids.detach().cpu().numpy().astype("int64")
    return hashlib.sha1(arr.tobytes()).hexdigest()
```

#### 2. PrefixCache 类

```python
class PrefixCache:
    def __init__(self):
        self.cache = {}

    def get(self, input_ids):
        return self.cache.get(hash_token_ids(input_ids))

    def put(self, input_ids, kv_caches, length):
        self.cache[hash_token_ids(input_ids)] = clone_kv_caches(kv_caches, length)
```

注意必须 clone，否则后续 decode 会污染 prefix cache。

#### 3. Benchmark

测同一个 prompt 连续生成两次：

| Request | Prefix Hit | Prefill Time | Decode Time |
|---|---|---:|---:|
| first | no | | |
| second | yes | | |

### 验收标准

- int8 KV cache 能跑。
- exact prefix cache 能跳过第二次 prefill。
- `metrics.md` 有 trade-off 说明。

### 降级方案

如果两个都做不完，优先做 KV quant。Prefix cache 可以只实现 hash + demo，不接完整 benchmark。

## Day 7：Chunked Prefill + 图表 + README + 简历材料

### 今日目标

收尾。不要再开大坑。今天的核心是把项目包装成能展示的成果。

如果前 6 天都顺利，再做 chunked prefill 简化版；如果不顺利，先补 benchmark 和文档。

## Part A：Chunked Prefill 简化版

### 目标

长 prompt 不一次性 prefill，而是分 chunk 喂给模型，同时持续写 KV cache。

### 实现

```python
def prefill_chunked(model, input_ids, kv_caches, chunk_size):
    start_pos = 0
    logits = None
    for i in range(0, input_ids.shape[1], chunk_size):
        chunk = input_ids[:, i:i + chunk_size]
        logits = model(chunk, kv_caches=kv_caches, use_cache=True, start_pos=start_pos)
        start_pos += chunk.shape[1]
    return logits, start_pos
```

注意 RoPE position 必须从 `start_pos` 开始。

### Benchmark

| Prefill | Prompt Len | Chunk Size | Prefill Time | Peak Mem |
|---|---:|---:|---:|---:|
| full | 1024 | - | | |
| chunked | 1024 | 128 | | |
| chunked | 1024 | 256 | | |

Chunked prefill 不一定更快，它主要是调度和显存友好。报告里要诚实写。

## Part B：画图

新建 `scripts/plot_benchmarks.py` 或直接 notebook。

三张图：

1. `context_length -> tokens/sec`
2. `context_length -> peak_memory`
3. `variant -> speedup over naive`

最小代码：

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("portfolio/results/minillm-inference/ablation_table.csv")
...
plt.savefig("portfolio/results/minillm-inference/figures/speedup_by_context.png", dpi=200)
```

## Part C：README

写：

```text
# MiniLLM Inference Lab

## What
基于 CS336 from-scratch TransformerLM 的推理优化实验系统。

## Features
- KV Cache with prefill/decode
- Static KV buffer
- Manual vs SDPA attention backend
- GQA/MQA
- INT8 KV cache quantization
- Exact prefix caching
- Lightweight Eval Harness

## Results
表格 + 图

## How to Run
训练 / 生成 / benchmark / eval 命令

## Lessons Learned
每个优化的收益和代价
```

## Part D：最终自查

必须能回答：

1. KV cache 为什么加速？
2. prefill 和 decode 区别是什么？
3. GQA 为什么省 KV cache？
4. SDPA 和手写 attention 差在哪里？
5. KV quant 为什么可能省显存但不一定加速？
6. Prefix cache 适合什么场景？
7. 为什么 PPL 可能变化？
8. 你的 benchmark 是否公平？

### Day 7 验收标准

- 至少 1 张图。
- 至少 1 个完整 ablation table。
- README 能让别人复现。
- 简历 bullet 有真实数据。
- 你能讲 10-15 分钟。

## 最终优先级：如果时间爆炸，砍什么

必须完成：

1. KV cache prefill/decode。
2. static KV buffer。
3. benchmark harness。
4. eval harness。
5. SDPA backend。

强烈建议完成：

6. GQA/MQA。
7. KV int8 quant。

可砍：

8. FlashAttention。
9. chunked prefill。
10. prefix cache 的 longest-prefix/block-level 版本。
11. 完整 PagedAttention。

不要做：

- 完整 vLLM-style scheduler。
- 完整 PagedAttention block allocator。
- DeepSeek 权重加载。
- MoE。
- MLA。

## 推荐最终简历版本

有数字后写成：

```text
MiniLLM Inference Lab：自研 Transformer 推理优化与评测系统
- 基于 CS336 from-scratch decoder-only TransformerLM，实现 KV Cache 与 prefill/decode 两阶段生成，将 context=___ 下 tokens/sec 从 ___ 提升到 ___，加速 ___x。
- 构建轻量级 LLM Eval Harness，自动评估 perplexity、distinct-n、重复率、latency、tokens/sec 和 peak memory，并生成 CSV/Markdown benchmark 报告。
- 扩展 attention 模块支持 GQA/MQA、PyTorch SDPA backend、INT8 KV cache quantization 与 exact prefix caching，量化分析显存、吞吐与生成质量 trade-off。
```

如果没有跑出漂亮 speedup，就写得稳一点：

```text
- 实现 KV Cache 增量解码和静态 KV buffer，系统 benchmark 不同 context length、attention backend 与 KV dtype 下的 latency、吞吐和显存变化。
```

## 面试讲法

建议按这个顺序讲：

1. 原始问题：naive autoregressive decoding 重复计算 prefix。
2. 解决方案：KV cache，把 prompt prefill 和 token-by-token decode 分开。
3. 第一层优化：static KV buffer，避免每步 `torch.cat`。
4. 第二层优化：GQA/MQA，减少 KV heads，降低 cache memory。
5. 第三层优化：SDPA backend，让 attention kernel 更接近真实系统。
6. 第四层优化：KV int8 和 prefix cache，面向长上下文和共享 prompt 场景。
7. 验证：Eval Harness + ablation table + 曲线。
8. 反思：不同优化的收益不是免费，有质量、显存、复杂度 trade-off。

## 你每天应该避免的坑

- 不要同时改太多模块。
- 不要一开始就做 PagedAttention。
- 不要为了跑分牺牲正确性。
- 不要只测一个 context length。
- 不要只报 speedup，不报配置。
- 不要提前写死简历数字。
- 不要把 toy implementation 说成 production-ready。

## 最小成功定义

如果一周后你只完成这些，也已经足够写简历：

- KV Cache with prefill/decode。
- Static KV buffer。
- Manual vs SDPA backend 对比。
- Eval Harness。
- Benchmark 表和两张图。

这就是一个完整、有技术含金量、能被追问也讲得住的保研项目。
