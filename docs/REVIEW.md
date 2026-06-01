# CS336 Proj1 代码审阅

日期：2026-04-18

## 基线情况

- 测试结果：`uv run pytest` -> `46 passed, 2 skipped`
- 审阅范围：`cs336_basics/tokenizer.py`、`cs336_basics/model.py`、`cs336_basics/train.py`
- 关注点：实现正确性、潜在行为回归，以及当前测试未覆盖到的风险

## 发现的问题

### P1：`Tokenizer.__init__` 在加入 special token 时会覆盖词表中最后一个已有条目

- 位置：`cs336_basics/tokenizer.py:379-385`
- 问题：
  - special token 是从 `len(vocab) - 1` 开始插入的。
  - 因此，当传入的 `vocab` 本身还不包含这些 special token 时，最后一个原始 token 会被覆盖，而不是在词表末尾追加新 id。
- 复现方式：
  - `Tokenizer({i: bytes([i]) for i in range(256)}, [], ["<s>"])`
  - 构造后，id `255` 会映射到 `b"<s>"`，而原来的 `b"\xff"` 会从 `encode_vocab` 中消失。
- 影响：
  - 会破坏词表内容。
  - 至少有一个原始 token 无法再被编码。
  - 会破坏 `train_bpe(...)` 之后再调用 `Tokenizer(vocab, merges, special_tokens=...)` 这一常见流程。
- 建议：
  - 从 `max(vocab) + 1` 或 `len(vocab)` 开始分配新 id。
  - 仅在 special token 尚未存在于词表时才追加。

### P1：虽然暴露了 `TrainConfig.device`，但非 CPU 训练路径实际上不可用

- 位置：
  - `cs336_basics/model.py:138-140`
  - `cs336_basics/model.py:226-227`
  - `cs336_basics/train.py:129`
  - `cs336_basics/train.py:144-149`
- 问题：
  - `RoPE.forward` 用 `device=self.device` 且固定 `float32` 来分配输出，而不是跟随 `x.device` 和 `x.dtype`。
  - `MultiHeadSelfAttention.forward` 构造因果 mask 时固定落在 CPU。
  - `cross_entropy_loss` 中的 `torch.arange(...)` 也是 CPU tensor。
  - `clip_gradients` 对 tensor reduction 使用了 `math.sqrt(...)`，需要先转成宿主端标量。
- 影响：
  - 表面上支持 `device` 配置，实际上整个训练栈基本只能在 CPU 上运行。
  - 把模型或 batch 移到 `cuda` / `mps` 后，会出现 device mismatch 或标量转换错误。
- 建议：
  - 所有派生 tensor 都从现有 tensor 推导，例：`torch.zeros_like(...)`、`torch.arange(..., device=...)`、在 `x.device` 上创建 mask。
  - 显式保持 dtype 传播正确。
  - 用 tensor 原生方式计算 norm，不要对 tensor 直接走 `math.sqrt(...)`。

### P1：默认训练流程会在第一次保存 checkpoint 时直接崩溃

- 位置：`cs336_basics/train.py:171-183`，调用点在 `cs336_basics/train.py:258-259`
- 问题：
  - 默认 checkpoint 路径是 `checkpoints/latest.pt`，但 `save_checkpoint` 没有创建父目录。
  - 训练循环会在 iteration `0` 就触发一次保存。
- 复现方式：
  - 在当前默认配置下，如果 `checkpoints/` 目录事先不存在，`torch.save(...)` 会抛出 `RuntimeError: Parent directory checkpoints does not exist.`
- 影响：
  - 一次全新的训练运行会在真正开始之前就失败。
- 建议：
  - 当 `out_path` 是路径类型时，先执行 `Path(out_path).parent.mkdir(parents=True, exist_ok=True)`。

### P2：`.bin` token 文件会被按原始字节读取，而不是按 token id 读取

- 位置：`cs336_basics/train.py:91-96`
- 问题：
  - `np.memmap(path, mode="r")` 调用时没有指定 `dtype`。
  - 因此 NumPy 会默认用 `uint8`，导致多字节 token 文件被逐字节解释，而不是按元素解释。
- 复现方式：
  - 把 `np.array([1, 256, 511], dtype=np.uint16)` 写入 `.bin` 文件，再读取时会变成 `uint8`、shape 为 `(6,)`，值为 `[1, 0, 0, 1, 255, 1]`。
- 影响：
  - 任何使用 `uint16` / `int32` 等多字节 token id 的 `.bin` 数据集，加载时都会被静默损坏。
- 建议：
  - 对 `.bin` 文件强制要求显式 `dtype`。
  - 或者在文件格式/元数据中同时保存并恢复 `dtype` 信息。

### P2：并行 BPE 计数忽略了调用者传入的 special token 集合

- 位置：`cs336_basics/tokenizer.py:62-70`
- 问题：
  - `count_word_frequencies_parallel(...)` 无论 `special_tokens` 参数是什么，都会固定按 `b"<|endoftext|>"` 去对齐 chunk 边界。
- 影响：
  - 如果语料用的是别的 special token，chunking 可能退化成单块处理。
  - 如果语料包含多个 special token，用户自定义的 special token 仍可能被 chunk 边界切开。
  - 这使得并行实现的通用性低于接口本身所表达的能力。
- 建议：
  - 根据 `special_tokens` 动态选择边界标记。
  - 如果拿不到可保证安全切分的边界 token，就退回到安全的串行实现。

## 剩余风险

- 当前单元测试对于 CPU 路径、以及作业提供的 fixture 覆盖得还不错。
- 但这些测试没有覆盖以下场景：
  - 默认目录结构下从零开始跑训练
  - 非 CPU 设备训练
  - 元素大小大于 1 字节的 `.bin` 数据集
