# Assignment1 补全与复习路径

目标: 先完成 tokenizer 的增量 BPE 主线，再完成 model 的 Transformer 主线，最后用 pytest 分阶段验证。

## 1. Tokenizer 路径（先做）

### 1.1 你要先建立的心智模型

1. 语料状态: `word_freqs` 记录当前 token 序列词频。
2. 统计状态: `pair_freq` 记录 pair 全局频率，`pair_to_words` 是 pair 到 word 的倒排索引。
3. 选择策略: 用 `pq` 找最高频 pair，使用懒惰删除跳过陈旧项。
4. 增量更新: 每轮只处理包含 `best_pair` 的受影响词，避免全量重算。

### 1.2 补全顺序（5 个 TODO）

1. [cs336_basics/tokenizer.py](cs336_basics/tokenizer.py#L125): `count_pairs_in_word`
2. [cs336_basics/tokenizer.py](cs336_basics/tokenizer.py#L165): `pop_valid_best_pair`
3. [cs336_basics/tokenizer.py](cs336_basics/tokenizer.py#L142): `build_pair_statistics` 内局部统计接回全局
4. [cs336_basics/tokenizer.py](cs336_basics/tokenizer.py#L152): `push_pair_to_pq` 的过滤与 tie-break
5. [cs336_basics/tokenizer.py](cs336_basics/tokenizer.py#L283): `train_bpe` step 5 的增量加回

建议顺序原因: 1/2 是基础工具，3/4 是结构接线，5 是主循环核心。

### 1.3 关键思考问题

1. 为什么 `count_pairs_in_word` 不能只记录“是否出现”而要记录“出现次数”？
2. 为什么 `pop_valid_best_pair` 必须校验 `pair_freq` 当前值？
3. 为什么 step 5 只更新 `changed_pairs` 就够了？

## 2. Model 路径（后做）

### 2.1 先读结构，再补函数

1. 总入口: [cs336_basics/model.py](cs336_basics/model.py#L171)
2. Block 结构: [cs336_basics/model.py](cs336_basics/model.py#L159)
3. Attention 主体: [cs336_basics/model.py](cs336_basics/model.py#L138)

### 2.2 补全顺序（从小到大）

1. [cs336_basics/model.py](cs336_basics/model.py#L32): `Linear.forward`
2. [cs336_basics/model.py](cs336_basics/model.py#L50): `Embedding.forward`
3. [cs336_basics/model.py](cs336_basics/model.py#L70): `SiLU`
4. [cs336_basics/model.py](cs336_basics/model.py#L65): `RMSNorm.forward`
5. [cs336_basics/model.py](cs336_basics/model.py#L86): `SwiGLU.forward`
6. [cs336_basics/model.py](cs336_basics/model.py#L126): `RoPE.forward`
7. [cs336_basics/model.py](cs336_basics/model.py#L130): `softmax`
8. [cs336_basics/model.py](cs336_basics/model.py#L135): `scaled_dot_product_attention`
9. [cs336_basics/model.py](cs336_basics/model.py#L154): `MultiHeadSelfAttention.forward`
10. [cs336_basics/model.py](cs336_basics/model.py#L167): `TransformerBlock.forward`
11. [cs336_basics/model.py](cs336_basics/model.py#L180): `TransformerLM.forward`

### 2.3 关键公式与结构

1. RMSNorm:

$$
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon},\quad y=\frac{x\odot g}{\mathrm{RMS}(x)}
$$

2. Attention:

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)V
$$

3. RoPE（偶奇维旋转）:

$$
\begin{bmatrix}x_{2i}'\\x_{2i+1}'\end{bmatrix}=\begin{bmatrix}\cos\theta_i&-\sin\theta_i\\\sin\theta_i&\cos\theta_i\end{bmatrix}\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}
$$

4. Pre-Norm block:

$$
x \leftarrow x + \mathrm{Attn}(\mathrm{Norm}(x)),\quad x \leftarrow x + \mathrm{FFN}(\mathrm{Norm}(x))
$$

## 3. Pytest 分步验证路径

### 3.1 Tokenizer 先验收

1. 先跑 BPE 训练逻辑:

```bash
uv run pytest tests/test_train_bpe.py -q
```

2. 再跑 tokenizer 编解码一致性:

```bash
uv run pytest tests/test_tokenizer.py -q
```

### 3.2 Model 再验收

1. 先跑基础算子与模块:

```bash
uv run pytest tests/test_model.py -q -k "linear or embedding or silu or rmsnorm or rope or scaled_dot_product_attention or swiglu"
```

2. 再跑 attention/block/lm:

```bash
uv run pytest tests/test_model.py -q -k "multihead_self_attention or transformer_block or transformer_lm"
```

3. 最后全量:

```bash
uv run pytest tests -q
```

## 4. 怎么看代码、怎么思考

1. 每次只跟一个状态变量走完一轮。
: tokenizer 先跟 `word_freqs`，model 先跟 `x` 的形状。
2. 每写完一个 TODO，立即跑对应最小测试，不要攒一堆再测。
3. 遇到错误先做“形状检查/不变量检查”，再看数值是否一致。

## 5. 最终 Takeaway

1. Tokenizer 的核心不是“合并”本身，而是“如何只更新受影响区域”。
2. Transformer 的核心不是模块数量，而是“形状流 + 规范化位置 + 残差连接”。
3. 工程上最关键的是: 写一小步、测一小步、保证状态一致。
