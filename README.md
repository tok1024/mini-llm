# MiniLLM

MiniLLM 是一个 from-scratch decoder-only Transformer 训练与推理实验项目。

当前项目包含：

- BPE tokenizer
- TransformerLM
- training loop
- checkpoint / metrics
- simplified PagedAttention 学习与实现指南

## Setup

本项目使用 `uv` 管理环境。安装方式见 [uv 官方文档](https://docs.astral.sh/uv/)。

运行项目代码：

```sh
uv run <python_file_path>
```

运行测试：

```sh
uv run pytest
```

## Training

训练脚本示例：

```sh
bash scripts/train_0.5b.sh
```

训练入口：

```sh
uv run python -m mini_llm.train
```

## Data

可使用 TinyStories 和 OpenWebText 子集作为训练数据：

```sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Docs

- `docs/STUDY_GUIDE.md`
- `docs/paged_attention_implementation_guide.md`

## Acknowledgements

本项目起源于公开课 Transformer 训练作业，并在此基础上扩展训练和推理实验。