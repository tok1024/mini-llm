#!/usr/bin/env bash
set -euo pipefail

cd /root/mini-llm

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

RUN_NAME=${RUN_NAME:-0.5b}
RUN_DIR=${RUN_DIR:-runs/${RUN_NAME}}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/${RUN_NAME}}
LOG_PATH=${LOG_PATH:-${RUN_DIR}/train.log}

mkdir -p "${RUN_DIR}" "${CHECKPOINT_DIR}"

{
  echo "run_name=${RUN_NAME}"
  echo "run_dir=${RUN_DIR}"
  echo "checkpoint_dir=${CHECKPOINT_DIR}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "started_at=$(date -Is)"

  uv run python -m cs336_basics.train \
    --train_tokens_path data/ts-train.npy \
    --valid_tokens_path data/ts-val.npy \
    --vocab_size 50257 \
    --context_length 1024 \
    --d_model 1152 \
    --num_layers 24 \
    --num_heads 18 \
    --d_ff 3072 \
    --rope_theta 10000 \
    --batch_size 4 \
    --total_iters 100000 \
    --eval_interval 500 \
    --eval_batches 20 \
    --checkpoint_interval 1000 \
    --max_learning_rate 3e-4 \
    --min_learning_rate 3e-5 \
    --warmup_iters 2000 \
    --cosine_cycle_iters 100000 \
    --max_grad_norm 1.0 \
    --seed 42 \
    --dtype bf16 \
    --device cuda \
    --checkpoint_path "${CHECKPOINT_DIR}/latest.pt"

  echo "finished_at=$(date -Is)"
} | tee "${LOG_PATH}"
