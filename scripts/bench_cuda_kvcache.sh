#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/checkpoints/latest.pt}"
PROMPT="${PROMPT:-i like computer}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
WARMUP_RUNS="${WARMUP_RUNS:-2}"
TIMED_RUNS="${TIMED_RUNS:-5}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"

cd "${PROJECT_ROOT}"

if [[ "${DEVICE}" == "cuda" ]]; then
  uv run python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Set DEVICE=cpu to smoke test locally.")
print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY
fi

uv run python scripts/bench_inference.py \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --prompt "${PROMPT}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --warmup_runs "${WARMUP_RUNS}" \
  --timed_runs "${TIMED_RUNS}" \
  --method naive \
  --method simple_kvcache \
  --method static_kvcache \
  --method paged_kvcache