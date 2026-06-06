from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from mini_llm.inference import GenerationConfig, InferenceEngine
from mini_llm.model import ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "latest.pt"


@dataclass
class BenchResult:
    method: str
    prompt_tokens: int
    generated_tokens: int
    warmup_runs: int
    timed_runs: int
    avg_ms: float
    tokens_per_second: float


def build_default_model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=50257,
        context_length=256,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=2048,
        rope_theta=10000.0,
    )


def load_checkpoint_into_engine(engine: InferenceEngine, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=engine.device)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    engine.model.load_state_dict(state)
    engine.model.eval()


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def set_generation_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def reset_engine_state(engine: InferenceEngine) -> None:
    if hasattr(engine, "kv_cache"):
        engine.kv_cache.reset()


def run_once(engine: InferenceEngine, prompt: str, method: str, seed: int) -> int:
    reset_engine_state(engine)
    set_generation_seed(seed, engine.device)
    result = engine.generate(prompt, method=method)
    return len(result.generated_ids)


def benchmark_method(
    engine: InferenceEngine,
    prompt: str,
    method: str,
    seed: int,
    warmup_runs: int,
    timed_runs: int,
) -> BenchResult:
    prompt_tokens = len(engine.tokenizer.encode(prompt))

    for idx in range(warmup_runs):
        run_once(engine, prompt, method, seed + idx)
    sync_device(engine.device)

    generated_tokens = 0
    start = time.perf_counter()
    for idx in range(timed_runs):
        generated_tokens = run_once(engine, prompt, method, seed + warmup_runs + idx)
    sync_device(engine.device)
    elapsed = time.perf_counter() - start

    avg_ms = elapsed * 1000 / timed_runs
    tokens_per_second = generated_tokens * timed_runs / elapsed if elapsed > 0 else 0.0
    return BenchResult(
        method=method,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        avg_ms=avg_ms,
        tokens_per_second=tokens_per_second,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark InferenceEngine.generate methods.")
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--prompt", type=str, default="i like computer")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--eos_token_id", type=int, default=50256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--timed_runs", type=int, default=3)
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        default=None,
        help="Generation method passed to InferenceEngine.generate. Can be repeated.",
    )
    parser.add_argument("--vocab_path", type=Path, default=None)
    parser.add_argument("--merges_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = args.methods or ["simple_kvcache", "paged_kvcache"]

    config_kwargs = {
        "model": build_default_model_config(),
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": args.eos_token_id,
        "device": args.device,
        "lm_path": str(args.checkpoint_path),
    }
    if args.vocab_path is not None:
        config_kwargs["vocab_filepath"] = args.vocab_path
    if args.merges_path is not None:
        config_kwargs["merges_filepath"] = args.merges_path

    engine = InferenceEngine(GenerationConfig(**config_kwargs))
    load_checkpoint_into_engine(engine, args.checkpoint_path)

    print(f"checkpoint={args.checkpoint_path}")
    print(f"device={engine.device}")
    print(f"prompt={args.prompt!r}")
    print(f"max_new_tokens={args.max_new_tokens}")
    print(f"methods={methods}")
    print("method,prompt_tokens,generated_tokens,warmup_runs,timed_runs,avg_ms,tokens_per_second")

    for method in methods:
        result = benchmark_method(
            engine=engine,
            prompt=args.prompt,
            method=method,
            seed=args.seed,
            warmup_runs=args.warmup_runs,
            timed_runs=args.timed_runs,
        )
        print(
            f"{result.method},{result.prompt_tokens},{result.generated_tokens},"
            f"{result.warmup_runs},{result.timed_runs},{result.avg_ms:.3f},{result.tokens_per_second:.3f}"
        )


if __name__ == "__main__":
    main()