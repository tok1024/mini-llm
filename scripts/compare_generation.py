from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mini_llm.inference import GenerationConfig, InferenceEngine
from mini_llm.model import ModelConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "latest.pt"


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


def set_generation_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def reset_engine_state(engine: InferenceEngine) -> None:
    if hasattr(engine, "kv_cache"):
        engine.kv_cache.reset()
    if hasattr(engine, "total_tokens"):
        delattr(engine, "total_tokens")


def run_generation(
    engine: InferenceEngine,
    prompt: str,
    method: str,
    seed: int,
) -> list[int]:
    reset_engine_state(engine)
    set_generation_seed(seed, engine.device)
    result = engine.generate(prompt, method=method)
    return list(result.generated_ids)


def compare_sequences(name_to_ids: dict[str, list[int]]) -> bool:
    baseline_name, baseline_ids = next(iter(name_to_ids.items()))
    ok = True

    for name, ids in list(name_to_ids.items())[1:]:
        if ids == baseline_ids:
            print(f"[match] {baseline_name} == {name}")
            continue

        ok = False
        first_diff = next(
            (idx for idx, pair in enumerate(zip(baseline_ids, ids)) if pair[0] != pair[1]),
            min(len(baseline_ids), len(ids)),
        )
        print(f"[mismatch] {baseline_name} != {name}")
        print(f"  first_diff_index={first_diff}")
        print(f"  {baseline_name}={baseline_ids}")
        print(f"  {name}={ids}")

    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare InferenceEngine.generate methods.")
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--prompt", type=str, default="i like computer")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--eos_token_id", type=int, default=50256)
    parser.add_argument("--seed", type=int, default=42)
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
    methods = args.methods or ["naive", "simple_kvcache"]

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

    results = {
        method: run_generation(engine, args.prompt, method, args.seed)
        for method in methods
    }

    print(f"checkpoint={args.checkpoint_path}")
    print(f"prompt={args.prompt!r}")
    print(f"methods={methods}")
    for name, ids in results.items():
        print(f"{name}_ids={ids}")
        print(f"{name}_text={engine.tokenizer.decode(ids)!r}")

    if not compare_sequences(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()