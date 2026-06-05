from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import torch

from mini_llm.model import ModelConfig, build_model
from mini_llm.tokenizer import get_tokenizer_from_vocab_merges_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB_PATH = PROJECT_ROOT / "tests" / "fixtures" / "gpt2_vocab.json"
DEFAULT_MERGES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "gpt2_merges.txt"


@dataclass
class GenerationConfig:
    """生成配置"""
    model: ModelConfig
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    do_sample: bool = True
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    device: str = "mps"
    dtype: str = "fp32"
    merges_filepath: str | Path = DEFAULT_MERGES_PATH
    vocab_filepath: str | Path = DEFAULT_VOCAB_PATH
    lm_path: str = "checkpoints/latest.pt"


@dataclass
class GenerationResult:
    """生成结果"""
    generated_ids: List[int]               # 生成的 token ids（不含 prompt）
    full_ids: List[int]                    # prompt + generated
    tokens_per_second: float = 0.0
    total_time: float = 0.0


class InferenceEngine:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = build_model(config.model).to(self.device)
        self.model.eval()
        self.tokenizer = get_tokenizer_from_vocab_merges_path(
            config.vocab_filepath,
            config.merges_filepath,
            special_tokens=["<|endoftext|>"],
        )

    
    @torch.no_grad()
    def generate_naive(self, prompt: str) -> GenerationResult:
        total_tokens = 0
        new_token = -1
        x = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.long, device=self.device).unsqueeze(0) # B, S
        while total_tokens < self.config.max_new_tokens and new_token != self.config.eos_token_id:
            logits = self.model(x)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            new_token = torch.multinomial(probs, num_samples=1).to(x.device)
            x = torch.cat([x, new_token], dim=-1)
            total_tokens += 1
        return GenerationResult(generated_ids=x[0, -total_tokens:].tolist(), full_ids=x[0].tolist())

    # def generate(self, prompt: str) -> GenerationResult:
    #     pass

    # def _prepare_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
    #     pass

    # def _generate_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
    #     pass