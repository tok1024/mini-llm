from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import torch

from mini_llm.model import ModelConfig, build_model
from mini_llm.tokenizer import get_tokenizer_from_vocab_merges_path
from mini_llm.kv_cache import SimpleKVCache

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
    generated_text: str = ""
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
        self.kv_cache = SimpleKVCache(config.model.num_layers)

    def _sample(self, logits: torch.Tensor) -> int:
        # logits: (B, vocab_size)
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).to(logits.device).item()
        return int(next_token)

    # 预填充kvcache，返回next_token,用于后续生成
    def prefill_kvcache(self, input_ids: List[int]) -> int:
        x = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0) # B, S
        """用输入的prompt填充kv cache"""
        logits = self.model(x, start_pos=0, kv_cache=self.kv_cache)
        new_token = self._sample(logits[:, -1, :])
        return new_token
    
    @torch.no_grad()
    def generate_naive(self, input_ids: List[int]) -> GenerationResult:
        total_tokens = 0
        new_token = -1
        x = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0) # B, S
        while total_tokens < self.config.max_new_tokens and new_token != self.config.eos_token_id:
            logits = self.model(x)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            new_token = torch.multinomial(probs, num_samples=1).to(x.device)
            x = torch.cat([x, new_token], dim=-1)
            total_tokens += 1
        return GenerationResult(generated_ids=x[0, -total_tokens:].tolist(), full_ids=x[0].tolist(), generated_text=self.tokenizer.decode(x[0, -total_tokens:].tolist()))

    @torch.no_grad()
    def generate_simple_kvcache(self, input_ids: List[int]) -> GenerationResult:
        self.kv_cache.reset()
        if self.config.max_new_tokens <= 0:
            return GenerationResult(generated_ids=[], full_ids=list(input_ids), generated_text="")

        new_token = self.prefill_kvcache(input_ids)
        x = torch.tensor([[new_token]], dtype=torch.long, device=self.device) # 1,
        generated_ids = [new_token]
        full_ids = input_ids + [new_token]
        total_tokens = 1
        
        while total_tokens < self.config.max_new_tokens and new_token != self.config.eos_token_id:
            start_pos = self.kv_cache.get_length()
            logits = self.model(x, start_pos=start_pos, kv_cache=self.kv_cache)
            new_token = self._sample(logits[:, -1, :])
            total_tokens += 1
            x = torch.tensor([[new_token]], dtype=torch.long, device=self.device) # 1,
            generated_ids.append(new_token)
            full_ids.append(new_token)
        return GenerationResult(generated_ids=generated_ids, full_ids=full_ids, generated_text=self.tokenizer.decode(generated_ids))

    def generate(self, prompt: str, method: str = "naive") -> GenerationResult:
        input_ids = self.tokenizer.encode(prompt)
        if method == "naive":
            return self.generate_naive(input_ids)
        elif method == "simple_kvcache":
            return self.generate_simple_kvcache(input_ids)
        else:
            raise ValueError(f"Invalid generation method: {method}")

    # def _prepare_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
    #     pass

    # def _generate_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
    #     pass