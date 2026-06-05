import argparse
from collections.abc import Callable
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import IO, BinaryIO, Iterable, Optional

import numpy as np
import torch

from mini_llm.model import TransformerLM

@dataclass
class TrainConfig:
    train_tokens_path: str
    valid_tokens_path: str
    vocab_size: int = 10000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 1344
    rope_theta: float = 10000.0
    batch_size: int = 32
    total_iters: int = 2000
    eval_interval: int = 200
    eval_batches: int = 20
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 200
    cosine_cycle_iters: int = 2000
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "cpu"
    checkpoint_path: str = "checkpoints/latest.pt"
    checkpoint_interval: int = 40
    dtype: str = "bf16"
    metrics_path: str = "runs/0.5b/metrics.csv"

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "beta1": betas[0], "beta2": betas[1], "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        
        for group in self.param_groups: # 这个 param_groups 是参数组，代表模型不同模块的参数可以有不同的学习率
            lr = group["lr"]  # Get the learning rate.
            beta1 = group["beta1"] # 一阶动量平滑系数
            beta2 = group["beta2"] # 二阶动量平滑系数
            lamd = group["weight_decay"] # weight decay
            epsilon = group["eps"] # 分母
            
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # 1. 取state
                state = self.state[p]  # state 是一个map，从参数param映射到字典
                t = state.get("t", 1)  # Get iteration number from the state, or initial value.
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                
                # 2. 更新动量
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2
                
                # 3. 更新参数
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= alpha_t * m / (torch.sqrt(v) + epsilon)
                
                # 4. weight decay(正则化)
                p.data -= lr * lamd * p.data
                
                # 5. 更新状态
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        
        return loss

def load_token_array(path: str) -> np.ndarray:
    # 建议先支持.npy；若你有.bin可再扩展。
    # 要求: 返回dtype可用于torch.long，且shape为(N,)。
    # return np.load(path, mmap_mode="r")
    if(path.endswith(".npy")): return np.load(path, mmap_mode="r")
    elif(path.endswith(".bin")): return np.memmap(path, mode="r")
    raise ValueError(f"Unsupported token array format: {path}")


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    
    # 1) 起点均匀采样于[0, len(dataset)-context_length)
    starts = np.random.randint(0, len(dataset) - context_length, batch_size)
    # 2) 一次性构造数据，避免逐个torch.tensor()创建中间张量
    x_data = np.array([dataset[i:i+context_length] for i in starts], dtype=np.int64)
    y_data = np.array([dataset[i+1:i+1+context_length] for i in starts], dtype=np.int64)
    # 3) 一次转为torch张量并移到device
    x = torch.from_numpy(x_data).to(device)
    y = torch.from_numpy(y_data).to(device)
    return x, y


def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # 输入形状: [B, V] 与 [B]；输出标量平均loss。
    # import pdb; pdb.set_trace()
    # 1. 普通写法，转换到概率 prob 空间，指数容易溢出！
    # logits = logits - logits.max(dim=-1, keepdim=True).values
    # exp = torch.exp(logits)
    # probs = exp[torch.arange(logits.shape[0]), targets] / exp.sum(dim=-1) # 这里我们的targets需要给最后一位取元素
    # neg_log = - torch.log(probs)
    # nlh = neg_log.mean()
    # 2. 聪明的写法，把log和指数合并一下, 在logits空间计算
    # 指数：来自于softmax
    # log：来自于交叉熵
    logits = logits - logits.max(dim=-1, keepdim=True).values
    correct_logits = logits[torch.arange(logits.shape[0]), targets] # (B)
    log_sum = torch.log(torch.exp(logits).sum(dim=-1)) # (B)
    nlh = -(correct_logits - log_sum).mean()
    return nlh

def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> float:
    # 任务拆解:
    # 1) 收集所有非None梯度
    grads = []
    for param in parameters:
        if param.grad is None:
            continue
        grads.append(param.grad)

    # 2) 计算全局L2 norm
    l2_norm = math.sqrt(sum((g.data**2).sum().item() for g in grads))

    # 3) 若norm > max_l2_norm，按同一比例缩放每个梯度
    if l2_norm > max_l2_norm:
        for g in grads:
            g.data = g.data * max_l2_norm / (l2_norm + 1e-6)

    return l2_norm


def get_grad_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squared_norm = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        squared_norm += float((param.grad.data**2).sum().item())
    return math.sqrt(squared_norm)


def safe_exp(value: float) -> float:
    if value > 100:
        return float("inf")
    return math.exp(value)


def format_metrics(metrics: dict[str, float | int | str]) -> str:
    return " ".join(
        f"{key}={value:.4e}" if isinstance(value, float) and (abs(value) >= 1e4 or 0 < abs(value) < 1e-3)
        else f"{key}={value:.4f}" if isinstance(value, float)
        else f"{key}={value}"
        for key, value in metrics.items()
    )


def get_autocast_dtype(dtype: str) -> torch.dtype | None:
    if dtype == "fp32":
        return None
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


def get_autocast_device(device: str) -> str:
    return "cuda" if device.startswith("cuda") else "cpu"


def should_use_autocast(cfg: TrainConfig) -> bool:
    return cfg.dtype != "fp32"


def forward_loss(
    model: TransformerLM,
    batch: torch.Tensor,
    targets: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    # 优化点：使用autocast自动选择dtype，使用bf16可以加速计算
    autocast_dtype = get_autocast_dtype(cfg.dtype)
    with torch.autocast(
        device_type=get_autocast_device(cfg.device),
        dtype=autocast_dtype or torch.float32,
        enabled=should_use_autocast(cfg),
    ):
        logits = model(batch)

    logits = logits.float().reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    return cross_entropy_loss(logits, targets)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    # 任务拆解:
    # 1) it < warmup_iters: 线性warmup到max_learning_rate
    if it < warmup_iters:
        lr = it / warmup_iters * max_learning_rate
    # 2) warmup后到cosine_cycle_iters: cosine从max到min
    elif it >= warmup_iters and it <= cosine_cycle_iters:
        lr = min_learning_rate + 0.5*(1 + math.cos((it - warmup_iters)*math.pi / (cosine_cycle_iters - warmup_iters)))*(max_learning_rate - min_learning_rate)    
    # 3) it > cosine_cycle_iters: 固定min_learning_rate
    else: lr = min_learning_rate
    return lr


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out_path: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    # 保存model/optimizer/iteration三项状态。
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'iteration': iteration
    }
    # 自动创建目录
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)


def load_checkpoint(
    src_path: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    # 正确恢复model与optimizer状态，并返回iteration。
    checkpoint = torch.load(src_path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint['iteration']


def evaluate(
    model: TransformerLM,
    valid_data: np.ndarray,
    cfg: TrainConfig,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(cfg.eval_batches):
            x, y = get_batch(valid_data, cfg.batch_size, cfg.context_length, cfg.device)
            loss = forward_loss(model, x, y, cfg)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def build_model(cfg: TrainConfig) -> TransformerLM:
    model = TransformerLM(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        num_layers=cfg.num_layers,
        rope_theta=cfg.rope_theta,
    )
    return model.to(cfg.device)

def write_metrics(metrics: dict[str, float | int | str], path:str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not out_path.exists()
    with out_path.open("a", encoding="utf-8") as f:
        # csv 格式： 列名，值
        if should_write_header:
            f.write(",".join(metrics.keys()) + "\n")
        f.write(",".join(str(value) for value in metrics.values()) + "\n")

def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    model = build_model(cfg)
    opt = AdamW(model.parameters(), lr=cfg.max_learning_rate)

    train_data = load_token_array(cfg.train_tokens_path)
    eval_data = load_token_array(cfg.valid_tokens_path)
    tokens_per_iter = cfg.batch_size * cfg.context_length
    train_start_time = time.perf_counter()

    print(
        format_metrics(
        {
            "params": sum(param.numel() for param in model.parameters()),
            "train_tokens": len(train_data),
            "valid_tokens": len(eval_data),
            "tokens_per_iter": tokens_per_iter,
            "dtype": str(cfg.dtype),
        }),
        flush=True,
    )

    for it in range(cfg.total_iters):
        step_start_time = time.perf_counter()
        # 优化器每次循环初始化
        opt.zero_grad() # 首先记得给优化器梯度清零
        # 应用余弦调度
        lr = get_lr_cosine_schedule(it, cfg.max_learning_rate, cfg.min_learning_rate, cfg.warmup_iters, cfg.cosine_cycle_iters)
        for group in opt.param_groups:
            group['lr'] = lr
        # 取一个小batch
        batch, targets = get_batch(train_data, cfg.batch_size, cfg.context_length, cfg.device)
        
        loss = forward_loss(model, batch, targets, cfg)
        loss.backward()
        
        grad_norm = clip_gradients(model.parameters(), cfg.max_grad_norm)
        opt.step()
        step_sec = time.perf_counter() - step_start_time
        tokens_seen = (it + 1) * tokens_per_iter
        elapsed_sec = time.perf_counter() - train_start_time
        tokens_per_sec = tokens_seen / elapsed_sec if elapsed_sec > 0 else 0.0

        if it % cfg.eval_interval == 0:
            val_loss = evaluate(model, eval_data, cfg)
            metrics = {
                "it": it,
                "tokens_seen": tokens_seen,
                "train_loss": loss.item(),
                "train_ppl": safe_exp(loss.item()),
                "val_loss": val_loss,
                "val_ppl": safe_exp(val_loss),
                "lr": lr,
                "grad_norm": grad_norm,
                "step_sec": step_sec,
                "elapsed_sec": elapsed_sec,
                "tokens_per_sec": tokens_per_sec,
            }
            print(format_metrics(metrics))
            write_metrics(metrics, cfg.metrics_path)
        if it % cfg.checkpoint_interval == 0:
            save_checkpoint(model, opt, it, cfg.checkpoint_path)

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Training entry for mini-llm")
    parser.add_argument("--train_tokens_path", type=str, required=True)
    parser.add_argument("--valid_tokens_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_iters", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--checkpoint_interval", type=int, default=40)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--max_learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=200)
    parser.add_argument("--cosine_cycle_iters", type=int, default=2000)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--metrics_path", type=str, default="runs/0.5b/metrics.csv")

    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    config = parse_args()
    train(config)
