import torch
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class LayerKVCache:
    """单层 KV Cache"""
    k: Optional[torch.Tensor] = None   # [B, num_heads, current_len, head_dim]
    v: Optional[torch.Tensor] = None   # [B, num_heads, current_len, head_dim]
    length: int = 0                     # 当前已缓存的 token 数量

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """追加新的 K/V"""
        assert k_new.shape == v_new.shape, "k_new and v_new shape mismatch"
        assert k_new.dim() == 4, "k_new should be [B, H, T_new, D]"

        if self.k is None or self.v is None:
            # 第一次 append
            self.k = k_new
            self.v = v_new
        else:
            # 后续 append
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)

        self.length += k_new.shape[2]

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回当前全部 K 和 V"""
        assert self.k is not None and self.v is not None, "Cache is empty"
        return self.k, self.v

    def reset(self):
        """清空 cache 用于新 sequence"""
        self.k = None
        self.v = None
        self.length = 0


class SimpleKVCache:
    """多层 KV Cache 容器"""
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.layers: List[LayerKVCache] = [LayerKVCache() for _ in range(num_layers)]

    def append(self, k_new_list: List[torch.Tensor], v_new_list: List[torch.Tensor]):
        """给所有层追加 K/V（通常在 TransformerLM forward 中调用）"""
        assert len(k_new_list) == self.num_layers
        assert len(v_new_list) == self.num_layers

        for i in range(self.num_layers):
            self.layers[i].append(k_new_list[i], v_new_list[i])

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取某一层的完整 K/V"""
        return self.layers[layer_idx].get()

    def reset(self):
        """重置所有层"""
        for layer in self.layers:
            layer.reset()

    def get_length(self) -> int:
        """返回当前缓存长度（所有层应该一致）"""
        return self.layers[0].length if self.layers else 0


# ==================== 后续扩展 ====================

class StaticKVCache:
    """预分配版本（下一阶段实现）"""
    pass


class PagedKVCache:
    """分页版本（最终目标）"""
    pass