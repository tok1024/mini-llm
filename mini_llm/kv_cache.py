import torch
from typing import Optional, List, Tuple, Callable, Any
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

class StaticLayerKVCache:
    """单层预分配 KV Cache"""
    def __init__(
        self,
        max_batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        self.max_batch_size = max_batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.k = torch.empty(max_batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.v = torch.empty(max_batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
        self.length = 0
        self.batch_size = 0

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """把新的 K/V 写入预分配 buffer"""
        assert k_new.shape == v_new.shape, "k_new and v_new shape mismatch"
        assert k_new.dim() == 4, "k_new should be [B, H, T_new, D]"

        batch_size, num_heads, new_len, head_dim = k_new.shape
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}")
        if num_heads != self.num_heads:
            raise ValueError(f"num_heads mismatch: got {num_heads}, expected {self.num_heads}")
        if head_dim != self.head_dim:
            raise ValueError(f"head_dim mismatch: got {head_dim}, expected {self.head_dim}")

        end = self.length + new_len
        if end > self.max_seq_len:
            raise ValueError(f"KV cache length {end} exceeds max_seq_len {self.max_seq_len}")

        # 写切片
        self.k[:batch_size, :, self.length:end, :].copy_(k_new)
        self.v[:batch_size, :, self.length:end, :].copy_(v_new)
        self.length = end
        self.batch_size = batch_size

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回当前有效 K/V 视图"""
        assert self.length > 0, "Cache is empty"
        return (
            self.k[:self.batch_size, :, :self.length, :],
            self.v[:self.batch_size, :, :self.length, :],
        )

    def reset(self):
        """清空 cache 的有效长度，不清零 buffer"""
        self.length = 0
        self.batch_size = 0


class StaticKVCache:
    """多层预分配 KV Cache 容器"""
    def __init__(
        self,
        num_layers: int,
        max_batch_size: int,
        num_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.layers: List[StaticLayerKVCache] = [
            StaticLayerKVCache(
                max_batch_size=max_batch_size,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

    def append(self, k_new_list: List[torch.Tensor], v_new_list: List[torch.Tensor]):
        """给所有层写入 K/V"""
        assert len(k_new_list) == self.num_layers
        assert len(v_new_list) == self.num_layers

        for i in range(self.num_layers):
            self.layers[i].append(k_new_list[i], v_new_list[i])

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取某一层的完整有效 K/V"""
        return self.layers[layer_idx].get()

    def reset(self):
        """重置所有层"""
        for layer in self.layers:
            layer.reset()

    def get_length(self) -> int:
        """返回当前缓存长度（所有层应该一致）"""
        return self.layers[0].length if self.layers else 0


@dataclass
class PagedLayerState:
    """Paged attention 所需的单层 cache 状态"""
    k_pool: torch.Tensor
    v_pool: torch.Tensor
    block_table: torch.Tensor
    physical_to_logical: torch.Tensor
    length: int
    block_size: int

    def build_block_mask(self, query_len: int, start_pos: int, mask_mod: Callable[..., Any] | None = None) -> Any:
        """构造基于 physical KV slot 的 causal BlockMask。"""
        from torch.nn.attention.flex_attention import create_block_mask

        num_heads = self.k_pool.shape[1]
        physical_kv_len = self.physical_to_logical.shape[0]

        def default_mask_mod(batch: torch.Tensor, head: torch.Tensor, query_pos: torch.Tensor, kv_pos: torch.Tensor) -> torch.Tensor:
            _ = (batch, head)
            logical_key_pos = self.physical_to_logical[kv_pos]
            logical_query_pos = start_pos + query_pos
            return (logical_key_pos >= 0) & (logical_key_pos <= logical_query_pos)

        return create_block_mask(
            mask_mod or default_mask_mod,
            B=1,
            H=num_heads,
            Q_LEN=query_len,
            KV_LEN=physical_kv_len,
            device=str(self.k_pool.device),
        )

    def as_flex_inputs(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 physical KV pool 和 Tensor page table。"""
        return self.k_pool, self.v_pool, self.block_table


class PagedLayerKVCache:
    """单层分页 KV Cache 骨架"""
    is_paged = True

    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        block_size: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = head_dim
        # 这里的直觉是：block的维度和sequence对齐
        self.k_pool = torch.zeros(num_blocks, num_heads, block_size, head_dim, device=device, dtype=dtype)
        self.v_pool = torch.zeros(num_blocks, num_heads, block_size, head_dim, device=device, dtype=dtype)
        self.free_blocks: List[int] = list(range(num_blocks))
        self.block_table: List[int] = []
        self.block_table_tensor = torch.full((num_blocks,), -1, device=device, dtype=torch.int32)
        self.physical_to_logical = torch.full((num_blocks * block_size,), -1, device=device, dtype=torch.int32)
        self.length = 0
        self.batch_size = 0

    def allocate_block(self) -> int:
        """申请一个物理 block"""
        if not self.free_blocks:
            raise RuntimeError("No free KV blocks")
        return self.free_blocks.pop()

    def append(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """把新的 K/V 写入分页 pool。TODO: 补完整 token -> block/offset 写入逻辑"""
        assert k_new.shape == v_new.shape, "k_new and v_new shape mismatch"
        assert k_new.dim() == 4, "k_new should be [B, H, T_new, D]"

        batch_size, num_heads, new_len, head_dim = k_new.shape
        if batch_size != 1:
            raise NotImplementedError("PagedLayerKVCache skeleton only supports batch_size=1")
        if num_heads != self.num_heads:
            raise ValueError(f"num_heads mismatch: got {num_heads}, expected {self.num_heads}")
        if head_dim != self.head_dim:
            raise ValueError(f"head_dim mismatch: got {head_dim}, expected {self.head_dim}")

        self.batch_size = batch_size

        for token_offset in range(new_len):
            token_pos = self.length
            logical_block_idx = token_pos // self.block_size
            offset_in_block = token_pos % self.block_size

            if logical_block_idx == len(self.block_table):
                physical_block_idx = self.allocate_block()
                self.block_table.append(physical_block_idx)
                self.block_table_tensor[logical_block_idx] = physical_block_idx

            physical_block_idx = self.block_table[logical_block_idx]
            physical_pos = physical_block_idx * self.block_size + offset_in_block

            #   k_new[0, :, token_offset, :] -> [H, D]
            #   self.k_pool[physical_block_idx, :, offset_in_block, :] -> [H, D]
            self.k_pool[physical_block_idx, :, offset_in_block, :] = k_new[0, :, token_offset, :]
            self.v_pool[physical_block_idx, :, offset_in_block, :] = v_new[0, :, token_offset, :]
            self.physical_to_logical[physical_pos] = token_pos
            self.length += 1

    def get_paged_state(self) -> PagedLayerState:
        """返回 paged attention 需要的元信息，不返回连续 K/V"""
        return PagedLayerState(
            k_pool=self.k_pool,
            v_pool=self.v_pool,
            block_table=self.block_table_tensor,
            physical_to_logical=self.physical_to_logical,
            length=self.length,
            block_size=self.block_size,
        )

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Paged attention 不应该走连续 K/V 接口"""
        raise RuntimeError("PagedLayerKVCache does not expose contiguous K/V. Use get_paged_state().")

    def reset(self):
        """释放当前 sequence 占用的物理 blocks"""
        self.free_blocks.extend(reversed(self.block_table))
        self.block_table = []
        self.block_table_tensor.fill_(-1)
        self.physical_to_logical.fill_(-1)
        self.length = 0
        self.batch_size = 0


class PagedKVCache:
    """多层分页 KV Cache 容器骨架"""
    is_paged = True

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        num_heads: int,
        block_size: int,
        head_dim: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.layers: List[PagedLayerKVCache] = [
            PagedLayerKVCache(
                num_blocks=num_blocks,
                num_heads=num_heads,
                block_size=block_size,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ]

    def reset(self):
        """重置所有层"""
        for layer in self.layers:
            layer.reset()

    def get_length(self) -> int:
        """返回当前缓存长度（所有层应该一致）"""
        return self.layers[0].length if self.layers else 0
