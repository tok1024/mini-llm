from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import math
from einops import einsum, rearrange, reduce
from mini_llm.kv_cache import LayerKVCache, SimpleKVCache, PagedLayerKVCache

END_TOKEN = 50256

@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 1152
    num_layers: int = 24
    num_heads: int = 18
    head_dim: Optional[int] = None
    d_ff: int = 3072
    rope_theta: float = 10000.0

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        # 初始化权重
        super().__init__()
        # 权重 W 存储为原矩阵(dout, din)，而非转置. 使用时用x乘w的转置 x @ w.T
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype)) 
        self.init_param()
        
        
    def init_param(self):
        
        mean = 0
        d_out, d_in = self.weight.shape
        std = math.sqrt(2/(d_in + d_out))
        nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=-3*std, b=3*std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # pytorch中必须把向量视为行向量
        # return x @ self.weight.T
        # 但是对于einsum就无所谓
        # (复习-挖空): 补全线性层的einsum维度映射
        return einsum(self.weight, x, 'o i, b s i -> b s o')
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.d_model = embedding_dim
        self.init_embeddings()
        
    def init_embeddings(self):
        mean = 0
        std = 1
        nn.init.trunc_normal_(self.embeddings, mean=mean, std=std, a=-3, b=3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 根据token_ids做embedding查表
        return self.embeddings[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.gain = nn.Parameter(torch.randn(d_model, device=device, dtype=dtype) / math.sqrt(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        in_dtype = x.dtype
        x = x.float()

        # 要求: 结果转回输入dtype
        rms = torch.sqrt(((x**2).mean(dim=-1, keepdim=True) + self.eps))
        out = x * self.gain / rms
        return out.to(in_dtype)
    
def SiLU(x: torch.Tensor):
    # 写出SiLU定义
    return torch.sigmoid(x) * x
    
class SwiGLU(nn.Module):
    def __init__(self, d_model, dff=None):
        super().__init__()
        if not dff:
            self.dff = 8  * d_model // 3
        else:
            self.dff = dff
        self.w1 = Linear(d_model, self.dff)
        self.w2 = Linear(self.dff, d_model)
        self.w3 = Linear(d_model, self.dff)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 补全SwiGLU前向
        # 提示: 两个分支 + 门控 + 输出投影
        gated = SiLU(self.w1(x))
        projected = self.w3(x)
        return self.w2(gated * projected)
    
class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        self.d_k = d_k
        self.device = device
        
        # 构建逆频率
        # 试着先自己写，再和这一行对照
        inv_freqs = theta ** (- torch.arange(0, d_k, 2, dtype=torch.float32) / self.d_k)
        
        # 构建频率和位置
        pos = torch.arange(0, max_seq_len, dtype=torch.float32) # (max_seq_len)
        
        # 构建sin和cos
        # 我们需要的sin和cos的形状是什么?
        # 需要能和 x 进行计算, 那么他们都是2d向量
        freqs = pos.unsqueeze(1) @ inv_freqs.unsqueeze(0)
        # 轮椅写法
        # freqs = einsum(pos, inv_freqs, 's, d -> s d')

        cos = torch.cos(freqs) # seq d/2
        sin = torch.sin(freqs) # seq d/2
        self.cos: torch.Tensor
        self.sin: torch.Tensor
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        # 分组
        x_even = x[..., 0::2] # b, s, d/2
        x_odd = x[..., 1::2]

        # 获取 sin/cos
        cos = self.cos[token_positions].to(device=x.device, dtype=x.dtype) # s, d/2
        sin = self.sin[token_positions].to(device=x.device, dtype=x.dtype)
        
        # 进行旋转
        # 补全旋转公式，并按偶/奇位置写回输出
        # 这里的idea是：把矩阵乘法转换成 element-wise
        y_even = x_even * cos - x_odd * sin
        y_odd = x_even * sin + x_odd * cos
        
        y = torch.empty_like(x)
        y[..., 0::2] = y_even
        y[..., 1::2] = y_odd
        
        return y
    
def rotate_half(x: torch.Tensor):
    d_k = x.shape[-1]
    x1 = x[..., : d_k // 2] 
    x2 = x[..., d_k // 2 : ]
    return torch.cat([-x2, x1], dim=-1)

class RoPE_Qwen(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        self.device = device
        inv_freqs = theta ** (- torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k)
        pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = pos.unsqueeze(1) @ inv_freqs.unsqueeze(0)
        cos = torch.cos(freqs)
        cos = torch.cat([cos, cos], dim=-1) # s, d_k
        sin = torch.sin(freqs)
        sin = torch.cat([sin, sin], dim=-1)
        self.cos: torch.Tensor
        self.sin: torch.Tensor
        self.register_buffer('cos', cos, persistent=False) #buffer只影响向量和模型的device保持一致
        self.register_buffer('sin', sin, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions].to(dtype=x.dtype, device=x.device)  # (S, d_k)
        sin = self.sin[token_positions].to(dtype=x.dtype, device=x.device)
        return x * cos + rotate_half(x) * sin
    
    
    
def softmax(in_features: torch.Tensor, dim: int):
    # 数值稳定版softmax
    mx = in_features.max(dim=dim, keepdim=True).values
    exp = torch.exp(in_features - mx)
    return exp / exp.sum(dim=dim, keepdim=True)
    

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None) -> torch.Tensor:
    # import pdb; pdb.set_trace()
    d_k = query.shape[-1] # b, h, s, d

    pre_sm_attn = query @ key.transpose(-1, -2) / math.sqrt(d_k)

    if mask is not None:
        pre_sm_attn = pre_sm_attn.masked_fill(mask == False, float('-inf'))
    
    attn = softmax(pre_sm_attn, dim=-1)
    
    return attn @ value


def paged_scaled_dot_product_attention(
    query: torch.Tensor,
    kv_cache: PagedLayerKVCache,
    start_pos: int,
) -> torch.Tensor:
    """Paged attention 骨架：直接读取 paged KV pool，不 materialize 连续 K/V。"""
    state = kv_cache.get_paged_state()
    batch_size, num_heads, query_len, head_dim = query.shape
    if batch_size != 1:
        raise NotImplementedError("TODO: paged attention currently only plans for batch_size=1")

    # TODO: 等级 A：按 block_table 遍历物理 block，计算每页 scores，最后合并 scores 后算输出。
    # 需要的数据：
    #   state.k_pool: [num_blocks, H, block_size, D]
    #   state.v_pool: [num_blocks, H, block_size, D]
    #   state.block_table: logical block -> physical block
    #   state.length: 当前有效 KV token 数
    #   state.block_size: 每个物理 block 容纳 token 数
    #
    # TODO: 等级 B：用 streaming softmax，不 cat K/V，也不 cat scores。
    # 核心状态：running_max / running_sum / running_out。
    #
    # TODO: causal mask 需要用逻辑位置：
    #   q_positions = start_pos + arange(query_len)
    #   key_positions = logical_block_start + arange(valid_block_len)
    _ = (state, num_heads, head_dim)
    raise NotImplementedError("TODO: implement paged attention over block_table without contiguous K/V")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, rope=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)
        self.rope = rope
        
    # 增加 start_pos 参数，用于指定开始位置，用于KV Cache
    def forward(self, x: torch.Tensor, start_pos: int = 0, kv_cache: Optional[LayerKVCache] = None) -> torch.Tensor:
        # 流程总览: x -> QKV投影 -> 分头 -> (可选RoPE) -> 因果注意力 -> 合并头 -> Wo
        # 需要注意：当使用KV Cache时，传入的x是当前token，需要补充start_pos到token_positions中
        b, s, d = x.shape
        # 1.投影
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
        
        # 2.分头行动
        # import pdb; pdb.set_trace()
        q = q.reshape(b, s, self.num_heads, d // self.num_heads).transpose(1, 2)
        k = k.reshape(b, s, self.num_heads, d // self.num_heads).transpose(1, 2)
        v = v.reshape(b, s, self.num_heads, d // self.num_heads).transpose(1, 2)

        
        # 3. rope
        # 必须先分头，再rope，因为rope是涉及维度的，不能把不同头的维度信息搞混
        if self.rope:
            token_positions = start_pos + torch.arange(0, s, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        # 加入KV Cache
        if kv_cache is not None:
            kv_cache.append(k, v)
            if getattr(kv_cache, "is_paged", False):
                out = paged_scaled_dot_product_attention(q, kv_cache, start_pos)
                return self.Wo(out.transpose(1, 2).reshape(b, s, d))
        
        
        # 4. causual attention
        # mask = torch.tril(torch.ones(s, s, dtype=torch.bool, device=x.device))
        # out = scaled_dot_product_attention(q, k, v, mask)
        # prefill，decode通用的casual mask模式
        q_positions = start_pos + torch.arange(0, s, device=x.device) # (q_len, )
        if kv_cache is not None:
            k, v = kv_cache.get()
            kv_positions = torch.arange(0, k.shape[2], device=x.device) # (key_len, )
        else:
            kv_positions = q_positions
        # none 就是插入维度，等价于squeeze(0)
        mask = (q_positions[..., None] >= kv_positions[None, ...]) # (q_len, key_len)
        
        # 5. 计算注意力
        out = scaled_dot_product_attention(q, k, v, mask)
        
        # 6. 合并头并投影
        out = self.Wo(out.transpose(1, 2).reshape(b, s, d))
        return out
        
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, theta=10000.0, max_seq_len=1024, rope=None):
        super().__init__()
        if not rope:
            rope = RoPE(theta, d_model//num_heads, max_seq_len)
        self.rope = rope
        self.attn = MultiHeadSelfAttention(d_model, num_heads, self.rope)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        
    def forward(self, x: torch.Tensor, start_pos: int = 0, kv_cache: Optional[LayerKVCache] = None) -> torch.Tensor:
        x = self.attn(self.ln1(x), start_pos=start_pos, kv_cache=kv_cache) + x
        x = self.ffn(self.ln2(x)) + x
        return x
    
class TransformerLM(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, vocab_size:int, context_length:int, num_layers:int, rope_theta:float):
        super().__init__()
        self.rope = RoPE(rope_theta, d_model // num_heads, context_length) # 每个block共用一个rope，节省计算量
        self.embd = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, theta=rope_theta, max_seq_len=context_length, rope = self.rope) for i in range(num_layers)])
        self.ln = RMSNorm(d_model=d_model)
        self.output_embd = Linear(d_model, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0, kv_cache: Optional[SimpleKVCache] = None) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        # 注意: forward输出logits，不在这里做softmax
        x = self.embd(input_ids)

        # 修改activation的dtype
        if torch.is_autocast_enabled("cuda"):
            x = x.to(torch.get_autocast_dtype("cuda"))
        
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = None
            if kv_cache is not None:
                layer_cache = kv_cache.layers[layer_idx]
            x = layer(x, start_pos=start_pos, kv_cache=layer_cache)
        logits = self.output_embd(self.ln(x))
        return logits
    
    # def generate(self, input_ids: torch.Tensor, max_new_tokens: int=256, temperature: Optional[float]=None, topp: Optional[float]=None, kv_cache: Optional[SimpleKVCache] = None):
    #     total_tokens = 0
    #     new_token = -1
    #     x = input_ids # B, S
    #     while total_tokens <= max_new_tokens and new_token != END_TOKEN:
    #         # 排序，然后找出满足p的token索引
    #         sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    #         cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    #         idx = torch.where(cumulative_probs >= topp)[0][0]
    #         probs = sorted_probs[..., :idx+1]
    #         new_token = torch.multinomial(probs, num_samples=1).to(x.device)
    #         x = torch.cat([x, new_token], dim=-1)
    #         total_tokens += 1
    #     output_ids = x
    #     return output_ids
    
    def load_checkpoint(self, src):
        checkpoint = torch.load(src)
        self.load_state_dict(checkpoint['model_state'])

def build_model(model_config: ModelConfig) -> TransformerLM:
    model = TransformerLM(
        model_config.d_model,
        model_config.num_heads,
        model_config.d_ff,
        model_config.vocab_size,
        model_config.context_length,
        model_config.num_layers,
        model_config.rope_theta,
    )
    return model
