import torch
import torch.nn as nn
from typing import Optional
import math

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
        bias: bool = False, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
    ):
        kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **kwargs)  # 出于内存按行存储的考虑
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features), **kwargs)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias:
            return x @ self.weight.T + self.bias
        else:
            return x @ self.weight.T
        

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
        device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
    ):
        kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **kwargs)
        )
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,
        device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,             
    ):
        kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(d_model, **kwargs)
        )
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim = -1, keepdim = True) + self.eps
        x_output = x * var.rsqrt() * self.weight
        return x_output.to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwishGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int,
        device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
    ):
        kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff    # roughly d_ff = 8 / 3 * d_model
        self.gate = Linear(in_features=d_model, out_features=d_ff, **kwargs)
        self.up = Linear(in_features=d_model, out_features=d_ff, **kwargs)
        self.down = Linear(in_features=d_ff, out_features=d_model, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down((silu(self.gate(x)) * self.up(x)))


class RoPE(nn.Module):
    ''' Rotary Positional Embedding '''
    '''
        RoPE通过绝对位置编码的方式实现相对位置编码, 综合了绝对位置编码和相对位置编码的优点。
        主要就是对attention中的q, k向量注入了绝对位置信息, 然后用更新的q,k向量做attention中的内积就会引入相对位置信息了。
    '''
    def __init__(self, theta: float, d_qk: int, max_seq_len: int,
        device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,
    ):
        self.kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.theta = theta
        self.d_qk = d_qk
        self.max_seq_len = max_seq_len
        self.register_buffer(
            'cos_sin',
            self._compute_freqs_cis(d_qk, max_seq_len, theta),
            persistent = False                     
        )
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_sin = self.cos_sin[:x.size(-2), :] if token_positions is None else self.cos_sin[token_positions]
        return self._apply_rope(x, cos_sin)

    def _compute_freqs_cis(self, head_dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, **self.kwargs) / head_dim))  # shape (head_dim/2, )
        pos = torch.arange(max_len, **self.kwargs).float()  # shape (max_len, )

        freqs = torch.outer(pos, freqs)     # shape (max_len, head_dim/2)
        freqs_cis = torch.polar(torch.ones_like(freqs, **self.kwargs), freqs)

        cos_sin = torch.cat([freqs_cis.real, freqs_cis.imag], dim = -1) # shape (max_len, head_dim)
        return cos_sin

    def _apply_rope(self, x: torch.Tensor, cos_sin: torch.Tensor) -> torch.Tensor:
        # 这里建议用个测试看看, 关于reshape, chunk, stack的使用
        x1, x2 = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)     # x1, x2 shape (seq_len, head_dim/2)
        cos, sin = torch.chunk(cos_sin, 2, dim = -1)            # cos, sin shape (seq_len, head_dim/2)
        xout = torch.stack([x1 * cos - x2 * sin, x1 *sin + x2 * cos], dim = -1)     # shape (seq_len, head_dim/2, 2)
        return xout.reshape(*x.shape)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_exp = torch.exp(x - x.max(dim = dim, keepdim = True).values)
    return x_exp / x_exp.sum(dim = dim, keepdim = True)


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    dim = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(dim)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    scores_weight = softmax(scores, dim = -1)
    output = scores_weight @ V
    return output

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
        device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None             
    ):
        self.kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.fc_qkv = Linear(d_model, 3 * d_model, **self.kwargs)
        self.fc_out = Linear(d_model, d_model, **self.kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.fc_qkv(x)                                                # (batch_size, seq_len, 3*d_model)
        seq_len = x.size(1)
        xq, xk, xv = torch.chunk(qkv, 3, dim = -1)                           # (batch_size, seq_len, d_model)
        xq = xq.reshape(*xq.shape[:-1], self.num_heads, -1).transpose(1, 2)       # 注意此时不连续了 (batch_size, self.num_heads, seq_len, self.head_dim)
        xk = xk.reshape(*xk.shape[:-1], self.num_heads, -1).transpose(1, 2)       # 注意此时不连续了 (batch_size, self.num_heads, seq_len, self.head_dim)
        xv = xv.reshape(*xv.shape[:-1], self.num_heads, -1).transpose(1, 2)       # 注意此时不连续了 (batch_size, self.num_heads, seq_len, self.head_dim)
        mask = torch.ones((seq_len, seq_len), device = self.kwargs['device']).tril()        # 下三角
        # mask = mask.reshape(1, 1, *mask.shape)
        xout = scaled_dot_product_attention(xq, xk, xv, mask)  # (batch_size, self.num_heads, seq_len, self.head_dim)
        xout = xout.transpose(1,2).reshape(*x.shape[:-1], -1)  # (batch_size, seq_len, d_model)
        return self.fc_out(xout)

class MultiheadRoPESelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float,
        device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,        
    ):
        super().__init__()
        self.kwargs = {'device': device, 'dtype': dtype}
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.head_dim = d_model // num_heads
        self.fc_qkv = Linear(d_model, 3 * d_model, **self.kwargs)
        self.fc_out = Linear(d_model, d_model, **self.kwargs)
        self.rope = RoPE(theta, self.head_dim, max_seq_len, **self.kwargs)
    
    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        qkv = self.fc_qkv(x)        # (batch_size, seq_len, 3*d_model)
        xq, xk, xv = torch.chunk(qkv, 3, dim=-1)
        xq = self.rope(xq.reshape(*xq.shape[:-1], self.num_heads, self.head_dim).transpose(1,2), token_positions)    # (batch_size, self.num_heads, seq_len, self.head_dim)
        xk = self.rope(xk.reshape(*xk.shape[:-1], self.num_heads, self.head_dim).transpose(1,2), token_positions)    # (batch_size, self.num_heads, seq_len, self.head_dim)
        xv = xv.reshape(*xv.shape[:-1], self.num_heads, self.head_dim).transpose(1,2)               # (batch_size, self.num_heads, seq_len, self.head_dim)
        mask = torch.ones((seq_len, seq_len), device = self.kwargs["device"]).tril()
        # mask = mask.reshape(1, 1, *mask.shape)
        xout = scaled_dot_product_attention(xq, xk, xv, mask)
        xout = xout.transpose(1, 2).reshape(*x.shape[:-1], self.d_model)
        return self.fc_out(xout)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,
        ffn_type: str = "SwishGLU", use_rope: bool = True,            
        device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,  
    ):
        super().__init__()
        self.kwargs = {'device': device, 'dtype': dtype}
        if use_rope:
            self.attn = MultiheadRoPESelfAttention(d_model, num_heads, max_seq_len, theta, **self.kwargs)
        else:
            self.attn = MultiheadSelfAttention(d_model, num_heads, **self.kwargs)

        if ffn_type == "SwishGLU":
            self.ffn = SwishGLU(d_model, d_ff, **self.kwargs)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}\n")
        self.ln1 = RMSNorm(d_model, **self.kwargs)
        self.ln2 = RMSNorm(d_model, **self.kwargs)

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

def init_weights(block: nn.Module):
    if isinstance(block, Linear):
        std = math.sqrt(2.0 / (block.in_features + block.out_features))
        nn.init.trunc_normal_(block.weight, mean = 0.0, std = std, a = -3*std, b = 3*std)
        if block.bias is not None:
            nn.init.zeros_(block.bias)
    elif isinstance(block, Embedding):
        nn.init.trunc_normal_(block.weight, mean = 0.0, std = 1.0, a = -3.0, b = 3.0)
    elif isinstance(block, RMSNorm):
        nn.init.ones_(block.weight)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int,
        d_ff: int, rope_theta: float, ffn_type: str = "SwishGLU", use_rope: bool = True,
        device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None,         
    ):
        super().__init__()
        self.kwargs = {'device': device, 'dtype': dtype}
        self.token_embeddings = Embedding(vocab_size, d_model, **self.kwargs)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, ffn_type, use_rope, **self.kwargs)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model, **self.kwargs)
        self.lm_head = Linear(d_model, vocab_size)
        self.apply(init_weights)
        self.max_seq_len = context_length
    
    def forward(self, token_ids: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = token_ids.size(1)
        assert seq_len <= self.max_seq_len, "Sequence length exceeds model capacity"
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        #return softmax(logits, dim=-1)     # 验证没有使用softmax
        return logits
