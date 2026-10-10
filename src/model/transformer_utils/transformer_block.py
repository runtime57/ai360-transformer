import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, use_rope: bool = False, rope_base: float = 10000.0, max_seq_len=None):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.use_rope = use_rope
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len
        assert not use_rope or self.head_dim % 2 == 0

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout

        cos, sin = self.precalc_rope()
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def precalc_rope(self):
        if not self.use_rope:
            return None, None

        positions = torch.arange(self.max_seq_len, dtype=torch.float32)
        dim_indices = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        inv_freq = self.rope_base ** (-dim_indices / self.head_dim)

        angles = positions[:, None] * inv_freq[None, :]

        return angles.cos(), angles.sin()

    @staticmethod
    def apply_rope(x: Tensor, cos: Tensor, sin: Tensor):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        return torch.stack(
            (
                x_even * cos - x_odd * sin,
                x_even * sin + x_odd * cos,
            ),
            dim=-1,
        ).flatten(-2)


    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos = self.rope_cos[:T].to(dtype=q.dtype)[None, None, :, :]
            sin = self.rope_sin[:T].to(dtype=q.dtype)[None, None, :, :]

            q = self.apply_rope(q, cos, sin)
            k = self.apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_hidden_dim: int,
        num_heads: int,
        ffn_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        max_seq_len = None
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads, attn_dropout, use_rope, rope_base, max_seq_len)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = MLP(d_model, mlp_hidden_dim, ffn_dropout)
        self.ffn_dropout = nn.Dropout(ffn_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn_dropout(self.attention(self.norm1(x)))
        x = x + self.ffn_dropout(self.feed_forward(self.norm2(x)))
        return x
