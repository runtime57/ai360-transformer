import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from .linear_attention import causal_linear_attention


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, use_rope: bool = False, rope_base: float = 10000.0, max_seq_len: int | None = None, use_headwise_gate: bool = False, use_elementwise_gate: bool = False, attention_type: str = "softmax", linear_chunk_size: int = 64, linear_eps: float = 1e-6):
        super().__init__()
        if num_heads <= 0 or d_model <= 0 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if attention_type not in {"softmax", "linear"}:
            raise ValueError("attention_type must be 'softmax' or 'linear'")
        if linear_chunk_size <= 0 or not linear_eps > 0:
            raise ValueError("linear_chunk_size and linear_eps must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        assert not (use_headwise_gate and use_elementwise_gate)

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.use_rope = use_rope
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len
        self.attention_type = attention_type
        self.linear_chunk_size = linear_chunk_size
        self.linear_eps = linear_eps
        if use_rope and (self.head_dim % 2 or rope_base <= 0):
            raise ValueError("RoPE requires an even head_dim and positive rope_base")

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout

        self.gate = None
        if use_headwise_gate:
            self.gate = nn.Linear(d_model, num_heads, bias=False)
        elif use_elementwise_gate:
            self.gate = nn.Linear(d_model, d_model, bias=False)

        cos, sin = self.precalc_rope()
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def precalc_rope(self):
        if not self.use_rope:
            return None, None

        positions = torch.arange(self.max_seq_len or 0, dtype=torch.float32)
        dim_indices = torch.arange(0, self.head_dim, 2, dtype=torch.float32)
        inv_freq = self.rope_base ** (-dim_indices / self.head_dim)

        angles = positions[:, None] * inv_freq[None, :]

        return angles.cos(), angles.sin()

    def _rotate_queries_and_keys(self, q, k):
        length = q.shape[-2]
        if self.rope_cos.shape[0] < length:
            # A cache grown during generation must also work in later training.
            with torch.inference_mode(False):
                positions = torch.arange(length, device=q.device, dtype=torch.float32)
                dims = torch.arange(
                    0, self.head_dim, 2, device=q.device, dtype=torch.float32
                )
                angles = positions[:, None] * self.rope_base ** (-dims / self.head_dim)
                self.rope_cos, self.rope_sin = angles.cos(), angles.sin()
        cos = self.rope_cos[:length].to(dtype=q.dtype)[None, None]
        sin = self.rope_sin[:length].to(dtype=q.dtype)[None, None]
        return self.apply_rope(q, cos, sin), self.apply_rope(k, cos, sin)

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

        gate = None
        if self.gate is not None:
            gate = torch.sigmoid(self.gate(x))

        if self.attention_type == "linear":
            y = causal_linear_attention(
                q, k, v,
                chunk_size=self.linear_chunk_size,
                eps=self.linear_eps,
                dropout_p=self.dropout,
                training=self.training,
                rotary=self._rotate_queries_and_keys if self.use_rope else None,
            )
        else:
            if self.use_rope:
                q, k = self._rotate_queries_and_keys(q, k)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )

        if self.gate is not None:
            y = y * gate.view(B, T, self.num_heads, -1).transpose(1, 2)

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
        max_seq_len: int | None = None,
        use_headwise_gate: bool = False,
        use_elementwise_gate: bool = False,
        attention_type: str = "softmax",
        linear_chunk_size: int = 64,
        linear_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, num_heads, attn_dropout, use_rope, rope_base, max_seq_len, use_headwise_gate, use_elementwise_gate, attention_type=attention_type, linear_chunk_size=linear_chunk_size, linear_eps=linear_eps)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = MLP(d_model, mlp_hidden_dim, ffn_dropout)
        self.ffn_dropout = nn.Dropout(ffn_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn_dropout(self.attention(self.norm1(x)))
        x = x + self.ffn_dropout(self.feed_forward(self.norm2(x)))
        return x
