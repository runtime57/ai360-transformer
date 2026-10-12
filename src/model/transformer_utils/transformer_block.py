import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        rope_base: float = 10000.0,
        max_seq_len=None,
        attention_type: str = "softmax",
        linear_chunk_size: int = 64,
        linear_eps: float = 1e-6,
    ):
        super().__init__()
        if num_heads <= 0 or d_model <= 0 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if attention_type not in {"softmax", "linear"}:
            raise ValueError("attention_type must be 'softmax' or 'linear'")
        if linear_chunk_size <= 0 or linear_eps <= 0:
            raise ValueError("linear_chunk_size and linear_eps must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

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

    def _rope_factors(self, length: int, x: Tensor):
        if self.rope_cos.shape[0] < length:
            with torch.inference_mode(False):
                positions = torch.arange(length, device=x.device, dtype=torch.float32)
                dims = torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32)
                angles = positions[:, None] * self.rope_base ** (-dims / self.head_dim)
                self.rope_cos, self.rope_sin = angles.cos(), angles.sin()
        cos = self.rope_cos[:length].to(dtype=x.dtype)[None, None]
        sin = self.rope_sin[:length].to(dtype=x.dtype)[None, None]
        return cos, sin

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

    def _linear_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        output_dtype = v.dtype
        with torch.autocast(device_type=q.device.type, enabled=False):
            if q.dtype in {torch.float16, torch.bfloat16}:
                q, k, v = q.float(), k.float(), v.float()
            q_features, k_features = F.elu(q) + 1, F.elu(k) + 1
            q_rot, k_rot = q_features, k_features
            if self.use_rope:
                cos, sin = self._rope_factors(q.shape[-2], q_features)
                q_rot = self.apply_rope(q_features, cos, sin)
                k_rot = self.apply_rope(k_features, cos, sin)

            v = F.dropout(v, p=self.dropout, training=self.training)
            batch, heads, length, dim = q.shape
            chunk_size = min(self.linear_chunk_size, length)
            padding = (-length) % chunk_size
            num_chunks = (length + padding) // chunk_size

            def chunk(x):
                return F.pad(x, (0, 0, 0, padding)).reshape(batch, heads, num_chunks, chunk_size, dim)

            qc, kc, vc = chunk(q_rot), chunk(k_rot), chunk(v)
            chunk_kv = kc.transpose(-1, -2) @ vc
            kv_prefix = F.pad(chunk_kv[:, :, :-1].cumsum(dim=2), (0, 0, 0, 0, 1, 0))
            causal_mask = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).triu(1)
            local_weights = (qc @ kc.transpose(-1, -2)).masked_fill(causal_mask, 0)
            numerator = (qc @ kv_prefix + local_weights @ vc).reshape(batch, heads, length + padding, dim)[:, :, :length]
            denominator = (q_features * k_features.cumsum(dim=-2)).sum(-1, keepdim=True)
            return (numerator / denominator.clamp_min(self.linear_eps)).to(output_dtype)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.attention_type == "linear":
            y = self._linear_attention(q, k, v)
        else:
            if self.use_rope:
                cos, sin = self._rope_factors(T, q)
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
        max_seq_len=None,
        attention_type: str = "softmax",
        linear_chunk_size: int = 64,
        linear_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(
            d_model,
            num_heads,
            attn_dropout,
            use_rope,
            rope_base,
            max_seq_len,
            attention_type,
            linear_chunk_size,
            linear_eps,
        )
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = MLP(d_model, mlp_hidden_dim, ffn_dropout)
        self.ffn_dropout = nn.Dropout(ffn_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn_dropout(self.attention(self.norm1(x)))
        x = x + self.ffn_dropout(self.feed_forward(self.norm2(x)))
        return x
