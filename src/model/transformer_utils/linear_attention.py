"""Chunked causal ELU+1 attention without a full sequence attention matrix."""

import torch
import torch.nn.functional as F


def causal_linear_attention(
    q,
    k,
    v,
    *,
    chunk_size=64,
    eps=1e-6,
    dropout_p=0.0,
    training=False,
    rotary=None,
):
    """Return [batch, heads, time, head_dim], including the current token.

    ``rotary`` optionally rotates the mapped Q/K used in the numerator.
    Normalization always uses the unrotated, positive ELU+1 features, matching
    the original linear+RoPE experiment. Dropout acts on V, rather than on
    softmax probabilities. Half-precision inputs accumulate in float32.

    The implementation stores a small causal matrix per chunk and prefix
    sums of K^T V, rather than a time-by-time matrix for the whole sequence.
    It has no parameters or persistent state, so checkpoint keys stay intact.
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            "Q, K and V must have matching [batch, heads, time, dim] shapes"
        )
    if q.shape[-2] == 0:
        raise ValueError("Linear attention requires a nonempty sequence")
    if chunk_size <= 0 or not eps > 0:
        raise ValueError("chunk_size and eps must be positive")

    output_dtype = v.dtype
    with torch.autocast(device_type=q.device.type, enabled=False):
        if q.dtype in {torch.float16, torch.bfloat16}:
            q, k, v = q.float(), k.float(), v.float()
        q_features, k_features = F.elu(q) + 1, F.elu(k) + 1
        q_numerator, k_numerator = q_features, k_features
        if rotary is not None:
            q_numerator, k_numerator = rotary(q_features, k_features)

        v = F.dropout(v, p=dropout_p, training=training)
        batch, heads, length, dim = q.shape
        chunk_size = min(chunk_size, length)
        padding = (-length) % chunk_size
        num_chunks = (length + padding) // chunk_size

        def chunks(x):
            return F.pad(x, (0, 0, 0, padding)).reshape(
                batch, heads, num_chunks, chunk_size, dim
            )

        qc, kc, vc = chunks(q_numerator), chunks(k_numerator), chunks(v)
        chunk_kv = kc.transpose(-1, -2) @ vc
        kv_prefix = F.pad(chunk_kv[:, :, :-1].cumsum(dim=2), (0, 0, 0, 0, 1, 0))
        causal_mask = torch.ones(
            chunk_size, chunk_size, dtype=torch.bool, device=q.device
        ).triu(1)
        local_weights = (qc @ kc.transpose(-1, -2)).masked_fill(causal_mask, 0)
        numerator = (qc @ kv_prefix + local_weights @ vc).reshape(
            batch, heads, length + padding, dim
        )[:, :, :length]
        denominator = (q_features * k_features.cumsum(dim=-2)).sum(-1, keepdim=True)
        return (numerator / denominator.clamp_min(eps)).to(output_dtype)
