import torch


def alibi_slopes(num_heads, device="cpu"):
    return 2 ** (-8 * torch.arange(1, num_heads + 1, device=device, dtype=torch.float32) / num_heads)


def alibi_bias(num_heads, seq_len, device):
    slopes = alibi_slopes(num_heads, device)
    pos = torch.arange(seq_len, device=device)
    dist = pos[None, :] - pos[:, None]
    bias = slopes[:, None, None] * dist
    return bias.masked_fill(dist > 0, float("-inf"))
