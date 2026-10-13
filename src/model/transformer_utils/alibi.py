import torch


def alibi_bias(num_heads, seq_len, device):
    slopes = 2 ** (-8 * torch.arange(1, num_heads + 1, device=device) / num_heads)
    pos = torch.arange(seq_len, device=device)
    dist = pos[None, :] - pos[:, None]
    bias = slopes[:, None, None] * dist
    return bias.masked_fill(dist > 0, float("-inf"))
