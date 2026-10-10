import torch
import torch.nn as nn
from torch import Tensor

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x_float = x.float()
        rms = torch.sqrt((x_float * x_float).mean(dim=-1, keepdim=True) + self.eps)
        return ((x_float / rms) * self.weight).to(dtype)