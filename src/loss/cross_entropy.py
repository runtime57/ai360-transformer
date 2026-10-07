import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        loss = self.loss(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        return {"loss": loss}
