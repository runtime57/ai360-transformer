import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        return {
            "loss": self.loss(logits.flatten(0, 1), labels.flatten())
        }  # ну так чисто повыпендриваться
