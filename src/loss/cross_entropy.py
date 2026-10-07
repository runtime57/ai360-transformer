import torch
from torch import nn


class SequenceCrossEntropyLoss(nn.Module):
    """
    Cross-entropy over every position of a sequence (next-token prediction).
    """

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Args:
            logits (Tensor): (B, T, vocab_size) model output.
            labels (Tensor): (B, T) next-token ids.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        # nn.CrossEntropyLoss wants classes in dim 1, so flatten B and T together
        return {"loss": self.loss(logits.flatten(0, 1), labels.flatten())}
