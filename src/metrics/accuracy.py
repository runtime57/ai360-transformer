import torch

from src.metrics.base_metric import BaseMetric


class Accuracy(BaseMetric):
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        predictions = logits.argmax(dim=1)
        return (predictions == labels).float().mean().item()
