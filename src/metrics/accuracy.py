import torch

from src.metrics.base_metric import BaseMetric

class AccuracyMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        self.super().__init__(*args, **kwargs)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        classes = logits.argmax(dim=-1)
        return (classes == labels).mean(dtype=torch.float32)