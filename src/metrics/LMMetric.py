import torch

from src.metrics.base_metric import BaseMetric


class LMMetric(BaseMetric):
    def __init__(self, pad_id=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_id = pad_id

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        predictions = logits.argmax(dim=-1)
        mask = labels != self.pad_id
        correct = (predictions == labels) & mask
        return correct.sum() / mask.sum()
