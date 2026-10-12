import math
import torch
import torch.nn.functional as F

from src.metrics.base_metric import BaseMetric


class TokenNLL(BaseMetric):
    """Unsmoothed next-token NLL, weighted by non-padding target tokens."""

    def __init__(self, ignore_class_id=0, name="nll"):
        super().__init__(name=name)
        self.ignore_class_id = ignore_class_id

    def _set_context(self, ignore_class_id, **context):
        self.ignore_class_id = ignore_class_id

    def weight(self, target, **batch):
        return (target != self.ignore_class_id).sum().item()

    @torch.no_grad()
    def __call__(self, logits, target, **batch):
        count = self.weight(target)
        if count == 0:
            return 0.0
        total = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=self.ignore_class_id,
            reduction="sum",
        )
        return total.item() / count


class Perplexity(TokenNLL):
    """Exponentiate the token-weighted corpus NLL, not each batch's NLL."""

    def __init__(self, ignore_class_id=0, name="perplexity"):
        super().__init__(ignore_class_id=ignore_class_id, name=name)

    @staticmethod
    def aggregate(mean_nll):
        try:
            return math.exp(mean_nll)
        except OverflowError:
            return math.inf


class TokenAccuracy(TokenNLL):
    def __init__(self, ignore_class_id=0, name="token_accuracy"):
        super().__init__(ignore_class_id=ignore_class_id, name=name)

    @torch.no_grad()
    def __call__(self, logits, target, **batch):
        valid = target != self.ignore_class_id
        count = valid.sum().item()
        if count == 0:
            return 0.0
        correct = ((logits.argmax(-1) == target) & valid).sum().item()
        return correct / count
