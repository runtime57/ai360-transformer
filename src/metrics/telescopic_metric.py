import torch
import torch.nn.functional as F

from src.metrics.base_metric import BaseMetric

import torch
import torch.nn.functional as F

from src.metrics.base_metric import BaseMetric


class telescopic_metric(BaseMetric):
    def __init__(
        self,
        metric=None,
        device="auto",
        ignore_index=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.metric = (
            metric.to(device)
            if metric is not None and hasattr(metric, "to")
            else metric
        )
        self.ignore_index = ignore_index

    def __call__(
        self,
        all_logits: torch.Tensor,
        target: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        labels = target if target is not None else labels

        if labels is None:
            raise ValueError("Expected 'target' or 'labels'")

        # all_logits: [N, B, L, V]
        num_prefixes = all_logits.shape[0]
        losses = []

        loss_kwargs = {"reduction": "mean"}
        if self.ignore_index is not None:
            loss_kwargs["ignore_index"] = self.ignore_index

        for i in range(num_prefixes):
            logits = all_logits[i]
            vocab_size = logits.shape[-1]

            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
                **loss_kwargs,
            )
            losses.append(loss.detach())

        return torch.stack(losses)
