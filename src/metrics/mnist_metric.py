import torch

from src.metrics.base_metric import BaseMetric


class MnistMetric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        """
        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        classes = logits.argmax(dim=-1)
        return torch.nn.functional.mse_loss(classes, labels)
