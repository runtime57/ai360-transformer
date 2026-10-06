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

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        return accuracy.item()
