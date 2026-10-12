import torch
import torch.nn.functional as F

from src.metrics.base_metric import BaseMetric


class telescopic_metric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, all_logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        num_prefixes = all_logits.shape[0]
        losses = []
        
        for i in range(num_prefixes):
            logits_i = all_logits[i]
            
            if logits_i.dim() > 2:
                # Например, если [B, L, vocab_size] -> [B * L, vocab_size]
                B, L, V = logits_i.shape
                logits_flat = logits_i.reshape(-1, V)
                labels_flat = labels.reshape(-1)
            else:
                logits_flat = logits_i
                labels_flat = labels
                
            loss = F.cross_entropy(logits_flat, labels_flat, reduction='mean')
            losses.append(loss.item())
            
        return torch.tensor(losses, device=all_logits.device)
