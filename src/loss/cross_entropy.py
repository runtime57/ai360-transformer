import torch.nn.functional as F
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, temperature=1.0, label_smoothing=0.0, ignore_class_id=0):
        super().__init__()
        self.ignore_class_id = ignore_class_id
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def _set_context(self, ignore_class_id, **context):
        self.ignore_class_id = ignore_class_id


    def forward(self, logits, target, **batch):
        # logits: [B, L, N]
        # target: [B, L]

        B, L, _ = logits.shape

        logits = logits / self.temperature
        logits_flat = logits.view(B * L, -1)
        target_flat = target.view(-1)

        loss = F.cross_entropy(logits_flat, target_flat, ignore_index=self.ignore_class_id, label_smoothing=self.label_smoothing)

        return {'loss': loss}
