import torch
import torch.nn.functional as F
import torch.nn as nn


class TelescopicLoss(nn.Module):
    def __init__(
        self,
        telescopic_lambda=1.0,
        telescopic_gamma=1.0,
        temperature=1.0,
        label_smoothing=0.0,
        ignore_class_id=0
    ):
        super().__init__()
        self.telescopic_lambda = telescopic_lambda
        self.telescopic_gamma = telescopic_gamma
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.ignore_class_id = ignore_class_id

    def _set_context(self, ignore_class_id, **context):
        self.ignore_class_id = ignore_class_id

    def _calculate(self, logits, target):
        # logits: [B, L, N], target: [B, L]
        B, L, _ = logits.shape

        logits = logits / self.temperature
        logits_flat = logits.view(B * L, -1)
        target_flat = target.view(-1)

        return F.cross_entropy(
            logits_flat,
            target_flat,
            ignore_index=self.ignore_class_id,
            label_smoothing=self.label_smoothing
        )

    def forward(self, logits, target, prefix_logits=None, **batch):
        anchor_loss = self._calculate(logits, target)

        if prefix_logits is None:
            # evaluation pass: anchor only
            return {
                'loss': anchor_loss,
                'anchor_loss': anchor_loss.detach(),
                'prefix_loss': torch.zeros_like(anchor_loss)
            }

        prefix_loss = self._calculate(prefix_logits, target)
        loss = self.telescopic_gamma * anchor_loss + self.telescopic_lambda * prefix_loss

        return {
            'loss': loss,
            'anchor_loss': anchor_loss.detach(),
            'prefix_loss': prefix_loss.detach()
        }
