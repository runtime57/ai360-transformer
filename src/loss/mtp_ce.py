import torch.nn.functional as F
import torch.nn as nn


class MtpCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=1.0, label_smoothing=0.0, ignore_class_id=0, head_weights=None):
        super().__init__()
        self.ignore_class_id = ignore_class_id
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.head_weights = head_weights

    def _set_context(self, ignore_class_id, **context):
        self.ignore_class_id = ignore_class_id

    def forward(self, mtp_logits, target, **batch):
        # logits: [B, N, L, V]
        # target: [B, N, L]

        B, n, L, V = mtp_logits.shape
        out, tot = {}, 0.0
        head_weights = self.head_weights if self.head_weights is not None else [1.0] * n

        logits = mtp_logits / self.temperature
        for i in range(n):
            logits_i = logits[:, i] # [B, L, V]
            logits_i = logits_i.reshape(-1, V) # [B*L, V]
            targets_i = target[:, i] # [B, L]
            targets_i = targets_i.reshape(-1) # [B*L]
            out[f'loss_{i}'] = F.cross_entropy(logits_i, targets_i, ignore_index=self.ignore_class_id, label_smoothing=self.label_smoothing)
            tot += head_weights[i] * out[f'loss_{i}']
        out['loss'] = tot / sum(head_weights)
        return out
