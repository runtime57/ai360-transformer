import torch.nn.functional as F
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class MtpCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=1.0, label_smoothing=0.0, ignore_class_id=0, head_weights=None):
        super().__init__()
        self.ignore_class_id = ignore_class_id
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.head_weights = head_weights

    def _set_context(self, ignore_class_id, **context):
        self.ignore_class_id = ignore_class_id

    def one_head(self, logits, target, i):
        # slicing happens inside the checkpoint: only the (already existing) full logits tensor
        # is kept for backward, not a per-head copy or the fp32 log-softmax
        V = logits.shape[-1]
        logits_i = logits[:, i].reshape(-1, V) # [B*L, V]
        targets_i = target[:, i].reshape(-1) # [B*L]
        return F.cross_entropy(logits_i, targets_i, ignore_index=self.ignore_class_id, label_smoothing=self.label_smoothing)

    def forward(self, mtp_logits, target, **batch):
        # logits: [B, N, L, V]
        # target: [B, N, L]

        B, n, L, V = mtp_logits.shape
        out, tot = {}, 0.0
        head_weights = self.head_weights if self.head_weights is not None else [1.0] * n

        logits = mtp_logits / self.temperature if self.temperature != 1.0 else mtp_logits
        for i in range(n):
            out[f'loss_{i}'] = checkpoint(self.one_head, logits, target, i, use_reentrant=False)
            tot += head_weights[i] * out[f'loss_{i}']
        out['loss'] = tot / sum(head_weights)
        return out
