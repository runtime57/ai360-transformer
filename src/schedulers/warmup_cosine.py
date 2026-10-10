from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, min_lr_ratio, min_end_lr_ratio, warmup_ratio, steps, last_epoch=-1):
        self.warmup_steps = max(int(warmup_ratio * steps), 1)
        self.cosine_steps = steps - self.warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.min_end_lr_ratio = min_end_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_steps:
            warmup_ratio = self.last_epoch / self.warmup_steps
            scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * warmup_ratio
        else:
            ratio = (self.last_epoch - self.warmup_steps) / self.cosine_steps
            scale = self.min_end_lr_ratio + (1 - self.min_end_lr_ratio) * (1 + np.cos(ratio * np.pi)) / 2

        return [base_lr * scale for base_lr in self.base_lrs]
