import torch
from torch import nn


class ExampleLoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        B = labels.shape[0]
        Loss = 0
        for i in range(10):
            msk = (labels == i)
            cnt = msk.sum()
            if (cnt > 0):
                Loss -= (logits[msk, i]).sum()
        Loss += torch.logsumexp(logits, dim=1).sum()
        return {"loss" : Loss / B}
