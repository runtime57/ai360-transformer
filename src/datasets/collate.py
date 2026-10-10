import torch
from torch.nn.utils.rnn import pad_sequence

MAX_LEN = 512


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    seqs = [elem["data_object"][: MAX_LEN + 1] for elem in dataset_items]

    return {
        "token": pad_sequence([s[:-1] for s in seqs], batch_first=True),
        "labels": pad_sequence([s[1:] for s in seqs], batch_first=True),
    }