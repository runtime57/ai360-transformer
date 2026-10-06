import torch


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

    tokens = torch.stack(
        [torch.as_tensor(elem["token"], dtype=torch.long) for elem in dataset_items]
    )

    return {
        "token": tokens[:, :-1],
        "labels": tokens[:, 1:],
    }
