from itertools import repeat

from hydra.utils import instantiate

from src.datasets.collate import collate_fn
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def _set_tokenizer(dataset, tokenizer):
    """Attach one tokenizer to a dataset and any nested datasets."""
    if tokenizer is None:
        return

    if hasattr(dataset, "datasets"):
        for nested_dataset in dataset.datasets:
            _set_tokenizer(nested_dataset, tokenizer)

    if hasattr(dataset, "_set_tokenizer"):
        dataset._set_tokenizer(tokenizer)
    else:
        dataset.tokenizer = tokenizer


def get_dataloaders(config, device, tokenizer_config=None, logger=None):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
        tokenizer (BaseTokenizer | None): initialized tokenizer shared by
            all dataset partitions.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """

    batch_transforms = instantiate(config.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    datasets = instantiate(config.datasets)

    tokenizer = None
    if tokenizer_config is not None:
        from src.tokenizers import init_tokenizer
        train_dataset = datasets.get("train")
        texts = None

        if tokenizer_config.get("use_dataset_get_train_text_func", True) and hasattr(train_dataset, "_get_train_text"):
            texts = train_dataset._get_train_text()

        tokenizer = init_tokenizer(tokenizer_config, log=logger, texts=texts)

    for dataset in datasets.values():
        _set_tokenizer(dataset, tokenizer)

    context = {}
    for dataset in datasets.values():
        if not hasattr(dataset, "_get_context"):
            continue
        dataset_context = dataset._get_context()
        context[dataset_context.pop("part")] = dataset_context

    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        assert config.dataloader.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms, context, tokenizer
