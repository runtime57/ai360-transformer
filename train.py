import logging
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

from src.datasets.data_utils import get_dataloaders
from src.logger import WandBWriter
from src.trainer import Trainer
from src.utils.dist_utils import barrier, cleanup_distributed, init_distributed
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)

def set_context(object, context):
    if context is not None and hasattr(object, "_set_context"):
        object._set_context(**context)


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    # multi-GPU: launch via `torchrun --nproc_per_node=N train.py ...`
    rank, local_rank, world_size = init_distributed()
    is_main = rank == 0

    set_random_seed(config.trainer.seed + rank)

    if is_main:
        project_config = OmegaConf.to_container(config)
        logger = setup_saving_and_logging(config)
        writer = instantiate(config.writer, logger, project_config)
    else:
        # only the main process writes logs, checkpoints and to the tracker
        logger = logging.getLogger(f"train.rank{rank}")
        logger.setLevel(logging.WARNING)
        writer = WandBWriter(logger, None, project_name=None, disable=True)

    if world_size > 1:
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        logger.info(f"Distributed training on {world_size} GPUs")
    elif config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # the main process prepares data files and the tokenizer first,
    # the rest reuse them afterwards (avoids concurrent writes)
    tokenizer_config = config.get("tokenizer")
    if is_main:
        dataloaders, batch_transforms, context, tokenizer = get_dataloaders(config, device, tokenizer_config, logger)
    barrier()
    if not is_main:
        dataloaders, batch_transforms, context, tokenizer = get_dataloaders(config, device, tokenizer_config, logger)
    barrier()
    train_context = context.get("train")

    model = instantiate(config.model).to(device)
    set_context(model, train_context)

    if tokenizer is not None and hasattr(model, "_set_tokenizer"):
        model._set_tokenizer(tokenizer)

    logger.info(model)

    if world_size > 1:
        device_ids = [local_rank] if torch.cuda.is_available() else None
        model = DistributedDataParallel(model, device_ids=device_ids)

    loss = instantiate(config.loss).to(device)
    set_context(loss, train_context)

    metrics = instantiate(config.metrics)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)

    epoch_len = config.trainer.get("epoch_len")
    real_epoch_len = epoch_len if epoch_len is not None else len(dataloaders['train'])
    total_steps = config.trainer.epochs * real_epoch_len

    if config.get("lr_scheduler") is not None:
        config.lr_scheduler['steps'] = total_steps
        lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    else:
        lr_scheduler = None

    trainer = Trainer(
        model=model,
        criterion=loss,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        # with DDP a skipped batch on one GPU desyncs the gradient all-reduce
        skip_oom=config.trainer.get("skip_oom", True) and world_size == 1,
    )

    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
