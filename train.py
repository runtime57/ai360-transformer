import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
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
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = config.trainer.device

    tokenizer_config = config.get("tokenizer")
    dataloaders, batch_transforms, context, tokenizer = get_dataloaders(config, device, tokenizer_config, logger)
    train_context = context.get("train")

    model = instantiate(config.model).to(device)
    set_context(model, train_context)

    if tokenizer is not None and hasattr(model, "_set_tokenizer"):
        model._set_tokenizer(tokenizer)

    logger.info(model)

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
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
