import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


def _set_context_if_supported(component, context):
    if component is not None and hasattr(component, "_set_context"):
        component._set_context(**context)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )
    else:
        device = config.inferencer.device

    dataloaders, batch_transforms, datasets_context, tokenizer = get_dataloaders(
        config, device, config.get("tokenizer")
    )

    model = instantiate(config.model).to(device)
    inference_context = next(iter(datasets_context.values()), None)
    if inference_context is not None:
        _set_context_if_supported(model, inference_context)
    if tokenizer is not None and hasattr(model, "_set_tokenizer"):
        model._set_tokenizer(tokenizer)

    criterion = instantiate(config.loss).to(device)
    metrics = instantiate(config.get("metrics"))

    inferencer = Inferencer(
        model=model,
        criterion=criterion,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        metrics=metrics,
        skip_model_load=False,
        datasets_context=datasets_context,
    )

    logs = inferencer.run_inference()
    for part, part_logs in logs.items():
        for key, value in part_logs.items():
            print(f"    {part + '_' + key:10s}: {value:.4f}")


if __name__ == "__main__":
    main()
