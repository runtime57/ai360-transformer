import warnings

import hydra
import torch
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)

def set_context(obj, context):
    if context is not None and hasattr(obj, "_set_context"):
        obj._set_context(**context)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """

    p = "saved/testing/model_best.pth"
    try:
        c = torch.load(p, map_location="cpu", weights_only=False)
        print(p, "| epoch:", c.get("epoch"), "| monitor_best:", c.get("monitor_best"))
    except FileNotFoundError:
        print(p, "не найден")

    set_random_seed(config.inferencer.seed)

    tokenizer = None
    tokenizer_config = config.get("tokenizer")
    if tokenizer_config is not None:
        from src.tokenizers import init_tokenizer

        # Inference may load a vocabulary but must never fit one on evaluation data.
        tokenizer = init_tokenizer(tokenizer_config, allow_build=False)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms, context, tokenizer = get_dataloaders(
        config,
        device,
        tokenizer=tokenizer,
    )

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    print(model)
    # Set model context from the dataset
    model_context = (
        context.get("train")
        or context.get("inference")
        or context.get("val")
        or context.get("test")
    )

    set_context(model, model_context)

    # Attach the already initialized tokenizer
    if tokenizer is not None and hasattr(model, "_set_tokenizer"):
        model._set_tokenizer(tokenizer)

    print(model)

    # get metrics
    metrics = instantiate(config.metrics)

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    # ckpt = torch.load("saved/testing/model_best.pth", map_location="cpu",  weights_only=False)
    # sd = ckpt.get("state_dict", ckpt)
    # w = model.decoder.token_embedding.weight.detach().cpu()
    # print("model emb std:", w.std().item(), "| ckpt emb std:", sd["decoder.token_embedding.weight"].std().item())
    # print("equal:", torch.allclose(w, sd["decoder.token_embedding.weight"]))

    logs = inferencer.run_inference()
    import matplotlib.pyplot as plt


    part = "test" if "test" in logs else "val" if "val" in logs else None

    if part is None:
        raise RuntimeError(
            f"Expected a 'test' or 'val' partition, got: {list(logs)}"
        )

    part_logs = logs[part]

    prefix_losses = []
    for key, value in part_logs.items():
        if key.startswith("loss_prefix_"):
            depth = int(key.rsplit("_", 1)[1])
            prefix_losses.append((depth, float(value)))

    prefix_losses.sort(key=lambda item: item[0])

    if not prefix_losses:
        raise RuntimeError(
            f"No loss_prefix_N metrics found in logs[{part!r}]: "
            f"{part_logs}"
        )

    depths = [depth for depth, _ in prefix_losses]
    losses = [loss for _, loss in prefix_losses]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        depths,
        losses,
        marker="o",
        markersize=7,
        linewidth=2,
    )

    ax.set_xticks(depths)
    ax.set_xlabel("Prefix length (number of Transformer blocks)")
    ax.set_ylabel("Mean cross-entropy loss")
    ax.set_title(f"Loss vs. Transformer prefix depth — {part}")

    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    plot_path = save_path / f"{part}_loss_vs_prefix_length.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")

    print(f"\nPrefix loss plot saved to: {plot_path}")
    print("Losses: ", losses, '\n')

    plt.show()
    plt.close(fig)

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
