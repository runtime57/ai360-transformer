"""
Evaluate a checkpoint and log loss charts to Weights & Biases.

Runs two evaluations on the validation partition:

1. "val/loss" -- validation loss curve over batches (full model, anchor only).
2. "telescopic/loss_vs_depth" -- the telescopic chart from the paper
   (arXiv:2609.35769): cross-entropy / perplexity of the model truncated
   after each transformer block (depth = 1 .. num_layers), collected in a
   single forward pass (inference mode). Also logs AULB -- the area under
   the loss-vs-budget curve -- the headline metric of the paper.

The checkpoint is verified before evaluation: format detection (full
template checkpoint vs bare state_dict), missing/unexpected keys, shape
mismatches, NaN/Inf scan, and (if present) epoch/monitor_best/config
from the original training run.

Usage:
    python3 eval_wandb.py eval.checkpoint=path/to/model_best.pth
    python3 eval_wandb.py eval.checkpoint=path/to/model_best.pth \
        eval.max_batches=50 eval.run_name=my_run
"""

import math
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from src.datasets.collate import collate_fn
from src.metrics.telescopic_metric import telescopic_metric
from src.tokenizers import init_tokenizer
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)


def load_checkpoint(path):
    """Load a checkpoint, preferring the safe weights-only loader."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True), True
    except Exception:
        print(
            "WARNING: weights_only load failed; falling back to "
            "weights_only=False (pickle). Only do this for checkpoints "
            "from a trusted source!"
        )
        return torch.load(path, map_location="cpu", weights_only=False), False


def inspect_checkpoint(checkpoint, model):
    """
    Verify checkpoint structure and compatibility with the model.

    Returns:
        state_dict (dict): tensors to load into the model.
        report (dict): verification results.
    """
    report = {}

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        report["format"] = "full checkpoint (template)"
        state_dict = checkpoint["state_dict"]
        report["epoch"] = checkpoint.get("epoch")
        report["monitor_best"] = checkpoint.get("monitor_best")
        report["arch"] = checkpoint.get("arch")
        ckpt_config = checkpoint.get("config")
        if ckpt_config is not None:
            report["checkpoint_model_config"] = OmegaConf.to_yaml(
                OmegaConf.create(ckpt_config).model
            )
    elif isinstance(checkpoint, dict):
        report["format"] = "bare state_dict"
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint type: {type(checkpoint)}")

    report["num_tensors"] = len(state_dict)

    nan_inf = [
        key
        for key, value in state_dict.items()
        if torch.is_tensor(value) and value.is_floating_point()
        and (torch.isnan(value).any() or torch.isinf(value).any())
    ]
    report["nan_inf_tensors"] = nan_inf

    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(state_dict.keys())
    report["missing_keys"] = sorted(model_keys - ckpt_keys)
    report["unexpected_keys"] = sorted(ckpt_keys - model_keys)

    shape_mismatch = [
        f"{key}: checkpoint {tuple(state_dict[key].shape)} "
        f"vs model {tuple(model_state[key].shape)}"
        for key in model_keys & ckpt_keys
        if tuple(state_dict[key].shape) != tuple(model_state[key].shape)
    ]
    report["shape_mismatch"] = shape_mismatch

    # load only tensors that match the model exactly
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_keys
        and tuple(value.shape) == tuple(model_state[key].shape)
    }
    report["loaded_tensors"] = len(compatible)
    report["skipped_tensors"] = len(state_dict) - len(compatible)
    model.load_state_dict(compatible, strict=False)
    report["loaded_completely"] = (
        not report["missing_keys"]
        and not report["shape_mismatch"]
        and report["skipped_tensors"] == 0
    )

    return state_dict, report


def build_val_dataloader(config, logger):
    """Instantiate tokenizer and the validation dataloader only."""
    dataset = instantiate(config.datasets.val, tokenizer=None)
    tokenizer_config = config.get("tokenizer")

    tokenizer = None
    if tokenizer_config is not None:
        tokenizer = init_tokenizer(
            tokenizer_config,
            allow_build=True,
            log=logger,
            texts=dataset._get_train_text(),
        )

    if tokenizer is not None and hasattr(dataset, "_set_tokenizer"):
        dataset._set_tokenizer(tokenizer)

    dataloader = instantiate(
        config.dataloader,
        dataset=dataset,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
    )
    context = dataset._get_context()
    context.pop("part")

    return dataloader, context, tokenizer


@torch.no_grad()
def iterate_batches(dataloader, device, max_batches):
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        yield batch_idx, {
            key: value.to(device) for key, value in batch.items()
        }


@torch.no_grad()
def eval_loss_curve(model, loss_fn, dataloader, device, max_batches):
    """Per-batch validation loss of the full model (anchor only)."""
    model.eval()
    losses = []
    for _, batch in tqdm(
        iterate_batches(dataloader, device, max_batches),
        total=max_batches,
        desc="val loss curve",
    ):
        outputs = model(**batch)
        loss = loss_fn(**outputs, **batch)["loss"]
        losses.append(loss.item())
    return losses


@torch.no_grad()
def eval_loss_vs_prefix_depth(model, metric, dataloader, device, max_batches):
    """
    Telescopic evaluation: cross-entropy of the model truncated after each
    transformer block (one inference pass per batch, all depths at once).
    Returns rows of [depth, loss, perplexity].
    """
    model.eval()
    depth_losses = {}
    for _, batch in tqdm(
        iterate_batches(dataloader, device, max_batches),
        total=max_batches,
        desc="telescopic eval",
    ):
        outputs = model(seq=batch["seq"], inference=True)
        result = metric(all_logits=outputs["all_logits"], target=batch["target"])
        for key, value in result.items():
            depth = int(key.removeprefix("depth_"))
            depth_losses.setdefault(depth, []).append(value)

    rows = []
    for depth in sorted(depth_losses):
        mean_loss = float(sum(depth_losses[depth]) / len(depth_losses[depth]))
        rows.append([depth, mean_loss, math.exp(mean_loss)])
    return rows


def aulb(rows):
    """Area under the loss-vs-budget curve (trapezoid, budget normalized to [0, 1])."""
    depths = np.array([row[0] for row in rows], dtype=float)
    budget = depths / depths.max()
    return {
        "aulb_loss": float(np.trapezoid([row[1] for row in rows], budget)),
        "aulb_ppl": float(np.trapezoid([row[2] for row in rows], budget)),
    }


@hydra.main(version_base=None, config_path="src/configs", config_name="eval_wandb")
def main(config):
    set_random_seed(config.trainer.seed)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    import logging

    logger = logging.getLogger("eval_wandb")

    checkpoint_path = to_absolute_path(config.eval.checkpoint)
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Put model_best.pth into the project (e.g. saved/) and pass "
            "eval.checkpoint=<path>."
        )

    dataloader, context, tokenizer = build_val_dataloader(config, logger)

    model = instantiate(config.model).to(device)
    if hasattr(model, "_set_context"):
        model._set_context(**context)
    if tokenizer is not None and hasattr(model, "_set_tokenizer"):
        model._set_tokenizer(tokenizer)

    loss_fn = instantiate(config.loss).to(device)
    if hasattr(loss_fn, "_set_context"):
        loss_fn._set_context(**context)

    metric = telescopic_metric(ignore_index=context["ignore_class_id"])

    # --- checkpoint verification ---
    checkpoint, _ = load_checkpoint(checkpoint_path)
    _, report = inspect_checkpoint(checkpoint, model)
    model.to(device)

    for key, value in report.items():
        print(f"[checkpoint] {key}: {value}")

    # --- wandb ---
    import wandb

    wandb.login()
    wandb.init(
        project=config.eval.project_name,
        entity=config.eval.entity,
        name=config.eval.run_name,
        config=OmegaConf.to_container(config, resolve=True),
        mode=config.eval.mode,
        job_type="eval",
        notes=f"checkpoint: {checkpoint_path}",
    )

    report_html = "<h3>Checkpoint verification</h3><pre>" + "\n".join(
        f"{key}: {value}" for key, value in report.items()
    ) + "</pre>"
    wandb.log({"checkpoint_report": wandb.Html(report_html)})
    wandb.summary["checkpoint_path"] = checkpoint_path
    if report.get("epoch") is not None:
        wandb.summary["checkpoint_epoch"] = report["epoch"]
    if report.get("monitor_best") is not None:
        wandb.summary["checkpoint_monitor_best"] = report["monitor_best"]

    max_batches = config.eval.max_batches

    # --- chart 1: validation loss curve ---
    losses = eval_loss_curve(model, loss_fn, dataloader, device, max_batches)
    for batch_idx, loss in enumerate(losses):
        wandb.log({"val/loss": loss, "batch": batch_idx})
    mean_loss = sum(losses) / max(len(losses), 1)
    wandb.summary["val/loss_mean"] = mean_loss
    wandb.summary["val/perplexity"] = math.exp(mean_loss)
    print(f"val loss (mean over {len(losses)} batches): {mean_loss:.4f}")

    # --- chart 2: telescopic loss vs prefix depth + AULB ---
    if config.eval.eval_prefix_depths:
        rows = eval_loss_vs_prefix_depth(
            model, metric, dataloader, device, max_batches
        )
        table = wandb.Table(
            data=rows, columns=["prefix_depth", "loss", "perplexity"]
        )
        wandb.log(
            {
                "telescopic/loss_vs_prefix_depth": wandb.plot.line(
                    table,
                    "prefix_depth",
                    "loss",
                    title="Telescopic loss vs prefix depth",
                ),
                "telescopic/ppl_vs_prefix_depth": wandb.plot.line(
                    table,
                    "prefix_depth",
                    "perplexity",
                    title="Telescopic perplexity vs prefix depth",
                ),
                "telescopic/depth_table": table,
            }
        )
        for depth, depth_loss, depth_ppl in rows:
            wandb.summary[f"telescopic/loss_depth_{depth}"] = depth_loss
            wandb.summary[f"telescopic/ppl_depth_{depth}"] = depth_ppl
        for name, value in aulb(rows).items():
            wandb.summary[f"telescopic/{name}"] = value
        print("loss vs prefix depth:", rows)
        print("AULB:", aulb(rows))

    wandb.finish()


if __name__ == "__main__":
    main()
