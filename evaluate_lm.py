"""Evaluate TinyStories checkpoints with one frozen tokenizer and held-out split."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.collate import collate_fn
from src.utils.experiment_utils import autocast_context, resolve_device, save_results
from src.utils.io_utils import ROOT_PATH


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        config = OmegaConf.load(Path(path).with_name("config.yaml"))
    if not OmegaConf.is_config(config):
        config = OmegaConf.create(config)
    weights = {"state_dict": checkpoint["state_dict"], "epoch": checkpoint.get("epoch")}
    return weights, OmegaConf.to_container(config, resolve=True, throw_on_missing=True)


def vocabulary_path(config, override=None):
    return ROOT_PATH / (override or config["tokenizer"]["vocab_path"])


@torch.inference_mode()
def evaluate(model, dataloader, pad_id, device, dtype="float32", max_batches=None):
    """Accumulate token sums, then compute corpus NLL and exp(corpus NLL)."""
    model.eval()
    nll_sum, correct, token_count, batches = 0.0, 0, 0, 0
    total = min(len(dataloader), max_batches) if max_batches else len(dataloader)
    for batch in tqdm(dataloader, desc="quality", total=total):
        seq = batch["seq"].to(device)
        target = batch["target"].to(device)
        if target.ndim != 2:
            raise ValueError(
                "This evaluator expects next-token targets with shape [batch, time]"
            )
        with autocast_context(device, dtype):
            logits = model(seq)["logits"]
        valid = target != pad_id
        count = valid.sum().item()
        if count:
            nll_sum += F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                target.reshape(-1),
                ignore_index=pad_id,
                reduction="sum",
            ).item()
            correct += ((logits.argmax(-1) == target) & valid).sum().item()
            token_count += count
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    if not token_count:
        raise ValueError("Evaluation contains no non-padding target tokens")
    nll = nll_sum / token_count
    try:
        perplexity = math.exp(nll)
    except OverflowError:
        perplexity = math.inf
    return {
        "nll": nll,
        "perplexity": perplexity,
        "token_accuracy": correct / token_count,
        "evaluated_tokens": token_count,
        "evaluated_batches": batches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tokenization-workers", type=int, default=1)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--limit-stories", type=int)
    parser.add_argument("--tokenizer-vocab")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--output", default="outputs/attention_quality.json")
    args = parser.parse_args()
    if (
        args.batch_size <= 0
        or args.num_workers < 0
        or args.tokenization_workers <= 0
        or (args.max_batches is not None and args.max_batches <= 0)
        or (args.limit_stories is not None and args.limit_stories <= 0)
    ):
        parser.error(
            "Batch size and limits must be positive; num-workers must be non-negative"
        )
    try:
        device = resolve_device(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if args.dtype == "float16" and device.type == "cpu":
        parser.error("Use float32 or bfloat16 on CPU")

    reference_weights, reference = load_checkpoint(args.checkpoints[0])
    del reference_weights
    vocab_path = vocabulary_path(reference, args.tokenizer_vocab)
    vocab_hash = hashlib.sha256(vocab_path.read_bytes()).hexdigest()
    tokenizer = instantiate(reference["tokenizer"]["instance"])
    tokenizer.load_vocab(vocab_path)

    dataset_config = dict(reference["datasets"]["val"])
    if dataset_config["_target_"] != "src.datasets.TinyStoriesDataset":
        raise ValueError("Expected a TinyStories checkpoint")
    dataset_config.update(
        part=args.split,
        allow_download=args.allow_download,
        override=False,
        tokenization_workers=args.tokenization_workers,
    )
    if args.limit_stories is not None or args.split == "test":
        dataset_config["limit_stories"] = args.limit_stories
    dataset = instantiate(dataset_config, tokenizer=tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    report = {
        "split": args.split,
        "seq_length": dataset.max_seq_len,
        "tokenizer_sha256": vocab_hash,
        "device": str(device),
        "dtype": args.dtype,
        "dataset_config": dataset_config,
        "max_batches": args.max_batches,
        "results": [],
    }
    for path in args.checkpoints:
        checkpoint, config = load_checkpoint(path)
        other_vocab = vocabulary_path(config, args.tokenizer_vocab)
        if hashlib.sha256(other_vocab.read_bytes()).hexdigest() != vocab_hash:
            raise ValueError(
                "All checkpoints must use exactly the same frozen tokenizer"
            )
        if config["datasets"]["val"] != reference["datasets"]["val"]:
            raise ValueError(
                "All checkpoints must use the same validation data and context length"
            )

        decoder_config = config["model"]["decoder"]
        model = instantiate(config["model"])
        model._set_tokenizer(tokenizer)
        model._set_context(vocab_size=tokenizer.vocab_size)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device)
        row = {
            "checkpoint": str(path),
            "epoch": checkpoint.get("epoch"),
            "attention_type": decoder_config.get("attention_type", "softmax"),
            "use_rope": decoder_config.get("use_rope", False),
            "use_headwise_gate": decoder_config.get("use_headwise_gate", False),
            "use_elementwise_gate": decoder_config.get("use_elementwise_gate", False),
            "seed": config["trainer"]["seed"],
            **evaluate(
                model, loader, dataset.pad_id, device, args.dtype, args.max_batches
            ),
        }
        report["results"].append(row)
        print(json.dumps(row), flush=True)
        save_results(args.output, report)
        del model, checkpoint
    print(f"Saved {args.output} and {Path(args.output).with_suffix('.csv')}")


if __name__ == "__main__":
    main()
