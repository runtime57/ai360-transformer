"""Benchmark softmax/linear attention and complete decoders on synthetic inputs."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.model import TransformerDecoder
from src.model.transformer_utils.transformer_block import CausalSelfAttention
from src.utils.experiment_utils import autocast_context, resolve_device, save_results


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def measure(action, device, warmup, iterations, tokens):
    for _ in range(warmup):
        action()
    synchronize(device)
    baseline = None
    if device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(iterations):
        synchronize(device)
        start = time.perf_counter()
        action()
        synchronize(device)
        durations.append(time.perf_counter() - start)
    mean = statistics.mean(durations)
    peak = torch.cuda.max_memory_allocated(device) if baseline is not None else None
    return {
        "mean_ms": mean * 1000,
        "median_ms": statistics.median(durations) * 1000,
        "min_ms": min(durations) * 1000,
        "stdev_ms": statistics.stdev(durations) * 1000 if iterations > 1 else 0.0,
        "tokens_per_second": tokens / mean,
        "peak_cuda_memory_mib": peak / 2**20 if peak is not None else None,
        "extra_cuda_memory_mib": (peak - baseline) / 2**20
        if peak is not None
        else None,
    }


def benchmark_variant(args, device, component, attention_type, use_rope, length):
    torch.manual_seed(args.seed)
    common = dict(
        d_model=args.d_model,
        num_heads=args.num_heads,
        use_rope=use_rope,
        max_seq_len=length,
        attention_type=attention_type,
        linear_chunk_size=args.linear_chunk_size,
    )
    if component == "model":
        model = TransformerDecoder(
            **common,
            vocab_size=args.vocab_size,
            num_layers=args.num_layers,
            seq_dropout=args.dropout,
            ffn_dropout=args.dropout,
            attn_dropout=args.dropout,
        ).to(device)
        seq = torch.randint(args.vocab_size, (args.batch_size, length), device=device)
        target = torch.randint(args.vocab_size, seq.shape, device=device)
    else:
        model = CausalSelfAttention(**common, dropout=args.dropout).to(device)
        seq = torch.randn(args.batch_size, length, args.d_model, device=device)
        target = None

    def forward():
        with torch.inference_mode(), autocast_context(device, args.dtype):
            model(seq)

    def forward_backward():
        model.zero_grad(set_to_none=True)
        with autocast_context(device, args.dtype):
            output = model(seq)
            loss = (
                F.cross_entropy(output.reshape(-1, args.vocab_size), target.reshape(-1))
                if target is not None
                else output.float().square().mean()
            )
        loss.backward()
        model.zero_grad(set_to_none=True)

    def generate():
        with torch.inference_mode(), autocast_context(device, args.dtype):
            tokens = seq
            for _ in range(args.generation_tokens):
                logits = model(tokens[:, -length:])[:, -1]
                tokens = torch.cat((tokens, logits.argmax(-1, keepdim=True)), dim=-1)

    metadata = {
        "component": component,
        "attention_type": attention_type,
        "use_rope": use_rope,
        "seq_length": length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "parameters": sum(p.numel() for p in model.parameters()),
        "linear_chunk_size": args.linear_chunk_size,
    }
    rows = []
    modes = ["forward", "forward_backward"] if args.mode == "both" else [args.mode]
    if component == "model" and args.generation_tokens:
        modes.append("generation")
    for mode in modes:
        model.train(mode == "forward_backward")
        action = {
            "forward": forward,
            "forward_backward": forward_backward,
            "generation": generate,
        }[mode]
        tokens = args.batch_size * (
            args.generation_tokens if mode == "generation" else length
        )
        try:
            row = {
                **metadata,
                "mode": mode,
                "status": "ok",
                **measure(action, device, args.warmup, args.iterations, tokens),
            }
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            row = {**metadata, "mode": mode, "status": "oom"}
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    parser.add_argument(
        "--component", choices=["model", "attention", "both"], default="model"
    )
    parser.add_argument(
        "--mode", choices=["forward", "forward_backward", "both"], default="both"
    )
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 512, 1024])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--linear-chunk-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--generation-tokens", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/attention_benchmark.json")
    args = parser.parse_args()
    if (
        min(
            args.seq_lengths
            + [
                args.batch_size,
                args.d_model,
                args.num_layers,
                args.num_heads,
                args.vocab_size,
                args.linear_chunk_size,
                args.iterations,
            ]
        )
        <= 0
    ):
        parser.error(
            "Lengths, model dimensions, batch size and iterations must be positive"
        )
    if args.warmup < 0 or args.generation_tokens < 0:
        parser.error("warmup and generation_tokens must be non-negative")
    try:
        device = resolve_device(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if args.dtype == "float16" and device.type == "cpu":
        parser.error("Use float32 or bfloat16 on CPU")
    if (
        args.dtype == "bfloat16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        parser.error("This CUDA device does not support bfloat16")
    report = {
        "environment": {
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device),
            "cuda_version": torch.version.cuda,
            "cpu_threads": torch.get_num_threads(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        },
        "settings": vars(args),
        "notes": [
            "Synthetic inputs and identically seeded random weights; no quality estimate.",
            "forward_backward includes loss and backward, excludes optimizer/data loading/metrics.",
            "generation is greedy with a fixed context window and no KV/recurrent cache.",
            "Linear attention accumulates and normalizes in float32 under AMP.",
            "CUDA memory includes model weights; extra memory is relative to post-warmup allocation.",
        ],
        "results": [],
    }
    print(
        f"Python: {sys.executable} | PyTorch: {torch.__version__} | "
        f"Device: {device} ({report['environment']['device_name']})",
        flush=True,
    )
    components = (
        ["model", "attention"] if args.component == "both" else [args.component]
    )
    for component in components:
        for length in args.seq_lengths:
            for attention_type in ("softmax", "linear"):
                for use_rope in (False, True):
                    report["results"].extend(
                        benchmark_variant(
                            args, device, component, attention_type, use_rope, length
                        )
                    )
                    save_results(args.output, report)
    print(f"Saved {args.output} and {Path(args.output).with_suffix('.csv')}")


if __name__ == "__main__":
    main()
