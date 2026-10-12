import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from omegaconf import OmegaConf

from src.utils.attention_flops import estimate_attention_matmul_flops


VARIANTS = [("softmax", False), ("softmax", True), ("linear", False), ("linear", True)]
COLORS = dict(zip(VARIANTS, ["#2563eb", "#0891b2", "#ea580c", "#9333ea"]))
LINESTYLES = dict.fromkeys(VARIANTS, "-")
HATCHES = dict.fromkeys(VARIANTS, "")
LOG_VALUE = re.compile(r" - INFO -\s+(\w+)\s*:\s*([-+\d.eE]+)\s*$")


def configure_variant_style(style):
    palette = ["#2563eb", "#0891b2", "#ea580c", "#9333ea"]
    for key, default_color in zip(VARIANTS, palette):
        grouped = style == "attention"
        COLORS[key] = (
            {"softmax": "#2563eb", "linear": "#ea580c"}[key[0]]
            if grouped
            else default_color
        )
        LINESTYLES[key] = "--" if grouped and key[1] else "-"
        HATCHES[key] = "///" if grouped and key[1] else ""


def boolean(value):
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    raise ValueError(f"Invalid boolean: {value!r}")


def variant(row):
    key = row["attention_type"], boolean(row["use_rope"])
    if key not in COLORS:
        raise ValueError(f"Unknown attention variant: {key}")
    return key


def label(key):
    attention, rope = key
    return f"{attention} {'with' if rope else 'without'} RoPE"


def number(row, field):
    value = row.get(field)
    return float(value) if value not in (None, "") else np.nan


def load_report(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        metadata = json.loads(path.read_text(encoding="utf-8"))
        return metadata["results"], metadata
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    # JSON keeps the evaluation split, device and model settings missing in CSV.
    sidecar = path.with_suffix(".json")
    metadata = (
        json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    )
    return rows, metadata


def read_legacy_history(path, epoch_len, log_step):
    """Recover epoch summaries, preserving the old train-window semantics."""
    records, current = [], None
    training_start = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "Training budget:" in line:
            training_start = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f")
        match = LOG_VALUE.search(line)
        if not match:
            continue
        name, value = match.groups()
        if name == "epoch":
            if current:
                records.append(current)
            timestamp = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f")
            current = {"epoch": int(float(value))}
            current["elapsed_seconds"] = (
                (timestamp - training_start).total_seconds() if training_start else ""
            )
        elif current is not None:
            current[name] = float(value)
    if current:
        records.append(current)

    rows = []
    for record in records:
        epoch = record["epoch"]
        for split in ("train", "val", "test"):
            prefix = "" if split == "train" else f"{split}_"
            metrics = {
                key[len(prefix) :]: value
                for key, value in record.items()
                if key not in {"epoch", "elapsed_seconds"}
                and (
                    key.startswith(prefix)
                    if prefix
                    else not key.startswith(("val_", "test_"))
                )
            }
            if not metrics:
                continue
            # The previous trainer logged at batch indices 0, log_step, ...
            step = epoch * epoch_len
            if split == "train":
                step = (
                    (epoch - 1) * epoch_len
                    + ((epoch_len - 1) // log_step) * log_step
                    + 1
                )
            rows.append(
                dict(
                    epoch=epoch,
                    step=step,
                    split=split,
                    scope="legacy_window" if split == "train" else "epoch",
                    # The timestamp is an epoch summary, not a per-window measurement.
                    elapsed_seconds=record["elapsed_seconds"]
                    if split != "train"
                    else "",
                    **metrics,
                )
            )
    return rows


def load_histories(runs_dir):
    histories = []
    for config_path in sorted(Path(runs_dir).glob("*/config.yaml")):
        config = OmegaConf.load(config_path)
        decoder = config.model.decoder
        key = str(decoder.get("attention_type", "softmax")), bool(
            decoder.get("use_rope", False)
        )
        epoch_len = config.trainer.get("epoch_len")
        history_path = config_path.parent / "metrics.csv"
        if history_path.is_file():
            with history_path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
        else:
            history_path = config_path.parent / "info.log"
            if not history_path.is_file():
                print(f"No metric history: {config_path.parent}")
                continue
            # Full-pass runs can recover epoch coordinates even without epoch_len.
            rows = read_legacy_history(
                history_path, epoch_len or 1, config.trainer.log_step
            )
        if not rows:
            continue
        histories.append(
            dict(
                run=config_path.parent.name,
                key=key,
                seed=config.trainer.seed,
                epoch_len=epoch_len,
                config=config,
                rows=rows,
                source=str(history_path),
            )
        )
    return sorted(histories, key=lambda history: VARIANTS.index(history["key"]))


def save_figure(fig, output_dir, name):
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def decorate(ax, ylabel):
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def bars(ax, rows, field, scale=1, errors=False, include_rope=True):
    values = [number(row, field) * scale for row in rows]
    keys = [
        variant(row) if include_rope else (row["attention_type"], False) for row in rows
    ]
    uncertainty = [number(row, "stdev_ms") for row in rows] if errors else None
    patches = ax.bar(
        range(len(rows)),
        values,
        color=[COLORS[key] for key in keys],
        yerr=uncertainty,
        capsize=4 if errors else 0,
        edgecolor="white",
    )
    for patch, key in zip(patches, keys):
        patch.set_hatch(HATCHES[key] if include_rope else "")
    names = [
        label(key).replace(" ", "\n", 1) if include_rope else key[0] for key in keys
    ]
    if len(set(keys)) < len(rows):
        names = [
            f"{name}\nseed {row.get('seed', '?')}" for name, row in zip(names, rows)
        ]
    ax.set_xticks(range(len(rows)), names)
    finite = [value for value in values if np.isfinite(value)]
    if finite:
        top = max(
            finite
            + [
                value + error
                for value, error in zip(values, uncertainty or [])
                if np.isfinite(value + error)
            ]
        )
        ax.set_ylim(0, top * 1.25 if top else 1)
    for index, value in enumerate(values):
        if np.isfinite(value):
            offset = (
                uncertainty[index]
                if uncertainty and np.isfinite(uncertainty[index])
                else 0
            )
            ax.annotate(
                f"{value:.2f}",
                (index, value + offset),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )


def plot_quality(rows, metadata, output_dir):
    rows = sorted(rows, key=lambda row: VARIANTS.index(variant(row)))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for ax, field, title, scale in zip(
        axes,
        ["nll", "perplexity", "token_accuracy"],
        [
            "NLL ↓",
            "Perplexity ↓",
            "Token accuracy, % ↑",
        ],
        [1, 1, 100],
    ):
        bars(ax, rows, field, scale)
        decorate(ax, title)
    split = metadata.get("split")
    title = (
        f"TinyStories: quality on the {split} split"
        if split
        else "TinyStories: model quality"
    )
    fig.suptitle(title, fontsize=14)
    save_figure(fig, output_dir, "quality")


def benchmark_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["component"],
            int(row["batch_size"]),
            row["dtype"],
            int(row["linear_chunk_size"]),
        )
        groups[key].append(row)
    return groups


def plot_benchmarks(rows, output_dir, metadata=None):
    generation_tokens = (metadata or {}).get("settings", {}).get("generation_tokens")
    for (component, batch_size, dtype, chunk_size), group in benchmark_groups(
        rows
    ).items():
        modes = [
            mode
            for mode in ("forward", "forward_backward", "generation")
            if any(row["mode"] == mode for row in group)
        ]
        fig, axes = plt.subplots(
            2,
            len(modes),
            figsize=(5.5 * len(modes), 8),
            squeeze=False,
            layout="constrained",
        )
        for column, mode in enumerate(modes):
            mode_rows = [row for row in group if row["mode"] == mode]
            lengths = sorted({int(row["seq_length"]) for row in mode_rows})
            units = (
                [
                    f"Time for {generation_tokens} new tokens (ms) ↓"
                    if generation_tokens
                    else "Generation time (ms) ↓",
                    "Generated tokens/s ↑",
                ]
                if mode == "generation"
                else ["Latency (ms) ↓", "Throughput (k tokens/s) ↑"]
            )
            for ax, field, unit in zip(
                axes[:, column],
                ["mean_ms", "tokens_per_second"],
                units,
            ):
                scale = (
                    0.001
                    if field == "tokens_per_second" and mode != "generation"
                    else 1
                )
                successful = [row for row in mode_rows if row["status"] == "ok"]
                if not any(np.isfinite(number(row, field)) for row in successful):
                    ax.text(
                        0.5, 0.5, "No measurements", ha="center", transform=ax.transAxes
                    )
                elif len(lengths) == 1:
                    successful.sort(key=lambda row: VARIANTS.index(variant(row)))
                    bars(ax, successful, field, scale, errors=field == "mean_ms")
                else:
                    seen = set()
                    for key in VARIANTS:
                        selected = sorted(
                            [row for row in mode_rows if variant(row) == key],
                            key=lambda row: int(row["seq_length"]),
                        )
                        for row in selected:
                            identity = key, int(row["seq_length"])
                            if identity in seen:
                                raise ValueError(
                                    f"Duplicate benchmark measurement: {identity}, {mode}"
                                )
                            seen.add(identity)
                        x = [int(row["seq_length"]) for row in selected]
                        y = [
                            number(row, field) * scale
                            if row["status"] == "ok"
                            else np.nan
                            for row in selected
                        ]
                        uncertainty = (
                            [
                                number(row, "stdev_ms")
                                if row["status"] == "ok"
                                else np.nan
                                for row in selected
                            ]
                            if field == "mean_ms"
                            else None
                        )
                        ax.errorbar(
                            x,
                            y,
                            yerr=uncertainty,
                            capsize=4,
                            marker="o",
                            color=COLORS[key],
                            label=label(key),
                            linestyle=LINESTYLES[key],
                        )
                    ax.set_xlabel(
                        "Context window (tokens)"
                        if mode == "generation"
                        else "Sequence length (tokens)"
                    )
                    ax.set_xticks(lengths)
                    ax.legend(fontsize=8)
                    ax.set_ylim(bottom=0)
                decorate(ax, unit)
                ax.set_title(
                    {
                        "forward": "Forward",
                        "forward_backward": "Forward + backward",
                        "generation": "Generation",
                    }[mode]
                )
        lengths_text = ", ".join(
            str(length) for length in sorted({int(row["seq_length"]) for row in group})
        )
        notes = ["Latency error bars: timing SD."]
        if "forward_backward" in modes:
            notes.append("Forward + backward excludes optimizer and data loading.")
        if "generation" in modes:
            notes.append(
                "Generation uses a fixed context window without KV/recurrent cache."
            )
            notes.append("Synthetic inputs and random weights.")
        fig.suptitle(
            f"{component.capitalize()}: batch {batch_size}, {dtype}, sequence length {lengths_text}\n"
            + "\n".join(notes),
            fontsize=13,
        )
        save_figure(
            fig,
            output_dir,
            f"benchmark_{component}_b{batch_size}_{dtype}_c{chunk_size}",
        )


def plot_flops(rows, metadata, output_dir, seq_lengths=None, axis_scale="log"):
    if axis_scale not in {"log", "linear"}:
        raise ValueError("FLOPs axis scale must be log or linear")
    settings = metadata.get("settings")
    if not settings:
        raise ValueError(
            "--flops requires benchmark JSON settings or a CSV JSON sidecar"
        )
    estimates = []
    seen = set()
    if seq_lengths is not None:
        if not seq_lengths or any(length <= 0 for length in seq_lengths):
            raise ValueError("FLOPs context lengths must be positive")
        seq_lengths = sorted(set(seq_lengths))
        rows = [{**row, "seq_length": length} for row in rows for length in seq_lengths]
    for row in rows:
        identity = (
            row["attention_type"],
            row["component"],
            int(row["seq_length"]),
            int(row["batch_size"]),
            row["dtype"],
            int(row["linear_chunk_size"]),
        )
        if identity in seen:
            continue  # Timing modes and RoPE variants have the same matmul budget.
        seen.add(identity)
        key = variant(row)
        generation_tokens = (
            int(settings.get("generation_tokens", 0))
            if row["component"] == "model"
            else 0
        )
        parameters = dict(
            d_model=int(settings["d_model"]),
            num_layers=int(settings["num_layers"]),
            num_heads=int(settings["num_heads"]),
            vocab_size=int(settings["vocab_size"]),
            seq_length=int(row["seq_length"]),
            batch_size=int(row["batch_size"]),
            attention_type=key[0],
            linear_chunk_size=int(row["linear_chunk_size"]),
            generation_tokens=generation_tokens,
            component=row["component"],
        )
        estimate = estimate_attention_matmul_flops(**parameters)
        estimates.append(
            dict(
                **parameters,
                dtype=row["dtype"],
                **estimate,
                attention_core_gflops=estimate["attention_core_matmul_flops"] / 1e9,
                forward_gflops=estimate["forward_matmul_flops"] / 1e9,
                generation_gflops=estimate["generation_matmul_flops"] / 1e9
                if generation_tokens
                else None,
            )
        )
    if not estimates:
        raise ValueError("No configurations found for FLOPs estimation")
    groups = benchmark_groups(estimates)
    for (component, batch, dtype, chunk), group in groups.items():
        panels = [
            ("attention_core_gflops", "Forward: attention core"),
            (
                "forward_gflops",
                "Forward: full model"
                if component == "model"
                else "Forward: attention module",
            ),
        ]
        generation_tokens = group[0]["generation_tokens"]
        if generation_tokens:
            panels.append(
                ("generation_gflops", f"Generation: {generation_tokens} new tokens")
            )
        fig, axes = plt.subplots(
            1,
            len(panels),
            figsize=(5.8 * len(panels), 5),
            squeeze=False,
            layout="constrained",
        )
        lengths = sorted({row["seq_length"] for row in group})
        for ax, (field, title) in zip(axes.flat, panels):
            if len(lengths) == 1:
                selected = sorted(
                    group,
                    key=lambda row: ("softmax", "linear").index(row["attention_type"]),
                )
                bars(ax, selected, field, include_rope=False)
            else:
                for attention in ("softmax", "linear"):
                    selected = sorted(
                        [row for row in group if row["attention_type"] == attention],
                        key=lambda row: row["seq_length"],
                    )
                    if not selected:
                        continue
                    ax.plot(
                        [row["seq_length"] for row in selected],
                        [row[field] for row in selected],
                        color=COLORS[(attention, False)],
                        linestyle="-",
                        marker="o",
                        label=attention,
                    )
                ax.set_xlabel("Context window (tokens)")
                if axis_scale == "log":
                    ax.set_xscale("log", base=2)
                    ax.set_yscale("log")
                    ax.set_xticks(lengths, [str(length) for length in lengths])
                else:
                    ax.set_xlim(left=0)
                    ax.set_ylim(bottom=0)
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
                ax.legend(fontsize=8)
            decorate(ax, "Estimated matmul GFLOPs ↓")
            ax.set_title(title)
        fig.suptitle(
            f"Analytical matmul FLOPs: d_model={settings['d_model']}, "
            f"layers={group[0]['effective_layers']}, heads={settings['num_heads']}\n"
            f"context={', '.join(map(str, lengths))}, batch={batch}, vocab={settings['vocab_size']}, "
            f"chunk={chunk}, {dtype}",
            fontsize=13,
        )
        fig.supxlabel(
            "Multiply-add = 2 FLOPs. Softmax: causal triangle; linear: actual chunk matmuls.\n"
            "RoPE and elementwise operations excluded; RoPE variants merged. Generation without cache."
            + (f" Context and FLOPs axes: {axis_scale} scale." if len(lengths) > 1 else ""),
            fontsize=9,
        )
        name = "estimated_matmul_flops"
        if len(groups) > 1:
            name += f"_{component}_b{batch}_{dtype}_c{chunk}"
        if axis_scale == "linear":
            name += "_linear"
        save_figure(fig, output_dir, name)
    with (output_dir / "estimated_matmul_flops.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(estimates[0]))
        writer.writeheader()
        writer.writerows(estimates)
    report = dict(
        settings=settings,
        requested_seq_lengths=seq_lengths,
        results=estimates,
        convention="1 multiply-add = 2 FLOPs; 1 GFLOP = 1e9 operations",
        formulas={
            "symbols": "B=batch, L=context, D=d_model, H=heads, d=D/H, N=layers, V=vocab, G=new tokens, C=min(chunk,L), Lp=ceil(L/C)*C",
            "softmax_causal_core": "2*N*B*D*L*(L+1)",
            "softmax_dense_reference_core": "4*N*B*D*L^2",
            "linear_chunk_core": "4*N*B*D*Lp*(d+C)",
            "attention_projections": "8*N*B*L*D^2",
            "ffn": "16*N*B*L*D^2 (model only, hidden_dim=4D)",
            "vocab_projection": "2*B*L*D*V (model only, logits at every position)",
            "generation": "G*forward (entire batch, fixed full context, no cache)",
        },
        notes=[
            "Analytical forward matmul budget, not a profiler result or executed GPU instruction count.",
            "Softmax uses only causal pairs as a conservative logical count. Fused kernels can perform extra work within tiles.",
            "A full dense softmax reference is also exported, not used in the main chart.",
            "Linear counts all four dense chunk matmuls, including the padded last chunk and zero/masked entries.",
            "Attention core excludes QKV/output projections; the full model includes them, FFN and the full-position vocabulary head.",
            "RoPE, ELU, softmax, scans, normalization, biases, activations, dropout, loss, backward, optimizer, argmax and memory operations are excluded.",
            "RoPE variants have equal matmul FLOPs and are merged into one row per attention type/context. FP16 and FP32 are counted equally but can have different execution costs.",
            "requested_seq_lengths, when present, overrides context only for analytical FLOPs; no timing measurements are extrapolated.",
        ],
    )
    (output_dir / "estimated_matmul_flops.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def selected_history(history, split, train_points="window"):
    rows = [row for row in history["rows"] if row["split"] == split]
    if split != "train":
        return rows
    if train_points == "epoch":
        epochs = [row for row in rows if row["scope"] == "epoch"]
        if rows and not epochs:
            raise ValueError(
                "Full-epoch train averages are unavailable in legacy logs; "
                "use --train-points last-window to compare with older runs"
            )
        return epochs
    windows = [row for row in rows if row["scope"] in {"window", "legacy_window"}]
    if train_points == "last-window":
        if rows and not windows:
            raise ValueError("Train logging windows are unavailable in this history")
        final_windows = {}
        for row in windows:
            epoch = int(number(row, "epoch"))
            previous = final_windows.get(epoch)
            if previous is None or number(row, "step") > number(previous, "step"):
                final_windows[epoch] = row
        return [final_windows[epoch] for epoch in sorted(final_windows)]
    return windows or rows


def plot_histories(
    histories, output_dir, x_axis, train_points="window", perplexity_scale="log"
):
    panels = [
        ("train", "loss", "Train loss ↓"),
        ("train", "perplexity", "Train perplexity ↓"),
        ("train", "token_accuracy", "Train accuracy, % ↑"),
        ("val", "loss", "Validation loss ↓"),
        ("val", "perplexity", "Validation perplexity ↓"),
        ("val", "token_accuracy", "Validation accuracy, % ↑"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for ax, (split, field, title) in zip(axes.flat, panels):
        for history in histories:
            rows = selected_history(history, split, train_points)
            if x_axis == "time" and any(
                row["scope"] == "legacy_window" for row in rows
            ):
                raise ValueError(
                    "Legacy train windows have no timestamps; use --x-axis step or epoch"
                )
            x_field = {"epoch": "epoch", "step": "step", "time": "elapsed_seconds"}[
                x_axis
            ]
            if (
                x_axis == "step"
                and history["epoch_len"] is None
                and "info.log" in history["source"]
            ):
                raise ValueError(
                    "Legacy full-pass logs have no step count; use --x-axis epoch"
                )
            x = [number(row, x_field) for row in rows]
            if (
                x_axis == "epoch"
                and history["epoch_len"] is not None
                and train_points == "window"
                and split == "train"
            ):
                x = [number(row, "step") / history["epoch_len"] for row in rows]
            y = [
                number(row, field) * (100 if field == "token_accuracy" else 1)
                for row in rows
            ]
            name = label(history["key"])
            if len({entry["key"] for entry in histories}) != len(histories):
                name += f" (seed {history['seed']})"
            ax.plot(
                x,
                y,
                marker="o" if len(rows) < 20 else None,
                linewidth=1.8,
                markersize=4,
                color=COLORS[history["key"]],
                linestyle=LINESTYLES[history["key"]],
                label=name,
            )
        ax.set_title(title)
        if field == "perplexity":
            ax.set_yscale(perplexity_scale)
        ax.set_xlabel(
            {
                "epoch": "Epoch",
                "step": "Processed training batches",
                "time": "Training and validation time (s)",
            }[x_axis]
        )
        decorate(ax, "")
        handles, names = ax.get_legend_handles_labels()
        legend_lines = [
            Line2D(
                [],
                [],
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=2,
            )
            for line in handles
        ]
        ax.legend(legend_lines, names, fontsize=8, handlelength=4.5)
    if train_points == "epoch":
        note = "Train: full-epoch averages; validation: evaluated-batch averages."
    elif train_points == "last-window":
        note = "Train: final logging window of each epoch; validation: evaluated-batch averages."
    else:
        legacy = any(
            row["scope"] == "legacy_window"
            for history in histories
            for row in history["rows"]
        )
        note = (
            "Legacy logs: train shows the final logging window; validation averages the evaluated batches."
            if legacy
            else "Train: logging-window averages; validation: evaluated-batch averages."
        )
    if perplexity_scale == "log":
        note += " PPL: log scale."
    configurations = []
    for history in histories:
        cfg = history["config"]
        decoder = cfg.model.decoder
        dtype = (
            str(cfg.trainer.get("amp_dtype", "float16"))
            if cfg.trainer.get("use_amp", False)
            else "float32"
        )
        settings = (
            f"d_model={decoder.d_model}, layers={decoder.num_layers}, "
            f"heads={decoder.num_heads}, context={cfg.max_seq_len}, "
            f"batch={cfg.dataloader.batch_size}"
        )
        if cfg.get("tokenizer") is not None:
            settings += f", vocab={cfg.tokenizer.instance.vocab_size}"
        configurations.append(f"{settings}, {dtype}")
    configuration_text = "\n".join(dict.fromkeys(configurations))
    fig.suptitle(
        f"TinyStories: learning curves\n{configuration_text}\n{note}", fontsize=13
    )
    save_figure(fig, output_dir, "training_curves")

    totals = []
    for history in histories:
        times = [number(row, "elapsed_seconds") for row in history["rows"]]
        finite = [value for value in times if np.isfinite(value)]
        if finite:
            totals.append(
                dict(
                    attention_type=history["key"][0],
                    use_rope=history["key"][1],
                    elapsed_seconds=max(finite),
                    seed=history["seed"],
                )
            )
    if totals:
        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        bars(ax, totals, "elapsed_seconds")
        decorate(ax, "Time (s) ↓")
        ax.set_title(
            "Training and validation time through the last logged point\n"
            "Excludes data preparation and initial tokenization"
        )
        save_figure(fig, output_dir, "training_time")


def training_batch_times(histories):
    """Subtract recorded evaluation intervals from cumulative wall-clock time."""
    summaries = []
    for history in histories:
        epochs = defaultdict(list)
        for row in history["rows"]:
            epochs[int(number(row, "epoch"))].append(row)
        previous_end, previous_step, train_seconds = 0.0, 0, 0.0
        for expected_epoch, (epoch, rows) in enumerate(sorted(epochs.items()), start=1):
            train = [
                row
                for row in rows
                if row["split"] == "train" and row["scope"] == "epoch"
            ]
            if len(train) != 1 or epoch != expected_epoch:
                raise ValueError(
                    f"Complete native epoch timing is required: {history['source']}, epoch {epoch}"
                )
            end, step = number(train[0], "elapsed_seconds"), int(
                number(train[0], "step")
            )
            if not np.isfinite(end) or end < previous_end or step <= previous_step:
                raise ValueError(
                    f"Invalid epoch timing or batch count: {history['source']}"
                )
            if any(
                row["split"] == "train" and number(row, "step") > step for row in rows
            ):
                raise ValueError(
                    f"Incomplete training epoch: {history['source']}, epoch {epoch}"
                )
            train_seconds += end - previous_end
            evaluation = [
                row
                for row in rows
                if row["split"] != "train" and row["scope"] == "epoch"
            ]
            evaluation_times = [number(row, "elapsed_seconds") for row in evaluation]
            if any(not np.isfinite(time) or time < end for time in evaluation_times):
                raise ValueError(f"Invalid evaluation timing: {history['source']}")
            if not evaluation and any(
                part != "train" for part in history["config"].get("datasets", {})
            ):
                raise ValueError(
                    f"Missing evaluation timing: {history['source']}, epoch {epoch}"
                )
            previous_end = max([end, *evaluation_times])
            previous_step = step
        summaries.append(
            dict(
                attention_type=history["key"][0],
                use_rope=history["key"][1],
                seed=history["seed"],
                training_batches=previous_step,
                training_seconds=train_seconds,
                training_ms_per_batch=1000 * train_seconds / previous_step,
                training_source=history["source"],
            )
        )
    return summaries


def validate_training_generation(histories, benchmark, metadata):
    settings = metadata.get("settings")
    if not settings or not settings.get("generation_tokens", 0):
        raise ValueError(
            "Training/generation comparison requires benchmark JSON settings (or a CSV JSON sidecar)"
        )
    selected, configurations = [], []
    for history in histories:
        cfg, key = history["config"], history["key"]
        decoder = cfg.model.decoder
        expected = {
            "d_model": int(decoder.d_model),
            "num_layers": int(decoder.num_layers),
            "num_heads": int(decoder.num_heads),
            "linear_chunk_size": int(decoder.get("linear_chunk_size", 64)),
            "vocab_size": int(cfg.tokenizer.instance.vocab_size),
            "dtype": str(cfg.trainer.get("amp_dtype", "float16"))
            if cfg.trainer.get("use_amp", False)
            else "float32",
            "device": str(cfg.trainer.device),
            "seed": int(history["seed"]),
        }
        for field, value in expected.items():
            if settings.get(field) != value:
                raise ValueError(
                    f"Config mismatch for {label(key)}: {field}, training={value}, benchmark={settings.get(field)}"
                )
        for field in ("seq_dropout", "ffn_dropout", "attn_dropout"):
            if float(decoder.get(field, 0.1)) != settings.get("dropout"):
                raise ValueError(f"Dropout mismatch for {label(key)}: {field}")
        length, batch = int(cfg.max_seq_len), int(cfg.dataloader.batch_size)
        matches = [
            row
            for row in benchmark
            if row["component"] == "model"
            and row["mode"] == "generation"
            and variant(row) == key
            and int(row["seq_length"]) == length
            and int(row["batch_size"]) == batch
            and row["dtype"] == expected["dtype"]
        ]
        if len(matches) != 1 or matches[0]["status"] != "ok":
            raise ValueError(
                f"Need one successful matching generation measurement for {label(key)}: context={length}, batch={batch}"
            )
        row = matches[0]
        if int(row["linear_chunk_size"]) != expected["linear_chunk_size"]:
            raise ValueError(f"Row chunk-size mismatch for {label(key)}")
        seconds = number(row, "mean_ms") / 1000
        measured_tokens = number(row, "tokens_per_second") * seconds
        if (
            not np.isfinite(seconds)
            or seconds <= 0
            or not np.isclose(measured_tokens, batch * settings["generation_tokens"])
        ):
            raise ValueError(f"Inconsistent generation token count for {label(key)}")
        selected.append(row)
        configurations.append(
            dict(
                attention_type=key[0],
                use_rope=key[1],
                **expected,
                seq_length=length,
                batch_size=batch,
                generation_tokens=settings["generation_tokens"],
            )
        )
    if len({history["key"] for history in histories}) != len(histories):
        raise ValueError("Training/generation comparison requires one run per variant")
    signatures = {
        tuple(
            row[field]
            for field in (
                "d_model",
                "num_layers",
                "num_heads",
                "vocab_size",
                "seq_length",
                "batch_size",
                "dtype",
                "device",
                "seed",
                "linear_chunk_size",
            )
        )
        for row in configurations
    }
    if len(signatures) != 1:
        raise ValueError("Training variants have different comparison configurations")
    return selected, configurations


def plot_training_generation(histories, benchmark, metadata, output_dir):
    generation, configurations = validate_training_generation(
        histories, benchmark, metadata
    )
    summaries = training_batch_times(histories)
    if len({row["training_batches"] for row in summaries}) != 1:
        raise ValueError("Training variants have different batch counts")
    for summary, row in zip(summaries, generation):
        summary.update(
            generation_tokens_per_second=number(row, "tokens_per_second"),
            generation_mean_ms=number(row, "mean_ms"),
            seq_length=int(row["seq_length"]),
            batch_size=int(row["batch_size"]),
            dtype=row["dtype"],
            generation_tokens=metadata["settings"]["generation_tokens"],
        )
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5), layout="constrained")
    for ax, field, title, mode in zip(
        axes,
        ["training_ms_per_batch", "generation_mean_ms", "generation_tokens_per_second"],
        ["Mean batch time (ms) ↓", "Latency (ms) ↓", "Generated tokens/s ↑"],
        ["Training", "Generation latency", "Generation throughput"],
    ):
        bars(ax, summaries, field)
        decorate(ax, title)
        ax.set_title(mode)
    cfg = configurations[0]
    fig.suptitle(
        f"Training and generation speed: {summaries[0]['training_batches']:,} training batches\n"
        f"d_model={cfg['d_model']}, layers={cfg['num_layers']}, heads={cfg['num_heads']}, "
        f"context={cfg['seq_length']}, batch={cfg['batch_size']}, {cfg['dtype']}",
        fontsize=13,
    )
    fig.supxlabel(
        f"Training: validation excluded. Generation: {cfg['generation_tokens']} new tokens per sequence; no cache.\n"
        "Solid bars: without RoPE. Hatched bars: with RoPE.",
        fontsize=9,
    )
    save_figure(fig, output_dir, "training_generation_speed")
    with (output_dir / "training_generation_speed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output_dir / "training_generation_config.json").write_text(
        json.dumps(
            {
                "configurations": configurations,
                "benchmark_settings": metadata["settings"],
                "benchmark_environment": metadata.get("environment", {}),
                "notes": [
                    "Training time excludes recorded validation/test intervals and initial data preparation.",
                    "It includes optimizer, metrics, logging and overhead between epochs (including checkpoint saving).",
                    "Generation throughput counts new tokens across the whole batch; inputs and weights are synthetic.",
                    "Mean training time is total retained wall time divided by processed batches, including AMP-skipped updates.",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def export_histories(histories, output_dir):
    rows = []
    for history in histories:
        for row in history["rows"]:
            rows.append(
                dict(
                    run=history["run"],
                    attention_type=history["key"][0],
                    use_rope=history["key"][1],
                    seed=history["seed"],
                    source=history["source"],
                    **row,
                )
            )
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with (output_dir / "training_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    quality,
    benchmark,
    histories,
    metadata,
    output_dir,
    inputs,
    train_points="window",
    perplexity_scale="log",
    training_generation=False,
    flops=False,
):
    lines = ["# Сравнение вариантов attention", "", "Источники:", ""]
    lines += [f"- `{path}`" for path in inputs if path]
    if quality:
        lines += [
            "",
            f"Качество на split `{metadata.get('split', 'не указан')}`:",
            "",
            "| Вариант | Seed | NLL | Perplexity | Accuracy, % | Токены |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(quality, key=lambda row: VARIANTS.index(variant(row))):
            lines.append(
                f"| {label(variant(row))} | {row['seed']} | {number(row, 'nll'):.4f} | "
                f"{number(row, 'perplexity'):.2f} | {100 * number(row, 'token_accuracy'):.2f} | "
                f"{number(row, 'evaluated_tokens'):.0f} |"
            )
    if benchmark:
        lines += [
            "",
            "Бенчмарк вычислений модели на синтетических данных, со случайными весами.",
            "Forward + backward включает loss, исключает optimizer, загрузку данных и метрики.",
            "Усы на графиках времени — стандартное отклонение замеров, не доверительный интервал.",
            "",
            "| Вариант | Component | L | Batch | Dtype | Режим | Время, мс | Токены/с | CUDA MiB |",
            "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
        ]
        for row in benchmark:
            lines.append(
                f"| {label(variant(row))} | {row['component']} | {row['seq_length']} | "
                f"{row['batch_size']} | {row['dtype']} | {row['mode']} | "
                f"{number(row, 'mean_ms'):.2f} | {number(row, 'tokens_per_second'):.0f} | "
                f"{number(row, 'peak_cuda_memory_mib'):.1f} |"
            )
        failed = sum(row["status"] != "ok" for row in benchmark)
        if any(row["mode"] == "generation" for row in benchmark):
            lines += [
                "",
                "Generation: время генерации всей заданной последовательности новых токенов; "
                "throughput учитывает только новые токены. Длина L — фиксированное окно контекста. "
                "KV/recurrent cache не используется; время первого токена отдельно не измеряется.",
            ]
        if failed:
            lines += [
                "",
                f"Нет успешного замера в {failed} строках (например, OOM). Они не заменены нулями.",
            ]
    if histories:
        lines += ["", "История обучения:", ""]
        lines += [
            f"График: `train-points={train_points}`, `perplexity-scale={perplexity_scale}`.",
            "CSV сохраняет всю исходную историю, включая точки, не выбранные для графика.",
            "",
        ]
        for history in histories:
            cfg = history["config"]
            lines.append(
                f"- {label(history['key'])}, seed {history['seed']}: `{history['source']}`; "
                f"{cfg.model.decoder.d_model} / {cfg.model.decoder.num_layers} слоёв / "
                f"{cfg.model.decoder.num_heads} голов, контекст {cfg.max_seq_len}, "
                f"batch {cfg.dataloader.batch_size}, epoch_len {history['epoch_len']}, "
                f"validation до {cfg.trainer.get('evaluation_epoch_len', 'всех')} батчей."
            )
        if any(
            row["scope"] == "legacy_window"
            for history in histories
            for row in history["rows"]
        ):
            lines += [
                "",
                "Восстановленные train-точки из info.log — последнее окно логирования в каждой эпохе, "
                "а не среднее по эпохе. Более частая история прошлых шагов не сохранилась. "
                "Validation-точки усреднены по фактически проверенным батчам эпохи.",
            ]
        lines += [
            "",
            "training_time включает train и validation между началом цикла обучения и последней "
            "точкой истории; исключает подготовку данных. Для старых логов это время до печати "
            "последнего отчёта эпохи. CSV сохраняет scope и источник каждой точки.",
        ]
    if training_generation:
        with (output_dir / "training_generation_speed.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            speeds = list(csv.DictReader(file))
        lines += [
            "",
            "Среднее время обучения и скорость генерации:",
            "",
            "| Вариант | Обучающих батчей | Обучение, мс/батч | Новых токенов/с | Generation latency, мс |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for row in speeds:
            lines.append(
                f"| {label(variant(row))} | {row['training_batches']} | "
                f"{number(row, 'training_ms_per_batch'):.2f} | "
                f"{number(row, 'generation_tokens_per_second'):.2f} | "
                f"{number(row, 'generation_mean_ms'):.2f} |"
            )
        lines += [
            "",
            "Конфигурации train/generation сверены: архитектура, контекст, batch, "
            "FP16/FP32, dropout, chunk size и seed. Детали: `training_generation_config.json`.",
            "Среднее время обучения: суммарное время до train/epoch за вычетом записанных "
            "интервалов validation/test, делённое на число обработанных батчей. "
            "Включает optimizer, метрики, логирование и расходы между эпохами, в том числе "
            "сохранение checkpoints. Подготовка данных исключена.",
            "Generation измерен на случайных весах и синтетических входах, без кеша. "
            "Throughput суммирует новые токены всех последовательностей batch.",
        ]
    if flops:
        with (output_dir / "estimated_matmul_flops.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            estimates = list(csv.DictReader(file))
        lines += [
            "",
            "Аналитическая оценка матричных операций:",
            "",
            "| Вариант | Component | L | Batch | Attention core, GFLOPs | Forward, GFLOPs | Generation, GFLOPs |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in estimates:
            generation = number(row, "generation_gflops")
            generation_text = f"{generation:.2f}" if np.isfinite(generation) else "—"
            lines.append(
                f"| {row['attention_type']} | {row['component']} | {row['seq_length']} | "
                f"{row['batch_size']} | {number(row, 'attention_core_gflops'):.2f} | "
                f"{number(row, 'forward_gflops'):.2f} | {generation_text} |"
            )
        lines += [
            "",
            "Multiply-add = 2 FLOPs. Softmax: causal-треугольник; linear: все матричные "
            "умножения чанков с padding. Основной график использует causal-оценку softmax; "
            "плотный вариант сохранён отдельно в JSON/CSV. Это оценка матричных операций "
            "forward, не измерение фактических инструкций GPU и не стоимость backward.",
            "RoPE и поэлементные операции исключены, варианты с RoPE и без него объединены. "
            "Generation = число новых токенов на последовательность × forward всего batch "
            "при фиксированном контексте без кеша. Формулы: `estimated_matmul_flops.json`.",
        ]
    lines += [
        "",
        "Для этого набора данных сравнивайте одинаковые seed, длину контекста, batch и precision. "
        "Линейная сложность по длине не гарантирует меньшее время на коротком контексте. "
        "Один seed показывает результат одного запуска, а не разброс между обучениями.",
        "",
    ]
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", help="Quality JSON or CSV from evaluate_lm.py")
    parser.add_argument(
        "--benchmark", help="Speed JSON or CSV from benchmark_attention.py"
    )
    parser.add_argument(
        "--runs-dir", help="Parent directory containing saved model runs"
    )
    parser.add_argument("--output-dir", default="outputs/attention_plots")
    parser.add_argument(
        "--variant-style",
        choices=["variant", "attention"],
        default="variant",
        help="attention: one color per attention type; dashed/hatched RoPE variants",
    )
    parser.add_argument(
        "--training-generation",
        action="store_true",
        help="Compare mean training batch time and generation throughput; validate matching configurations",
    )
    parser.add_argument(
        "--flops",
        action="store_true",
        help="Estimate forward/generation matmul FLOPs from benchmark settings; no model execution",
    )
    parser.add_argument(
        "--flops-seq-lengths",
        nargs="+",
        type=int,
        help="Override context lengths only for analytical FLOPs; requires --flops",
    )
    parser.add_argument(
        "--flops-only",
        action="store_true",
        help="Export only analytical FLOPs plots/tables; requires --flops",
    )
    parser.add_argument(
        "--flops-scale", choices=["log", "linear", "both"], default="log",
        help="FLOPs context/value axes; both preserves the log plot and adds a _linear plot",
    )
    parser.add_argument("--x-axis", choices=["epoch", "step", "time"], default="step")
    parser.add_argument(
        "--train-points",
        choices=["window", "epoch", "last-window"],
        default="window",
        help="Train points: every logging window, full-epoch averages, or the last window per epoch",
    )
    parser.add_argument(
        "--perplexity-scale",
        choices=["linear", "log"],
        default="log",
        help="Perplexity scale for learning curves (default: log)",
    )
    args = parser.parse_args()
    if not any([args.quality, args.benchmark, args.runs_dir]):
        parser.error("Provide --quality, --benchmark or --runs-dir")
    if args.training_generation and not (args.benchmark and args.runs_dir):
        parser.error("--training-generation requires --benchmark and --runs-dir")
    if args.flops and not args.benchmark:
        parser.error("--flops requires --benchmark")
    if (args.flops_seq_lengths is not None or args.flops_only) and not args.flops:
        parser.error("--flops-seq-lengths and --flops-only require --flops")
    if args.flops_only and args.training_generation:
        parser.error("--flops-only cannot be combined with --training-generation")
    configure_variant_style(args.variant_style)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "figure.dpi": 160,
            "svg.fonttype": "none",
        }
    )
    output_dir = Path(args.output_dir)
    try:
        quality, quality_meta = load_report(args.quality) if args.quality else ([], {})
        benchmark, benchmark_meta = (
            load_report(args.benchmark) if args.benchmark else ([], {})
        )
        histories = load_histories(args.runs_dir) if args.runs_dir else []
        if not any([quality, benchmark, histories]):
            raise ValueError("No results found in the provided inputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.training_generation:
            plot_training_generation(histories, benchmark, benchmark_meta, output_dir)
        if args.flops:
            scales = ("log", "linear") if args.flops_scale == "both" else (args.flops_scale,)
            for scale in scales:
                plot_flops(benchmark, benchmark_meta, output_dir, args.flops_seq_lengths, scale)
        if quality and not args.flops_only:
            plot_quality(quality, quality_meta, output_dir)
        if benchmark and not args.flops_only:
            plot_benchmarks(benchmark, output_dir, benchmark_meta)
        if histories and not args.flops_only:
            plot_histories(
                histories,
                output_dir,
                args.x_axis,
                args.train_points,
                args.perplexity_scale,
            )
            export_histories(histories, output_dir)
        write_summary(
            quality if not args.flops_only else [],
            benchmark if not args.flops_only else [],
            histories if not args.flops_only else [],
            quality_meta,
            output_dir,
            [args.quality, args.benchmark, args.runs_dir],
            args.train_points,
            args.perplexity_scale,
            args.training_generation,
            args.flops,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(
        f"Saved plots (PNG/SVG), comparison.md and available training history to {output_dir}"
    )


if __name__ == "__main__":
    main()
