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
import numpy as np
from omegaconf import OmegaConf


VARIANTS = [("softmax", False), ("softmax", True), ("linear", False), ("linear", True)]
COLORS = dict(zip(VARIANTS, ["#2563eb", "#0891b2", "#ea580c", "#9333ea"]))
LOG_VALUE = re.compile(r" - INFO -\s+(\w+)\s*:\s*([-+\d.eE]+)\s*$")


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


def bars(ax, rows, field, scale=1, errors=False):
    values = [number(row, field) * scale for row in rows]
    uncertainty = [number(row, "stdev_ms") for row in rows] if errors else None
    ax.bar(
        range(len(rows)),
        values,
        color=[COLORS[variant(row)] for row in rows],
        yerr=uncertainty,
        capsize=4 if errors else 0,
    )
    names = [label(variant(row)).replace(" ", "\n", 1) for row in rows]
    if len({variant(row) for row in rows}) < len(rows):
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


def plot_benchmarks(rows, output_dir):
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
            for ax, field, unit in zip(
                axes[:, column],
                ["mean_ms", "tokens_per_second"],
                [
                    "Latency (ms) ↓",
                    "Throughput (k tokens/s) ↑",
                ],
            ):
                scale = 0.001 if field == "tokens_per_second" else 1
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
                        ax.plot(x, y, marker="o", color=COLORS[key], label=label(key))
                    ax.set_xlabel("Sequence length (tokens)")
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
        fig.suptitle(
            f"{component.capitalize()}: batch {batch_size}, {dtype}, sequence length {lengths_text}\n"
            "Error bars: timing SD. Forward + backward excludes optimizer and data loading.",
            fontsize=13,
        )
        save_figure(
            fig,
            output_dir,
            f"benchmark_{component}_b{batch_size}_{dtype}_c{chunk_size}",
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
        ax.legend(fontsize=8)
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
    fig.suptitle(f"TinyStories: learning curves\n{note}", fontsize=13)
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
        benchmark, _ = load_report(args.benchmark) if args.benchmark else ([], {})
        histories = load_histories(args.runs_dir) if args.runs_dir else []
        if not any([quality, benchmark, histories]):
            raise ValueError("No results found in the provided inputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        if quality:
            plot_quality(quality, quality_meta, output_dir)
        if benchmark:
            plot_benchmarks(benchmark, output_dir)
        if histories:
            plot_histories(
                histories,
                output_dir,
                args.x_axis,
                args.train_points,
                args.perplexity_scale,
            )
            export_histories(histories, output_dir)
        write_summary(
            quality,
            benchmark,
            histories,
            quality_meta,
            output_dir,
            [args.quality, args.benchmark, args.runs_dir],
            args.train_points,
            args.perplexity_scale,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(
        f"Saved plots (PNG/SVG), comparison.md and available training history to {output_dir}"
    )


if __name__ == "__main__":
    main()
