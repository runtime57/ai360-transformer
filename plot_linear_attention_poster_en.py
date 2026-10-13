"""Build English poster figures and a point-by-point report from saved runs."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from plot_attention_results import (
    VARIANTS,
    load_histories,
    load_report,
    training_batch_times,
    validate_training_generation,
    variant,
)


NAMES = {
    ("softmax", False): "softmax",
    ("softmax", True): "softmax + RoPE",
    ("linear", False): "linear",
    ("linear", True): "linear + RoPE",
}
COLORS = {"softmax": "#4453a6", "linear": "#f06445"}
CONTEXT_COLORS = {512: "#aeb9e8", 8192: "#4054a5"}


def epoch_rows(history, split):
    rows = sorted(
        (r for r in history["rows"] if r["split"] == split and r["scope"] == "epoch"),
        key=lambda r: int(r["epoch"]),
    )
    if [int(r["epoch"]) for r in rows] != list(range(1, 11)):
        raise ValueError(f"Expected ten complete {split} epochs: {history['source']}")
    for row in rows:
        for field in ("loss", "perplexity", "token_accuracy"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Invalid {field}: {history['source']}")
        if row.get("nll") and not math.isclose(
            math.exp(float(row["nll"])), float(row["perplexity"]), rel_tol=1e-9
        ):
            raise ValueError(f"PPL does not equal exp(NLL): {history['source']}")
    return rows


def load_bundle(runs_dir, generation_path, quality_path, expected_context):
    histories = load_histories(runs_dir)
    if [h["key"] for h in histories] != VARIANTS:
        raise ValueError(f"Need exactly four attention variants: {runs_dir}")
    generation, metadata = load_report(generation_path)
    measured, configurations = validate_training_generation(histories, generation, metadata)
    timing = {variant(row): row for row in training_batch_times(histories)}
    for row in measured:
        key = variant(row)
        timing[key].update(
            generation_mean_ms=float(row["mean_ms"]),
            generation_tokens_per_second=float(row["tokens_per_second"]),
            generation_stdev_ms=float(row["stdev_ms"]),
        )
    cfg = configurations[0]
    if cfg["seq_length"] != expected_context:
        raise ValueError(f"Expected context {expected_context}: {runs_dir}")
    if len({row["training_batches"] for row in timing.values()}) != 1:
        raise ValueError(f"Different training budgets: {runs_dir}")
    quality, quality_metadata = load_report(quality_path)
    quality_by_key = {variant(row): row for row in quality}
    if set(quality_by_key) != set(VARIANTS) or len(quality) != 4:
        raise ValueError(f"Need test quality for all four variants: {quality_path}")
    if quality_metadata.get("split") != "test" or quality_metadata["seq_length"] != expected_context:
        raise ValueError(f"Wrong quality split or context: {quality_path}")
    if len({float(row["evaluated_tokens"]) for row in quality}) != 1:
        raise ValueError(f"Test variants evaluated different token counts: {quality_path}")
    for row in quality:
        checkpoint = Path(row["checkpoint"])
        if checkpoint.parent.parent.resolve() != runs_dir.resolve():
            raise ValueError(f"Quality belongs to another run: {checkpoint}")
        if not math.isclose(math.exp(float(row["nll"])), float(row["perplexity"]), rel_tol=1e-9):
            raise ValueError(f"Test PPL does not equal exp(NLL): {checkpoint}")
    return {
        "runs": runs_dir,
        "generation_source": generation_path,
        "quality_source": quality_path,
        "histories": {h["key"]: h for h in histories},
        "val": {h["key"]: epoch_rows(h, "val") for h in histories},
        "train": {h["key"]: epoch_rows(h, "train") for h in histories},
        "speed": timing,
        "config": cfg,
        "environment": metadata.get("environment", {}),
        "quality": quality_by_key,
        "quality_metadata": quality_metadata,
    }


def speedup(bundle, rope, field):
    soft = bundle["speed"][("softmax", rope)][field]
    linear = bundle["speed"][("linear", rope)][field]
    return linear / soft if field == "generation_tokens_per_second" else soft / linear


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dfe3eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#384151")


def save_figure(fig, output_dir, name):
    for extension in ("svg", "png"):
        fig.savefig(output_dir / f"{name}.{extension}", dpi=240, facecolor="white")
    plt.close(fig)


def comparison_captions(bundles):
    short, long = bundles
    long_training = [100 * (1 - 1 / speedup(long, rope, "training_ms_per_batch")) for rope in (False, True)]
    long_generation = [100 * (speedup(long, rope, "generation_tokens_per_second") - 1) for rope in (False, True)]
    short_training = [100 * (1 / speedup(short, rope, "training_ms_per_batch") - 1) for rope in (False, True)]
    short_generation = [100 * (1 - speedup(short, rope, "generation_tokens_per_second")) for rope in (False, True)]

    def span(values):
        return f"{min(values):.0f}–{max(values):.0f}%"

    return [
        (
            "Context length 512: linear attention is slower",
            f"Batch time is about {np.mean(short_training):.0f}% higher; "
            f"generation throughput is {span(short_generation)} lower.\n"
            "At this short context, quadratic softmax computation is still cheap.\n"
            "Softmax is better optimized.",
        ),
        (
            "Context length 8192: linear attention is faster",
            f"Batch time is {span(long_training)} lower; "
            f"generation throughput is {span(long_generation)} higher.\n"
            "Core complexity: softmax O(T²d), linear O(Td²). For fixed d,\n"
            "linear scales linearly and softmax quadratically with context T.",
        ),
    ]


def plot_comparison(bundles, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 6.25), gridspec_kw={"width_ratios": [1.55, 1, 1]})
    fig.subplots_adjust(left=0.046, right=0.974, bottom=0.37, top=0.80, wspace=0.37)
    fig.suptitle("Linear Attention", x=0.046, y=0.967, ha="left", color=COLORS["softmax"], fontsize=25, weight="bold")
    ax = axes[0]
    for key in VARIANTS:
        rows = bundles[0]["val"][key]
        ax.plot(
            [int(r["epoch"]) for r in rows], [float(r["loss"]) for r in rows],
            color=COLORS[key[0]], linestyle="--" if key[1] else "-", linewidth=2.4, label=NAMES[key],
        )
    ax.set_title("Validation loss · context 512", loc="left", fontsize=14, pad=15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss ↓")
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(1.4, 2.7)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    style_axis(ax)
    for ax, field, title in zip(
        axes[1:], ("training_ms_per_batch", "generation_tokens_per_second"),
        ("Training speed", "Generation speed"),
    ):
        x = np.arange(2)
        for index, bundle in enumerate(bundles):
            context = bundle["config"]["seq_length"]
            values = [speedup(bundle, rope, field) for rope in (False, True)]
            bars = ax.bar(x + (index - 0.5) * 0.33, values, width=0.30, color=CONTEXT_COLORS[context], zorder=3)
            ax.bar_label(bars, labels=[f"{v:.2f}×" for v in values], padding=5, fontsize=12, weight="bold")
        ax.axhline(1, color="#707887", linewidth=1.2, linestyle="--", zorder=2)
        ax.set_title(title, loc="left", fontsize=14, pad=15)
        ax.set_xticks(x, ["Without RoPE", "+ RoPE"])
        ax.set_xlabel("Linear vs. matching softmax", fontsize=9.5, labelpad=10)
        ax.set_ylabel("Linear / softmax speed, × ↑")
        ax.set_ylim(0, 1.55)
        ax.set_yticks([0, 0.5, 1, 1.5])
        style_axis(ax)
    fig.legend(
        handles=[Patch(facecolor=CONTEXT_COLORS[t], label=f"Context {t}") for t in (512, 8192)],
        loc="upper right", bbox_to_anchor=(0.972, 0.972), ncol=2, frameon=False, fontsize=12,
    )
    fig.add_artist(plt.Line2D([0.046, 0.974], [0.22, 0.22], transform=fig.transFigure, color="#dfe3eb", linewidth=1))
    for x, (title, body) in zip((0.046, 0.523), comparison_captions(bundles)):
        fig.text(x, 0.182, title, ha="left", va="top", fontsize=13.5, weight="bold", color=COLORS["softmax"])
        fig.text(x, 0.130, body, ha="left", va="top", fontsize=11, color="#384151", linespacing=1.55)

    save_figure(fig, output_dir, "linear_attention_comparison")


def plot_absolute(bundles, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 7.0))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.82, hspace=0.75, wspace=0.30)
    fig.suptitle("Linear attention and RoPE: absolute values", x=0.055, y=0.97, ha="left", color=COLORS["softmax"], fontsize=21, weight="bold")
    fields = ("training_ms_per_batch", "generation_mean_ms", "generation_tokens_per_second")
    titles = ("Training, ms/batch ↓", "Generating 64 tokens, ms ↓", "New tokens/s ↑")
    for row_index, bundle in enumerate(bundles):
        for ax, field, title in zip(axes[row_index], fields, titles):
            values = [bundle["speed"][key][field] for key in VARIANTS]
            bars = ax.bar(range(4), values, color=[COLORS[k[0]] for k in VARIANTS], edgecolor="white", linewidth=1)
            for bar, key in zip(bars, VARIANTS):
                bar.set_hatch("///" if key[1] else "")
            ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=4, fontsize=10)
            ax.set_xticks(range(4), ["softmax", "softmax\n+ RoPE", "linear", "linear\n+ RoPE"], fontsize=9)
            ax.set_title(title, loc="left", fontsize=12, pad=10)
            ax.set_ylim(0, max(values) * 1.26)
            style_axis(ax)
        axes[row_index, 0].set_ylabel(f"Context {bundle['config']['seq_length']}", color=COLORS["softmax"], weight="bold", fontsize=13, labelpad=13)
    save_figure(fig, output_dir, "linear_attention_speed_absolute")


def f(value, digits=6):
    return f"{float(value):.{digits}f}"


def table(lines, headings, rows):
    lines.append(" | ".join(headings))
    lines.append(" | ".join("-" * len(str(h)) for h in headings))
    lines.extend(" | ".join(map(str, row)) for row in rows)
    lines.append("")


def export_points(bundles, output_dir):
    points = []
    for key in VARIANTS:
        for row in bundles[0]["val"][key]:
            points.append(dict(figure="linear_attention_comparison", panel="validation_loss", run=bundles[0]["runs"].name, context=512, variant=NAMES[key], epoch=int(row["epoch"]), step=int(row["step"]), value=float(row["loss"]), unit="loss"))
    for bundle in bundles:
        for field, panel in (("training_ms_per_batch", "training_speedup"), ("generation_tokens_per_second", "generation_speedup")):
            for rope in (False, True):
                points.append(dict(figure="linear_attention_comparison", panel=panel, run=bundle["runs"].name, context=bundle["config"]["seq_length"], variant="linear / softmax" + (" + RoPE" if rope else ""), epoch="", step="", value=speedup(bundle, rope, field), unit="ratio"))
        for field, unit in (("training_ms_per_batch", "ms/batch"), ("generation_mean_ms", "ms"), ("generation_tokens_per_second", "tokens/s")):
            for key in VARIANTS:
                points.append(dict(figure="linear_attention_speed_absolute", panel=field, run=bundle["runs"].name, context=bundle["config"]["seq_length"], variant=NAMES[key], epoch="", step="", value=bundle["speed"][key][field], unit=unit))
    with (output_dir / "linear_attention_comparison_points.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    return points


def write_report(bundles, points, output_dir):
    short, long = bundles
    last = {key: short["val"][key][-1] for key in VARIANTS}
    lines = ["LINEAR ATTENTION AND RoPE", "", "1. POSTER CAPTIONS", ""]
    lines += ["Captions at the bottom of the main Linear Attention figure:", ""]
    for title, body in comparison_captions(bundles):
        lines += [title, body, ""]
    lines.append(
        f"RoPE reduces validation loss for both variants (context 512, epoch 10): "
        f"softmax {f(last[('softmax', False)]['loss'], 3)} → {f(last[('softmax', True)]['loss'], 3)}; "
        f"linear {f(last[('linear', False)]['loss'], 3)} → {f(last[('linear', True)]['loss'], 3)}."
    )
    for bundle in bundles:
        context = bundle["config"]["seq_length"]
        for rope in (False, True):
            train = speedup(bundle, rope, "training_ms_per_batch")
            generation = speedup(bundle, rope, "generation_tokens_per_second")
            tag = "with RoPE" if rope else "without RoPE"
            train_change = 100 * (1 - 1 / train)
            throughput_change = 100 * (generation - 1)
            lines.append(
                f"Context {context}, {tag}: linear / softmax training speed = {train:.3f}×, "
                f"generation speed = {generation:.3f}×. "
                f"Training time is {abs(train_change):.1f}% {'lower' if train_change >= 0 else 'higher'}; "
                f"generation throughput is {abs(throughput_change):.1f}% {'higher' if throughput_change >= 0 else 'lower'}."
            )
    lines += [
        "", "In the comparison plot, 1× means equal speed for linear and softmax. "
        "Above 1×, linear is faster; below 1×, linear is slower. "
        "Each variant is compared with softmax at the same context length and with the same RoPE setting.",
        "Training speed = softmax time / linear time. "
        "Generation speed = linear throughput / softmax throughput = softmax latency / linear latency.",
        "", "2. CONCLUSIONS", "",
        "In attention_large_50k (context 512), RoPE improves quality for both softmax "
        "and linear attention. Softmax retains a quality advantage: test PPL "
        f"{f(short['quality'][('softmax', False)]['perplexity'], 2)} vs. "
        f"{f(short['quality'][('linear', False)]['perplexity'], 2)} without RoPE and "
        f"{f(short['quality'][('softmax', True)]['perplexity'], 2)} vs. "
        f"{f(short['quality'][('linear', True)]['perplexity'], 2)} with RoPE.",
        "In attention_large_context_1k (context 8192), linear attention is faster: "
        f"training time per batch is {100 * (1 - 1 / speedup(long, False, 'training_ms_per_batch')):.1f}% lower "
        f"without RoPE and {100 * (1 - 1 / speedup(long, True, 'training_ms_per_batch')):.1f}% lower with RoPE; "
        f"generation produces {100 * (speedup(long, False, 'generation_tokens_per_second') - 1):.1f}% "
        f"and {100 * (speedup(long, True, 'generation_tokens_per_second') - 1):.1f}% more new tokens/s, respectively. "
        "At context 512, our linear implementation is slower than softmax. "
        "Thus, linear complexity in sequence length does not guarantee a speedup at short contexts.",
        "On the test set for attention_large_context_1k, RoPE also reduces PPL: "
        f"softmax {f(long['quality'][('softmax', False)]['perplexity'], 2)} → "
        f"{f(long['quality'][('softmax', True)]['perplexity'], 2)}, "
        f"linear {f(long['quality'][('linear', False)]['perplexity'], 2)} → "
        f"{f(long['quality'][('linear', True)]['perplexity'], 2)}. "
        "Without RoPE, the two attention variants have similar quality; with RoPE, softmax achieves lower PPL. "
        "These results come from a single run with a small training budget (1000 batches).",
        "These two runs do not isolate the effect of increasing context length: "
        "model width, batch size, and training budget also change. "
        "The normalized bars show relative gains within each configuration; "
        "they do not eliminate these differences. Comparing PPL across the two configurations likewise does not isolate the effect of context length.",
        "", "3. CONFIGURATIONS AND METHODOLOGY", "",
    ]
    for bundle in bundles:
        cfg = bundle["config"]
        budget = bundle["speed"][VARIANTS[0]]["training_batches"]
        lines.append(f"{bundle['runs'].name}: context {cfg['seq_length']}, d_model={cfg['d_model']}, "
                     f"layers={cfg['num_layers']}, heads={cfg['num_heads']}, batch={cfg['batch_size']}, "
                     f"AMP={cfg['dtype']}, chunk={cfg['linear_chunk_size']}, seed={cfg['seed']}, "
                     f"vocab={cfg['vocab_size']}; {budget} training batches, 10 epochs.")
    lines += [
        f"GPU: {short['environment'].get('device_name', 'not specified')}; "
        f"PyTorch: {short['environment'].get('torch_version', 'not specified')}.",
        "The linear attention core is computed in float32 during training/benchmarking with float16 AMP.",
        "Average training time: elapsed time through train/epoch minus recorded validation/test intervals, "
        "divided by the number of processed batches. Includes the optimizer, metrics, logging, and overhead between epochs; "
        "data preparation is excluded. This is the average time per processed batch, including batches with AMP-skipped updates.",
        "Generation: a separate benchmark with random weights and synthetic inputs; "
        "64 new tokens per sequence, a fixed context window, and no KV/recurrent cache. "
        "Latency is the time to generate all 64 tokens for the entire batch; throughput is new tokens per second for the entire batch. "
        "This measures computational speed, not the quality of text generated by trained models.",
        "Validation: up to 50 batches per epoch; loss and metrics come from val/epoch rows. "
        "Train: full train/epoch averages, rather than the final logging window.",
        "The reference training_curves_512 plots previously used the final training window of each epoch; "
        "train/epoch values in this report may differ from those training curves. Validation loss matches.",
        "PPL = exp(NLL); NLL is aggregated with weights based on the number of non-padding target tokens. "
        "Loss and NLL may differ slightly due to numerical precision.",
        "One seed; variability across independent training runs was not evaluated.",
        "", "4. ALL POINTS IN THE MAIN PLOT", "", "4.1 Validation loss: attention_large_50k", "",
    ]
    table(lines, ["Epoch", "Batches", *NAMES.values()], [
        [epoch, short["val"][VARIANTS[0]][epoch - 1]["step"],
         *(f(short["val"][key][epoch - 1]["loss"], 12) for key in VARIANTS)]
        for epoch in range(1, 11)
    ])
    lines += ["4.2 Relative speed: all bars in the comparison plot", ""]
    table(lines, ["Context", "RoPE", "Training, ×", "Generation, ×"], [
        [bundle["config"]["seq_length"], "yes" if rope else "no",
         f(speedup(bundle, rope, "training_ms_per_batch"), 12),
         f(speedup(bundle, rope, "generation_tokens_per_second"), 12)]
        for bundle in bundles for rope in (False, True)
    ])
    lines += ["5. ABSOLUTE SPEED VALUES: ALL BARS IN THE SUPPLEMENTARY PLOT", ""]
    table(lines, ["Context", "Variant", "Batches", "Training, s", "Training, ms/batch", "Generation, ms", "New tokens/s", "Latency std, ms"], [
        [bundle["config"]["seq_length"], NAMES[key], row["training_batches"],
         *(f(row[field], 9) for field in ("training_seconds", "training_ms_per_batch", "generation_mean_ms", "generation_tokens_per_second", "generation_stdev_ms"))]
        for bundle in bundles for key, row in bundle["speed"].items()
    ])
    lines += ["Latency std is the standard deviation across synthetic benchmark repetitions, not across training runs.", "", "6. PPL AND OTHER METRICS BY EPOCH", ""]
    for bundle in bundles:
        lines += [f"6. Context {bundle['config']['seq_length']}: {bundle['runs'].name}", ""]
        table(lines, ["Epoch", "Batches", "Variant", "Train loss", "Train PPL", "Train accuracy, %", "Val loss", "Val NLL", "Val PPL", "Val accuracy, %"], [
            [epoch, val["step"], NAMES[key], f(train["loss"], 9), f(train["perplexity"], 9),
             f(100 * float(train["token_accuracy"]), 6), f(val["loss"], 9), f(val["nll"], 9),
             f(val["perplexity"], 9), f(100 * float(val["token_accuracy"]), 6)]
            for epoch in range(1, 11) for key in VARIANTS
            for train, val in [(bundle["train"][key][epoch - 1], bundle["val"][key][epoch - 1])]
        ])
    lines += ["7. BEST VALIDATION LOSS AND CORRESPONDING PPL", ""]
    table(lines, ["Context", "Variant", "Epoch", "Val loss", "Val PPL"], [
        [bundle["config"]["seq_length"], NAMES[key], best["epoch"], f(best["loss"], 9), f(best["perplexity"], 9)]
        for bundle in bundles for key in VARIANTS
        for best in [min(bundle["val"][key], key=lambda row: float(row["loss"]))]
    ])
    lines += ["8. TEST: NLL, PPL AND ACCURACY FOR MODEL_BEST", ""]
    for bundle in bundles:
        meta = bundle["quality_metadata"]
        lines += [f"{bundle['runs'].name}: {bundle['quality_source'].as_posix()}; "
                  f"split={meta['split']}, dtype={meta['dtype']}, "
                  f"limit_stories={meta['dataset_config'].get('limit_stories')}, "
                  f"max_batches={meta.get('max_batches')}.", ""]
        table(lines, ["Variant", "Checkpoint epoch", "Test NLL", "Test PPL", "Test accuracy, %", "Target tokens", "Evaluation batches"], [
            [NAMES[key], row["epoch"], f(row["nll"], 12), f(row["perplexity"], 12),
             f(100 * float(row["token_accuracy"]), 9), int(row["evaluated_tokens"]), row["evaluated_batches"]]
            for key, row in bundle["quality"].items()
        ])
    lines += [
        "Test for context 512 uses the entire saved test split; test for context 8192 uses the first 1000 stories. "
        "For each context, all four variants are evaluated on the same target tokens. "
        "Test set sizes differ between contexts.",
        "Test for attention_large_context_1k was additionally evaluated for all four model_best checkpoints "
        "in float32, batch=4, limit_stories=1000, with no limit on the number of batches.",
        "", "9. SOURCES AND REPRODUCTION", "",
    ]
    for bundle in bundles:
        lines.append(f"Runs: {bundle['runs'].as_posix()}")
        lines.extend(f"  {bundle['histories'][key]['source']}" for key in VARIANTS)
        lines.append(f"Generation benchmark: {bundle['generation_source'].as_posix()}")
        lines.append(f"Test quality: {bundle['quality_source'].as_posix()}")
    lines += [
        "", "Rebuild: .venv\\Scripts\\python.exe plot_linear_attention_poster_en.py",
        f"Numerical values of all {len(points)} points/bars in both figures at full machine precision: "
        "linear_attention_comparison_points.csv. Values in this text file are rounded to 6–12 decimal places.",
        "Main figure: linear_attention_comparison.svg / .png.",
        "Supplementary figure with absolute values: linear_attention_speed_absolute.svg / .png.",
    ]
    (output_dir / "linear_attention_comparison_data.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-512", type=Path, default=Path("saved/attention_large_50k"))
    parser.add_argument("--runs-8192", type=Path, default=Path("saved/attention_large_context_1k"))
    parser.add_argument("--generation-512", type=Path, default=Path("outputs/attention_inference_50k.json"))
    parser.add_argument("--generation-8192", type=Path, default=Path("outputs/attention_inference_large_context.json"))
    parser.add_argument("--quality-512", type=Path, default=Path("outputs/attention_quality_large_50k.json"))
    parser.add_argument("--quality-8192", type=Path, default=Path("outputs/attention_quality_large_context_1k_poster.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/en"))
    args = parser.parse_args()
    bundles = [
        load_bundle(args.runs_512, args.generation_512, args.quality_512, 512),
        load_bundle(args.runs_8192, args.generation_8192, args.quality_8192, 8192),
    ]
    if bundles[0]["quality_metadata"]["tokenizer_sha256"] != bundles[1]["quality_metadata"]["tokenizer_sha256"]:
        raise ValueError("Test evaluations use different tokenizers")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(bundles, args.output_dir)
    plot_absolute(bundles, args.output_dir)
    points = export_points(bundles, args.output_dir)
    write_report(bundles, points, args.output_dir)
    print(f"Saved two SVG/PNG figures, {len(points)} plot points, and the English report to {args.output_dir}")


if __name__ == "__main__":
    main()
