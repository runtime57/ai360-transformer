from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

BUNDLE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else BUNDLE
OUT.mkdir(parents=True, exist_ok=True)


def smooth(rows):
    values = np.array([[row[3], row[4]] for row in rows])
    spacing = np.median(np.diff(values[:, 0]))
    window = max(3, int(round(0.05 / spacing)))
    kernel = np.ones(window) / window
    return np.convolve(values[:, 0], kernel, mode='valid'), np.convolve(values[:, 1], kernel, mode='valid'), window

def style_axes(ax, title, start=0, end=10):
    ax.set_title(title, loc='left', fontsize=20, fontweight='bold', pad=17)
    ax.set_xlabel('Epoch', labelpad=10)
    ax.set_ylabel('Loss', labelpad=10)
    ax.set_xticks(range(start, end + 1))
    ax.set_xlim(start - 0.04, end + 0.12)
    ax.grid(axis='x', color='#CED7E2', linewidth=0.9)
    ax.grid(axis='y', color='#E1E6ED', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#BDC7D4')
    ax.tick_params(length=0, pad=8)

def save_variants(fig, stem, train=False, inset=None):
    ax = fig.axes[0]
    axes = [ax] + ([inset] if inset is not None else [])
    for axis in axes:
        axis.set_yscale('log')
        ticks = [1, 1.5, 2, 3, 4, 6, 8] if train and axis is ax else ([1.2, 1.3, 1.4, 1.5] if axis is inset else [1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5])
        axis.yaxis.set_major_locator(FixedLocator(ticks))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
        axis.yaxis.set_minor_locator(NullLocator())
    ax.set_title(('Training loss' if train else 'Validation loss') + ' · log Y', loc='left', fontsize=20, fontweight='bold')
    for suffix in ('png', 'svg'):
        fig.savefig(OUT / f'{stem}_log_y.{suffix}', dpi=220)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.labelsize': 13,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FCFDFE',
    'text.color': '#243047',
    'axes.labelcolor': '#243047',
    'xtick.color': '#516078',
    'ytick.color': '#516078',
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})

RUNS = {'Baseline': {'color': '#3366CC', 'rows': [], 'metrics': {}}, 'RoPE': {'color': '#E47729', 'rows': [], 'metrics': {}}, 'Headwise gate': {'color': '#C94D62', 'rows': [], 'metrics': {}}, 'Elementwise gate': {'color': '#1B9AAA', 'rows': [], 'metrics': {}}, 'RoPE + headwise gate': {'color': '#2B9871', 'rows': [], 'metrics': {}}, 'RoPE + elementwise gate': {'color': '#9467BD', 'rows': [], 'metrics': {}}}
RUNS['ALiBi'] = {'color': '#7A5195', 'rows': [], 'metrics': {}}
GROUPS = {'1_baseline_rope': ['Baseline', 'RoPE'], '2_baseline_gates': ['Baseline', 'Headwise gate', 'Elementwise gate'], '3_rope_gates': ['RoPE', 'RoPE + headwise gate', 'RoPE + elementwise gate'], '6_gates_with_without_rope': ['Headwise gate', 'RoPE + headwise gate', 'Elementwise gate', 'RoPE + elementwise gate']}

GROUPS["7_baseline_rope_alibi"] = ["Baseline", "RoPE", "ALiBi"]

with (BUNDLE / "train_points.csv").open(newline="") as file:
    for row in csv.DictReader(file):
        RUNS[row["run"]]["rows"].append((int(row["epoch"]), int(row["batch_index"]), int(row["batches_per_epoch"]), float(row["epoch_fraction"]), float(row["train_loss"])))
with (BUNDLE / "epoch_metrics.csv").open(newline="") as file:
    for row in csv.DictReader(file):
        RUNS[row["run"]]["metrics"][int(row["epoch"])] = {"loss": float(row["logged_train_loss"]), "val_loss": float(row["val_loss"])}
for run in RUNS.values():
    run["smooth_x"], run["smooth_y"], _ = smooth(run["rows"])
all_val = [record["val_loss"] for run in RUNS.values() for record in run["metrics"].values()]
span = max(all_val) - min(all_val)
val_limits = (min(all_val) - 0.05 * span, max(all_val) + 0.05 * span)

requested = sys.argv[2:] or list(GROUPS)
for group in requested:
    labels = GROUPS[group]
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.subplots_adjust(left=0.085, right=0.96, top=0.89, bottom=0.19)
    style_axes(ax, "Training loss")
    for label in labels:
        run = RUNS[label]
        values = np.array([[row[3], row[4]] for row in run["rows"]])
        ax.plot(values[:, 0], values[:, 1], color=run["color"], alpha=0.18, linewidth=0.55, marker=".", markersize=1.4, rasterized=True)
        ax.plot(run["smooth_x"], run["smooth_y"], color=run["color"], linewidth=2.1, label=label)
    if "RoPE" in labels:
        rope = RUNS["RoPE"]
        sparse_epochs = np.arange(1, 8)
        ax.plot(sparse_epochs, [rope["metrics"][int(epoch)]["loss"] for epoch in sparse_epochs], color=rope["color"], linestyle=(0, (4, 3)), linewidth=1.8, marker="o", markersize=4.5)
    ax.set_ylim(1.0, 8.8)
    ax.legend(loc="upper right", frameon=False, fontsize=13)
    inset = ax.inset_axes([0.43, 0.38, 0.52, 0.37])
    for label in labels:
        run = RUNS[label]
        values = np.array([[row[3], row[4]] for row in run["rows"]])
        inset.plot(values[:, 0], values[:, 1], color=run["color"], alpha=0.14, linewidth=0.45, rasterized=True)
        inset.plot(run["smooth_x"], run["smooth_y"], color=run["color"], linewidth=1.65)
    inset.set_xlim(6.85, 10.04)
    inset.set_ylim(1.20, 1.58)
    inset.set_xticks([7, 8, 9, 10])
    inset.set_yticks([1.2, 1.3, 1.4, 1.5])
    inset.tick_params(labelsize=9, length=0, pad=5)
    inset.grid(color="#DEE4EC", linewidth=0.65)
    inset.set_title("Last epochs", fontsize=11, loc="left", pad=7)
    inset.spines[["top", "right"]].set_visible(False)
    inset.spines[["left", "bottom"]].set_color("#BDC7D4")
    fig.text(0.085, 0.075, "Тонкие линии — все train-точки; сплошные — среднее в окне ≈ 0.05 эпохи.", fontsize=10, color="#627087")
    note = "RoPE: подробный лог с эпохи 6.90; ранние точки — пунктир." if "RoPE" in labels else ("Все три рана: полный train-лог." if group == "2_baseline_gates" else "Все показанные раны: полный train-лог.")
    fig.text(0.085, 0.040, note, fontsize=10, color="#627087")
    assert ax.get_legend_handles_labels()[1] == labels
    save_variants(fig, group + "_train_loss", train=True, inset=inset)
    assert ax.get_xscale() == inset.get_xscale() == "linear"
    assert ax.get_yscale() == inset.get_yscale() == "log"
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.subplots_adjust(left=0.085, right=0.96, top=0.88, bottom=0.15)
    style_axes(ax, "Validation loss", start=1)
    for label in labels:
        run = RUNS[label]
        epochs = sorted(run["metrics"])
        ax.plot(epochs, [run["metrics"][epoch]["val_loss"] for epoch in epochs], color=run["color"], linewidth=2.25, marker="o", markersize=5.7, markeredgecolor="white", markeredgewidth=1.2, label=label)
    ax.set_ylim(*val_limits)
    ax.legend(loc="upper right", frameon=False, fontsize=13)
    assert ax.get_legend_handles_labels()[1] == labels
    save_variants(fig, group + "_val_loss")
    assert ax.get_xscale() == "linear" and ax.get_yscale() == "log"
    plt.close(fig)

print(f"Saved {2 * len(requested)} plots as PNG and SVG to " + str(OUT))
