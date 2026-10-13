"""Redraw figures/sample2.jpg (TinyStories learning curves) in the minimal dark style of fig_c2.py.

The source run logs are not on this machine, so loss and accuracy are read off sample2.jpg by eye
(≈ ±0.01 loss, ≈ ±0.2 pp accuracy). Perplexity is exp(loss), which matches the original panels.
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import EXP, FIG

BG, FG = "#1c1c1c", "white"
EPOCHS = np.arange(1, 11)
# hue = attention type (validated pair: ΔE_OKLab ≥ 26.8 under protan/deutan/tritan, ≥4.4:1 on BG),
# line style + marker = RoPE
SOFTMAX, LINEAR = "#3987e5", "#d95926"
SERIES = [  # name, color, line style, marker, filled marker
    ("softmax without RoPE", SOFTMAX, "--", "s", False),
    ("softmax with RoPE", SOFTMAX, "-", "o", True),
    ("linear without RoPE", LINEAR, "--", "s", False),
    ("linear with RoPE", LINEAR, "-", "o", True),
]
TRAIN_LOSS = [
    [2.43, 2.06, 1.90, 1.82, 1.78, 1.745, 1.72, 1.65, 1.67, 1.66],
    [2.26, 1.96, 1.83, 1.765, 1.72, 1.69, 1.67, 1.60, 1.62, 1.61],
    [2.60, 2.15, 1.98, 1.90, 1.85, 1.815, 1.79, 1.72, 1.745, 1.725],
    [2.46, 2.08, 1.93, 1.86, 1.81, 1.78, 1.75, 1.68, 1.71, 1.69],
]
TRAIN_ACC = [
    [46.1, 51.9, 54.2, 55.5, 56.4, 57.1, 57.6, 58.8, 58.45, 58.75],
    [49.0, 53.5, 55.75, 56.7, 57.6, 58.1, 58.5, 59.8, 59.55, 59.7],
    [43.3, 50.3, 52.75, 54.3, 55.0, 55.8, 56.4, 57.3, 57.05, 57.3],
    [45.7, 51.5, 53.85, 55.05, 56.1, 56.4, 57.0, 58.0, 57.75, 58.1],
]
VAL_LOSS = [
    [2.35, 1.93, 1.80, 1.72, 1.655, 1.605, 1.575, 1.55, 1.54, 1.535],
    [2.20, 1.86, 1.745, 1.67, 1.61, 1.565, 1.535, 1.51, 1.50, 1.495],
    [2.55, 2.05, 1.905, 1.815, 1.745, 1.695, 1.655, 1.63, 1.615, 1.61],
    [2.43, 2.00, 1.855, 1.77, 1.705, 1.66, 1.63, 1.60, 1.59, 1.585],
]
VAL_ACC = [
    [47.6, 54.0, 56.3, 57.75, 58.9, 59.7, 60.35, 60.85, 61.0, 61.1],
    [50.1, 55.5, 57.4, 58.7, 59.75, 60.4, 61.05, 61.45, 61.75, 61.85],
    [44.1, 51.75, 54.3, 55.9, 57.1, 58.0, 58.7, 59.25, 59.45, 59.6],
    [46.0, 52.7, 55.05, 56.45, 57.75, 58.65, 59.1, 59.75, 60.0, 60.1],
]
PANELS = [
    ("Train loss ↓", TRAIN_LOSS), ("Train perplexity ↓", np.exp(TRAIN_LOSS)), ("Train accuracy, % ↑", TRAIN_ACC),
    ("Validation loss ↓", VAL_LOSS), ("Validation perplexity ↓", np.exp(VAL_LOSS)), ("Validation accuracy, % ↑", VAL_ACC),
]


def main(svg=False):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), facecolor=BG)
    for ax, (title, data) in zip(axes.flat, PANELS):
        ax.set_facecolor(BG)
        for (name, color, ls, mk, filled), ys in zip(SERIES, data):
            ax.plot(EPOCHS, ys, ls, marker=mk, color=color, lw=1.8, ms=5, mfc=color if filled else "none", mew=1.5, label=name)
        ax.set_title(title, color=FG)
        ax.set_xlabel("Epoch", color=FG)
        ax.set_xticks(range(2, 11, 2))
        ax.tick_params(colors=FG)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(FG)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, labelcolor=FG)
    fig.suptitle("TinyStories: learning curves\nTrain: final logging window of each epoch; validation: evaluated-batch averages.",
                 color=FG)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if svg:
        out = EXP / "figurs_svg"
        out.mkdir(exist_ok=True)
        fig.savefig(out / "sample2.svg", transparent=True)
    else:
        fig.savefig(FIG / "sample2_dark.png", dpi=150, facecolor=BG)


if __name__ == "__main__":
    main(svg="--svg" in sys.argv)
