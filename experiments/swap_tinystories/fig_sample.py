"""Redraw figures/sample.jpg (TinyStories test-split quality) in the minimal dark style of fig_c2.py.

Values are transcribed from the labels printed on sample.jpg (2 decimals).
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import EXP, FIG

BG, FG = "#1c1c1c", "white"
MODELS = ["softmax\nwithout RoPE", "softmax\nwith RoPE", "linear\nwithout RoPE", "linear\nwith RoPE"]
PANELS = [
    ("NLL ↓", [1.52, 1.48, 1.60, 1.57], "{:.2f}"),
    ("Perplexity ↓", [4.56, 4.39, 4.95, 4.79], "{:.2f}"),
    ("Token accuracy, % ↑", [61.26, 61.98, 59.61, 60.23], "{:.2f}"),
]


def main(svg=False):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=BG)
    for ax, (label, vals, fmt) in zip(axes, PANELS):
        ax.set_facecolor(BG)
        bars = ax.bar(range(len(vals)), vals, width=0.6, color=FG)
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", color=FG, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.15)
        ax.set_ylabel(label, color=FG)
        ax.set_xticks(range(len(vals)), MODELS, fontsize=9)
        ax.tick_params(colors=FG)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(FG)
    fig.suptitle("TinyStories: quality on the test split", color=FG)
    fig.tight_layout()
    if svg:
        out = EXP / "figurs_svg"
        out.mkdir(exist_ok=True)
        fig.savefig(out / "sample.svg", transparent=True)
    else:
        fig.savefig(FIG / "sample_dark.png", dpi=150, facecolor=BG)


if __name__ == "__main__":
    main(svg="--svg" in sys.argv)
