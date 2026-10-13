"""Minimal dark charts of dev swap success by single layer: J vs logit dictionaries, J-space vs random."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ART, EXP, FIG, jload
from report import single_layer_curves

BG, FG = "#1c1c1c", "white"
LENS_PAIR = [("J", "J-lens", "-o"), ("logit", "logit lens", "--s")]
CHARTS = {
    "C1": ("raw animal-token swap", [("C1",) + x for x in LENS_PAIR], "c1_j_vs_logit"),
    "C2": ("k16-component swap", [("C2",) + x for x in LENS_PAIR], "c2_j_vs_logit"),
    # C5 is the random vector with ‖Δh‖ matched to C2 of the same lens on each trial
    "random": ("J-space vs random vector", [("C2", "J", "J-space k16 component", "-o"),
                                            ("C5", "J", "random vector, same ‖Δh‖", "--s")], "j_vs_random"),
}


def main(chart="C2", svg=False):
    title, series, fname = CHARTS[chart]
    sw = jload(ART / "swap_results.json")
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=BG)
    ax.set_facecolor(BG)
    for cond, lens, name, style in series:
        layers, succ, _ = single_layer_curves(sw, cond, lens)
        ax.plot(layers, succ, style, color=FG, lw=1.5, ms=6, label=name)
    ax.set_xlabel("layer", color=FG)
    ax.set_ylabel("swap success", color=FG)
    ax.set_title(title, color=FG)
    ax.set_xticks(range(1, 7))
    ax.tick_params(colors=FG)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(FG)
    ax.legend(frameon=False, labelcolor=FG)
    fig.tight_layout()
    if svg:  # transparent background, same white marks
        out = EXP / "figurs_svg"
        out.mkdir(exist_ok=True)
        fig.savefig(out / f"{fname}.svg", transparent=True)
    else:
        fig.savefig(FIG / f"{fname}.png", dpi=150, facecolor=BG)


if __name__ == "__main__":
    import sys

    args = [a for a in sys.argv[1:] if a != "--svg"]
    for c in args or ["C2"]:
        main(c, svg="--svg" in sys.argv)
