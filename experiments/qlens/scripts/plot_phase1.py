"""Phase 1 plots: convergence of ||Hbar_N[delta,delta]|| vs the sign-flip null, per layer."""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

path, out = sys.argv[1], sys.argv[2]
r = json.load(open(path))
layers = r["per_layer"]
fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 3.4), squeeze=False)
for ax, (l, v) in zip(axes[0], layers.items()):
    Ns = sorted(int(n) for n in v["curve"])
    norm = [v["curve"][str(n)]["norm"] for n in Ns]
    null = [v["curve"][str(n)]["null_mean"] for n in Ns]
    q95 = [v["curve"][str(n)]["null_q95"] for n in Ns]
    ax.loglog(Ns, norm, "o-", label=r"$\|\bar H_N[\delta,\delta]\|$")
    ax.loglog(Ns, null, "s--", color="gray", label="sign-flip null (mean)")
    ax.fill_between(Ns, null, q95, color="gray", alpha=0.2, label="null 95%")
    ax.loglog(Ns, [null[0] * (Ns[0] / n) ** 0.5 for n in Ns], ":", color="k", label=r"$1/\sqrt{N}$")
    ax.set_title(f"layer {l}  (rel size {v['curve'][str(Ns[-1])]['rel_size_mean']:.2f})")
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.minorticks_off()
    ax.set_xlabel("contexts N (x S=2 positions)")
axes[0][0].set_ylabel("mean over 100 directions")
axes[0][0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(out, dpi=130)
print("saved", out)
