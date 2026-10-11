"""Per-layer paired differences V2 - V1 with 95% bootstrap CIs (from summary_k16.json)."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

s = json.load(open(sys.argv[1]))
L = s["layers"]
d = s["per_layer_V2_minus_V1"]
keys = [("kl", "KL(model||lens): V2 − V1  (lower = better)"), ("top1", "top-1 agreement: V2 − V1"),
        ("next_top1", "top-1 = true next token: V2 − V1"), ("df_cos", "cos(f̂, Δf): V2 − V1"),
        ("df_rel", "‖f̂ − Δf‖/‖Δf‖: V2 − V1  (lower = better)"), ("cos_c", "cos(f̂, y − ȳ): V2 − V1")]
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, (k, t) in zip(axes.flat, keys):
    m = [x[0] for x in d[k]]; lo = [x[1] for x in d[k]]; hi = [x[2] for x in d[k]]
    ax.errorbar(L, m, yerr=[[a - b for a, b in zip(m, lo)], [b - a for a, b in zip(m, hi)]], fmt="o-", capsize=3)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(t, fontsize=10); ax.set_xlabel("layer")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=130)
print("saved", sys.argv[2])
