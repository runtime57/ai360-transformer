"""Phase 2 tables + plots: V0/V1/V2/C1/C2 x (accuracy, SHL, ICR raw / normalised) with 95%
bootstrap CIs (1000 resamples over prompts), paired differences vs V1, per-layer plots."""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--n_boot", type=int, default=1000)
args = ap.parse_args()
raw = torch.load(os.path.join(args.dir, f"eval_raw_k{args.k}.pt"), weights_only=False)
dfr = torch.load(os.path.join(args.dir, f"deltaf_raw_k{args.k}.pt"), weights_only=False)
meta = json.load(open(os.path.join(args.dir, f"meta_k{args.k}.json")))
held, le = raw["heldout"], raw["lenseval"]
layers = held["_layers"]
V = ["V0", "V1", "V2", "C1", "C2"]
SETS = [s for s in le if not s.startswith("_")]
rng = np.random.default_rng(0)


def boot(x):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (args.n_boot, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def fmt(t, p=3):
    return f"{t[0]:.{p}f} [{t[1]:.{p}f}, {t[2]:.{p}f}]"


# per-prompt scalars ------------------------------------------------------------
per_prompt = {v: {} for v in V}
for v in V:
    h = held[v]
    for key in ("kl", "top1", "cos_c", "rel_c"):
        per_prompt[v][key] = np.mean(h[key], axis=1)                 # mean over layers
    per_prompt[v]["SHL_with"] = np.asarray(h["shl_with"])
    per_prompt[v]["SHL_without"] = np.asarray(h["shl_without"])
    for s in SETS:
        per_prompt[v][f"ICR_raw/{s}"] = np.array([r["icr_raw"] for r in le[s][v]], float)
        per_prompt[v][f"ICR_norm/{s}"] = np.array([r["norm_any"] for r in le[s][v]], float)
        per_prompt[v][f"ICR_ro/{s}"] = np.array([r["norm_ro"] for r in le[s][v]], float)
# Delta-f (per held-out sample, pooled over layers -> mean per layer list)
dv = [v for v in V if v in dfr]
for v in dv:
    per_prompt[v]["df_cos"] = np.mean([dfr[v][l]["cos"] for l in layers], axis=0)
    per_prompt[v]["df_rel"] = np.mean([dfr[v][l]["rel"] for l in layers], axis=0)

summary = {"meta": meta, "metrics": {}, "diff_vs_V1": {}, "diff_vs_V0": {}}
keys = list(per_prompt["V1"].keys())
for key in keys:
    summary["metrics"][key] = {v: boot(per_prompt[v][key]) for v in V if key in per_prompt[v]}
    for ref, name in (("V1", "diff_vs_V1"), ("V0", "diff_vs_V0")):
        summary[name][key] = {v: boot(per_prompt[v][key] - per_prompt[ref][key])
                              for v in V if v != ref and key in per_prompt[v] and key in per_prompt[ref]}
# set averages (mean over the six sets of per-set means; CI by resampling within each set)
for kind in ("ICR_raw", "ICR_norm", "ICR_ro"):
    for tgt, ref in (("metrics", None), ("diff_vs_V1", "V1"), ("diff_vs_V0", "V0")):
        row = {}
        for v in V:
            if ref == v:
                continue
            bs = []
            for s in SETS:
                x = per_prompt[v][f"{kind}/{s}"] - (per_prompt[ref][f"{kind}/{s}"] if ref else 0)
                idx = rng.integers(0, len(x), (args.n_boot, len(x)))
                bs.append(x[idx].mean(1))
            bs = np.mean(bs, axis=0)
            mean = np.mean([(per_prompt[v][f"{kind}/{s}"] - (per_prompt[ref][f"{kind}/{s}"] if ref else 0)).mean() for s in SETS])
            row[v] = (float(mean), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))
        summary[tgt][f"{kind}/avg"] = row

# per-layer curves -------------------------------------------------------------
per_layer = {v: {key: np.mean(held[v][key], axis=0).tolist() for key in ("kl", "top1", "cos_c", "next_top1")} for v in V}
for v in dv:
    per_layer[v]["df_cos"] = [float(np.mean(dfr[v][l]["cos"])) for l in layers]
    per_layer[v]["df_rel"] = [float(np.median(dfr[v][l]["rel"])) for l in layers]
for v in V:
    per_layer[v]["icr_cells"] = [float(np.mean([np.mean([r["icr_per_layer"][i] for r in le[s][v]]) for s in SETS]))
                                 for i in range(len(layers))]
per_layer["quad_vs_residual_cos"] = [float(np.mean(dfr["quad_vs_residual"][l])) for l in layers]
summary["per_layer"] = per_layer
summary["layers"] = layers
# paired per-layer V2 - V1 CIs (held-out prompts)
summary["per_layer_V2_minus_V1"] = {}
for key in ("kl", "top1", "cos_c", "next_top1"):
    a, b = np.asarray(held["V2"][key]), np.asarray(held["V1"][key])
    summary["per_layer_V2_minus_V1"][key] = [boot(a[:, i] - b[:, i]) for i in range(len(layers))]
summary["per_layer_V2_minus_V1"]["df_rel"] = [boot(np.asarray(dfr["V2"][l]["rel"]) - np.asarray(dfr["V1"][l]["rel"])) for l in layers]
summary["per_layer_V2_minus_V1"]["df_cos"] = [boot(np.asarray(dfr["V2"][l]["cos"]) - np.asarray(dfr["V1"][l]["cos"])) for l in layers]

with open(os.path.join(args.dir, f"summary_k{args.k}.json"), "w") as f:
    json.dump(summary, f, indent=1)

# markdown tables
lines = ["| metric | " + " | ".join(V) + " |", "|---|" + "---|" * len(V)]
show = ["kl", "top1", "cos_c", "df_cos", "df_rel", "SHL_with", "SHL_without"] + \
       [f"ICR_raw/{s}" for s in SETS] + ["ICR_raw/avg"] + [f"ICR_norm/{s}" for s in SETS] + ["ICR_norm/avg"] + \
       [f"ICR_ro/{s}" for s in SETS] + ["ICR_ro/avg"]
for key in show:
    row = summary["metrics"].get(key, {})
    p = 1 if key.startswith("ICR_raw") else 3
    lines.append(f"| {key} | " + " | ".join(fmt(row[v], p) if v in row else "—" for v in V) + " |")
lines += ["", "Paired difference vs V1 (mean [95% CI]):", "", "| metric | " + " | ".join(v for v in V if v != "V1") + " |",
          "|---|" + "---|" * (len(V) - 1)]
for key in show:
    row = summary["diff_vs_V1"].get(key, {})
    p = 1 if key.startswith("ICR_raw") else 3
    lines.append(f"| {key} | " + " | ".join(fmt(row[v], p) if v in row else "—" for v in V if v != "V1") + " |")
open(os.path.join(args.dir, f"tables_k{args.k}.md"), "w").write("\n".join(lines) + "\n")

# plots
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (key, title) in zip(axes.flat, [("kl", "KL(model || lens)"), ("top1", "top-1 agreement with model"),
                                        ("cos_c", "cos(fhat, y - ybar)"), ("df_rel", "median ||fhat - Δf|| / ||Δf||"),
                                        ("df_cos", "cos(fhat, Δf)"), ("icr_cells", "ICR cells per layer (avg of 6 sets)")]):
    for v in V:
        if key in per_layer[v]:
            ax.plot(layers, per_layer[v][key], "o-", label=v, ms=3)
    ax.set_title(title)
    ax.set_xlabel("layer")
axes[0][0].legend()
fig.tight_layout()
fig.savefig(os.path.join(args.dir, f"per_layer_k{args.k}.png"), dpi=130)
print("\n".join(lines))
