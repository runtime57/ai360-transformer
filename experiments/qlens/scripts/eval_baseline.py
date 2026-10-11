"""Phase 0 baseline: V0 (J-lens, as Anthropic) and V1 (centred) at every fitted layer,
plus the logit lens as a sanity reference. Held-out WikiText-103 validation + 6 lens-eval sets."""

import argparse
import os

import numpy as np
import torch
import yaml

from qlens.core import (WORK, Timer, encode, get_splits, git_hash, load_lens, load_model, save_json,
                        seed_all)
from qlens.evaluate import eval_heldout, eval_lenseval
from qlens.lenseval import load_sets

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
seed_all(cfg["seed"])
tag = cfg["tag"]
out_dir = os.path.join(os.environ["QLENS_ROOT"], "results", "phase0", tag)
model, tok = load_model(cfg["model"], torch.bfloat16)
L = model.config.num_hidden_layers
splits = get_splits(tok, cfg["data"])
J = {l: m.cuda() for l, m in load_lens(os.path.join(WORK, tag, "lens.pt")).items()}
st = torch.load(os.path.join(WORK, tag, "stats.pt"))
mu = st["mu"].cuda()
layers = sorted(J)

variants = {
    "logit": lambda l, h: h,
    "V0": lambda l, h: h @ J[l].T,
    "V1": lambda l, h: (h - mu[l]) @ J[l].T,
}
held_ids = [encode(tok, t, "cuda") for _, t in splits["heldout"]]
sets = load_sets(cfg["jlens_repo"], tok)
with Timer("heldout") as t1:
    held = eval_heldout(model, held_ids, layers, variants, st["ybar"], {})
with Timer("lenseval") as t2:
    le = eval_lenseval(model, sets, layers, variants, tuple(cfg["band"]))
torch.save({"heldout": held, "lenseval": le}, os.path.join(out_dir, "baseline_raw.pt"))

summ = {"config": cfg, "git": git_hash(), "seed": cfg["seed"], "seconds": {"heldout": t1.dt, "lenseval": t2.dt},
        "layers": layers, "band_layers": le["_band_layers"]}
for v in variants:
    h = held[v]
    s = {k: np.mean(h[k], axis=0).tolist() for k in ("kl", "top1", "cos", "rel", "cos_c", "rel_c")}
    s["SHL_with_last"] = float(np.mean(h["shl_with"]))
    s["SHL_without_last"] = float(np.mean(h["shl_without"]))
    icr = {name: float(np.mean([r["icr_raw"] for r in le[name][v]])) for name in sets}
    icr["avg"] = float(np.mean(list(icr.values())))
    s["ICR_raw"] = icr
    s["norm_any"] = {name: float(np.mean([r["norm_any"] for r in le[name][v]])) for name in sets}
    s["norm_ro"] = {name: float(np.mean([r["norm_ro"] for r in le[name][v]])) for name in sets}
    summ[v] = s
    print(v, "SHL", s["SHL_with_last"], s["SHL_without_last"], "ICR", icr, flush=True)
multi = {name: float(np.mean([c["multi_token"] for r in sets[name] for c in r["concepts"]])) for name in sets}
summ["multi_token_share"] = multi
save_json(summ, os.path.join(out_dir, "baseline.json"))
