"""Unbiased ||E_{c,t} H[delta,delta]||^2 = (||sum X||^2 - sum ||X_m||^2) / (M(M-1)) per direction,
and the resulting debiased relative size ||1/2 E H|| / ||J delta|| (Phase 1 add-on, CPU only)."""
import json, os, sys
import torch
from qlens.core import WORK, load_lens

tag = sys.argv[1]
r = json.load(open(os.path.join(os.environ["QLENS_ROOT"], "results", "phase1", tag, "phase1.json")))
D = torch.load(os.path.join(WORK, tag, "phase1", "directions.pt"))
J = load_lens(os.path.join(WORK, tag, "lens.pt"))
out = {}
for l, v in r["per_layer"].items():
    l = int(l)
    X = torch.cat([torch.load(os.path.join(WORK, tag, "phase1", f"L{l}_c{c:03d}.pt"))["H"] for c in range(v["N"])]).double()
    M = X.shape[0]
    sq = (X.sum(0).pow(2).sum(-1) - X.pow(2).sum((0, 2))) / (M * (M - 1))        # [n_dirs]
    Jd = (D["delta"][l].double() @ J[l].double().T).norm(dim=-1)
    rel = 0.5 * sq.clamp_min(0).sqrt() / Jd
    out[l] = {"M": M, "frac_dirs_positive_sq": (sq > 0).float().mean().item(),
              "debiased_rel_size_mean": rel.mean().item(), "debiased_rel_size_median": rel.median().item(),
              "debiased_norm_mean": sq.clamp_min(0).sqrt().mean().item()}
    print(tag, l, out[l], flush=True)
json.dump(out, open(os.path.join(os.environ["QLENS_ROOT"], "results", "phase1", tag, "debiased.json"), "w"), indent=1)
