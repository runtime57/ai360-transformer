"""Compact, re-usable artifacts (no per-sample Hessians): Hbar means for Phase 1 / Phase 2, U_J, mu."""
import os, sys, torch, json
from qlens.core import WORK
tag = sys.argv[1]
out = os.path.join(os.environ["QLENS_ROOT"], "results", "artifacts", tag)
os.makedirs(out, exist_ok=True)
st = torch.load(os.path.join(WORK, tag, "stats.pt"))
torch.save({"mu": st["mu"], "ybar": st["ybar"]}, os.path.join(out, "stats_mu_ybar.pt"))
p1 = os.path.join(os.environ["QLENS_ROOT"], "results", "phase1", tag, "phase1.json")
if os.path.exists(p1):
    r = json.load(open(p1))
    D = torch.load(os.path.join(WORK, tag, "phase1", "directions.pt"))
    H = {}
    for l, v in r["per_layer"].items():
        X = torch.cat([torch.load(os.path.join(WORK, tag, "phase1", f"L{l}_c{c:03d}.pt"))["H"] for c in range(v["N"])])
        H[int(l)] = {"mean": X.mean(0), "M": X.shape[0]}
    torch.save({"directions": D, "Hbar_dd": H}, os.path.join(out, "phase1_Hbar_dd.pt"))
w2 = os.path.join(WORK, tag, "phase2_k16")
if os.path.exists(os.path.join(w2, "subspace.pt")):
    sub = torch.load(os.path.join(w2, "subspace.pt"))
    H = {}
    for l in sub["U"]:
        fs = sorted(f for f in os.listdir(w2) if f.startswith(f"L{l}_c"))
        X = torch.cat([torch.load(os.path.join(w2, f)) for f in fs])
        H[l] = {"mean_pairs": X.mean(0), "M": X.shape[0]}
    torch.save({"U": sub["U"], "info": sub["info"], "Hbar_pairs_i_le_j": H, "k": 16}, os.path.join(out, "phase2_U_Hbar_k16.pt"))
print("exported", out, os.listdir(out))
