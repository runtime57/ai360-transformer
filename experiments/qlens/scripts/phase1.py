"""Phase 1 (H1): is the averaged Hessian ~ 0?

For layers at {15,40,65,90}% depth, 100 held-out directions delta = h - mu_l, and contexts c
(train, disjoint from the fit set) with S random valid source positions each, computes
H_{c,t}[delta, delta] (nested forward-mode AD over the tail, prefix KV reused, fp32 + TF32),
in blocks of contexts 16 -> 32 -> 64 with per-layer early stopping. One file per
(layer, context) under work/<tag>/phase1/ -> resumable.

  --bench   time one (context, position) per layer and extrapolate to the full run.
"""

import argparse
import glob
import math
import os
import random

import torch
import yaml

from qlens.core import (WORK, QwenTail, Timer, encode, get_splits, git_hash, layers_for, load_lens,
                        load_model, save_json, seed_all, unembed, valid_positions)

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--bench", action="store_true")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
seed_all(cfg["seed"])
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
tag = cfg["tag"]
wdir = os.path.join(WORK, tag, "phase1")
out_dir = os.path.join(os.environ["QLENS_ROOT"], "results", "phase1", tag)
os.makedirs(wdir, exist_ok=True)

model, tok = load_model(cfg["model"], torch.float32)
tail = QwenTail(model)
L = tail.n_layers
layers = cfg.get("layers") or layers_for(cfg["layer_pcts"], L)
splits = get_splits(tok, cfg["data"])
J = {l: m.cuda() for l, m in load_lens(os.path.join(WORK, tag, "lens.pt")).items() if l in layers}
mu = torch.load(os.path.join(WORK, tag, "stats.pt"))["mu"].cuda()

# ---- directions: delta = h_l - mu_l at random held-out (sequence, valid position)
dpath = os.path.join(wdir, "directions.pt")
if os.path.exists(dpath):
    D = torch.load(dpath)
else:
    rng = random.Random(cfg["seed"] + 10)
    held = [encode(tok, t, "cuda") for _, t in splits["heldout"]]
    picks = [(rng.randrange(len(held)),) for _ in range(cfg["n_dirs"])]
    picks = [(i, rng.choice(valid_positions(held[i].shape[1]))) for (i,) in picks]
    D = {"picks": picks, "delta": {}}
    by_seq = {}
    for k, (i, t) in enumerate(picks):
        by_seq.setdefault(i, []).append((k, t))
    vecs = {l: torch.zeros(cfg["n_dirs"], model.config.hidden_size) for l in layers}
    for i, lst in by_seq.items():
        cache = tail.clean(held[i])
        for l in layers:
            for k, t in lst:
                vecs[l][k] = (cache.hidden[l][0, t] - mu[l]).cpu()
    D["delta"] = vecs
    torch.save(D, dpath)

# ---- contexts and source positions
rng = random.Random(cfg["seed"] + 20)
ctx = []
for idx, (_, text) in enumerate(splits["hess"][: cfg["n_ctx_max"]]):
    ids = encode(tok, text, "cpu")
    ctx.append((ids, sorted(rng.sample(valid_positions(ids.shape[1]), cfg["S"]))))


def compute(l, c):
    path = os.path.join(wdir, f"L{l}_c{c:03d}.pt")
    if os.path.exists(path):
        return
    ids, ts = ctx[c]
    cache = tail.clean(ids.cuda())
    delta = D["delta"][l].cuda()
    out = torch.empty(len(ts), delta.shape[0], delta.shape[1])
    for si, t in enumerate(ts):
        for b0 in range(0, delta.shape[0], cfg["batch"]):
            U = delta[b0:b0 + cfg["batch"]]
            out[si, b0:b0 + len(U)] = tail.hess(l, t, cache, U, U).cpu()
    torch.save({"H": out, "t": ts}, path + ".tmp")
    os.replace(path + ".tmp", path)


if args.bench:
    res = {}
    for l in layers:
        compute(l, 0)  # warm-up / also a real result
        os.remove(os.path.join(wdir, f"L{l}_c000.pt"))
        with Timer(f"layer {l}: 1 context x S={cfg['S']} x {cfg['n_dirs']} dirs") as tm:
            compute(l, 0)
        res[l] = tm.dt
    tot = sum(res.values()) * cfg["n_ctx_max"]
    print(f"bench per-context seconds {res}; full run (all layers, N={cfg['n_ctx_max']}) ~ {tot/3600:.2f} h; "
          f"peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    save_json({"per_context_seconds": res, "extrapolated_hours": tot / 3600,
               "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, "config": cfg},
              os.path.join(out_dir, "bench.json"))
    raise SystemExit


# ---- analysis helpers
def load(l, n):
    return torch.cat([torch.load(os.path.join(wdir, f"L{l}_c{c:03d}.pt"))["H"] for c in range(n)])  # [n*S, nd, d]


gen = torch.Generator().manual_seed(cfg["seed"] + 30)


def stats(l, X, Jd):
    """X: [M, nd, d] per-sample H_{c,t}[delta,delta]; Jd: [nd, d] = Jbar delta."""
    M = X.shape[0]
    Hbar = X.mean(0)
    hn = Hbar.norm(dim=-1)                                            # per direction
    signs = (torch.randint(0, 2, (cfg["n_null"], M), generator=gen) * 2 - 1).float()
    null = torch.stack([(s[:, None, None] * X).mean(0).norm(dim=-1) for s in signs])  # [R, nd]
    rel = (0.5 * hn / Jd.norm(dim=-1))
    per_ctx_rel = (0.5 * X.norm(dim=-1) / Jd.norm(dim=-1)).mean().item()
    with torch.no_grad():
        a = unembed(model, Jd.cuda()).topk(10, -1).indices
        b = unembed(model, (Jd + 0.5 * Hbar).cuda()).topk(10, -1).indices
        overlap = torch.tensor([len(set(x.tolist()) & set(y.tolist())) / 10 for x, y in zip(a, b)])
    return {
        "M": M, "norm": hn.mean().item(), "null_mean": null.mean().item(),
        "null_q95": null.mean(1).quantile(0.95).item(),
        "z": (hn.mean() / null.mean()).item(),
        "frac_dirs_above_null_q95": (hn > null.quantile(0.95, dim=0)).float().mean().item(),
        "rel_size_mean": rel.mean().item(), "rel_size_median": rel.median().item(),
        "per_context_rel_size_mean": per_ctx_rel,
        "top10_overlap_mean": overlap.mean().item(), "top10_overlap_min": overlap.min().item(),
    }


# ---- main loop with blocks + early stopping
state_path = os.path.join(out_dir, "phase1.json")
res = {"config": cfg, "git": git_hash(), "seed": cfg["seed"], "layers": layers, "per_layer": {},
       "seconds": 0.0}
open_layers = list(layers)
done_n = 0
for N in cfg["blocks"]:
    with Timer(f"block N={N}, layers {open_layers}") as tm:
        for c in range(done_n, N):
            for l in open_layers:
                compute(l, c)
    res["seconds"] += tm.dt
    for l in list(open_layers):
        Jd = D["delta"][l].cuda() @ J[l].T
        X = load(l, N)
        curve = {}
        n = 2
        while n <= N:
            curve[n] = stats(l, X[: n * cfg["S"]], Jd.cpu())
            n *= 2
        cur, half = curve[N], curve[N // 2]
        ratio = cur["norm"] / half["norm"]
        nonzero = cur["z"] >= 3 and abs(ratio - 1) <= cfg["ratio_tol"]
        decision = "closed: nonzero mean" if nonzero else ("continue" if N < cfg["blocks"][-1] else "max N")
        res["per_layer"][l] = {"N": N, "curve": curve, "ratio_N_over_half": ratio,
                               "ratio_null_expect": 1 / math.sqrt(2), "decision": decision}
        print(l, N, {k: round(v, 4) if isinstance(v, float) else v for k, v in cur.items()},
              "ratio", round(ratio, 3), decision, flush=True)
        if nonzero:
            open_layers.remove(l)
    done_n = N
    save_json(res, state_path)
    if not open_layers:
        break

# ---- verdict
ver = {}
for l, r in res["per_layer"].items():
    cur = r["curve"][r["N"]]
    ver[l] = {"indistinguishable_from_null": cur["norm"] <= cur["null_q95"],
              "rel_size_lt_5pct": cur["rel_size_mean"] < 0.05,
              "top10_overlap_gt_0.95": cur["top10_overlap_mean"] > 0.95}
res["verdict_per_layer"] = ver
res["H1_confirmed"] = all(all(v.values()) for v in ver.values())
save_json(res, state_path)
print("H1 confirmed:", res["H1_confirmed"], ver, flush=True)
