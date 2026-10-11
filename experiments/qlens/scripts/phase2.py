"""Phase 2 (H2): the quadratic lens in a k-dim J-space subspace, accuracy and ICR.

Subspace U_J (per layer): for fit-corpus activations delta = h - mu, take the 25 tokens with the
largest J-lens score <delta, w_v>, w_v = Jbar^T (g * W_U[v]) (g = final-norm gain; this is the
V1 readout ranking), project delta onto span{w_v} (least squares; the repo has no sparse
non-negative decomposition), and take the top-k principal directions of these projections.

  Hbar[delta,delta] ~= sum_{i,j} z_i z_j Hbar[u_i,u_j],   z = U^T delta   (i <= j stored)

Variants (same layers): V0 = J h, V1 = J delta, V2 = J delta + 1/2 Hbar_k[delta,delta],
C1 = V1 + random vector with the norm of 1/2 Hbar_k[delta,delta],
C2 = V1 + the other-layer term 1/2 Hbar_k^{(l')}[delta,delta] rescaled to the own term's norm.

Extra accuracy target (fp32, held-out subsample): the exact mean-ablation response
Delta f = f_{c,t}(h) - f_{c,t}(mu) that fhat(delta) approximates.

  --stage subspace|hessian|eval|all   --bench
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import yaml

from qlens.core import (WORK, QwenTail, Timer, encode, get_splits, git_hash, layers_for, load_lens,
                        load_model, record, save_json, seed_all, unembed, valid_positions)
from qlens.evaluate import eval_heldout, eval_lenseval
from qlens.lenseval import load_sets

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--stage", default="all")
ap.add_argument("--bench", action="store_true")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
seed_all(cfg["seed"])
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
tag, k = cfg["tag"], cfg["k"]
wdir = os.path.join(WORK, tag, f"phase2_k{k}")
out_dir = os.path.join(os.environ["QLENS_ROOT"], "results", "phase2", tag)
os.makedirs(wdir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

model, tok = load_model(cfg["model"], torch.float32)
tail = QwenTail(model)
L, d = tail.n_layers, model.config.hidden_size
layers = layers_for(cfg["layer_pcts"], L, last_k=cfg["last_k"])
splits = get_splits(tok, cfg["data"])
J = {l: m.cuda() for l, m in load_lens(os.path.join(WORK, tag, "lens.pt")).items() if l in layers}
st = torch.load(os.path.join(WORK, tag, "stats.pt"))
mu = st["mu"].cuda()
pairs = [(i, j) for i in range(k) for j in range(i, k)]
meta = {"config": cfg, "git": git_hash(), "seed": cfg["seed"], "layers": layers, "n_pairs": len(pairs)}
stages = ["subspace", "hessian", "eval"] if args.stage == "all" else [args.stage]

# ------------------------------------------------------------------ subspace U_J
sub_path = os.path.join(wdir, "subspace.pt")
if "subspace" in stages and not os.path.exists(sub_path):
    with Timer("subspace") as tm:
        g = model.model.norm.weight
        WU = model.lm_head.weight * g                                   # [V, d]
        rng = random.Random(cfg["seed"] + 40)
        fit_ids = [encode(tok, t, "cuda") for _, t in splits["fit"]]
        held_ids = [encode(tok, t, "cuda") for _, t in splits["heldout"]]

        def sample_deltas(id_list, per_seq):
            out = {l: [] for l in layers}
            for ids in id_list:
                pos = torch.tensor(rng.sample(valid_positions(ids.shape[1]), per_seq), device="cuda")
                acts = record(model, ids, layers)
                for l in layers:
                    out[l].append(acts[l][0, pos] - mu[l])
            return {l: torch.cat(v) for l, v in out.items()}

        fit_d = sample_deltas(fit_ids, cfg["pca_per_seq"])
        held_d = sample_deltas(held_ids, cfg["pca_per_seq"])
        U, info = {}, {}
        for l in layers:
            dl = fit_d[l]
            proj = []
            for b0 in range(0, len(dl), 256):
                x = dl[b0:b0 + 256]
                top = ((x @ J[l].T) @ WU.T).topk(cfg["top_tokens"], -1).indices      # V1 ranking
                W = WU[top] @ J[l]                                                  # [b, 25, d]
                sol = torch.linalg.lstsq(W.transpose(1, 2), x[..., None]).solution  # [b, 25, 1]
                proj.append((W.transpose(1, 2) @ sol)[..., 0])
            P = torch.cat(proj)
            evals, evecs = torch.linalg.eigh(P.T @ P / len(P))
            Ul = evecs[:, -k:].flip(-1).contiguous()                                # [d, k]
            U[l] = Ul.cpu()
            hd = held_d[l]
            cap = ((hd @ Ul).pow(2).sum(-1) / hd.pow(2).sum(-1))
            jfrac = (P.pow(2).sum(-1) / dl.pow(2).sum(-1))
            info[l] = {"captured_frac_mean": cap.mean().item(),
                       "captured_frac_pooled": ((hd @ Ul).pow(2).sum() / hd.pow(2).sum()).item(),
                       "jspace_frac_of_delta_mean": jfrac.mean().item(),
                       "pca_explained_of_jspace": (evals[-k:].sum() / evals.sum()).item()}
            print(l, info[l], flush=True)
        torch.save({"U": U, "info": info}, sub_path)
    meta["subspace_seconds"] = tm.dt
sub = torch.load(sub_path) if os.path.exists(sub_path) else None
if sub is not None:
    meta["subspace"] = sub["info"]

# ------------------------------------------------------------------ Hbar[u_i, u_j]
rng = random.Random(cfg["seed"] + 20)  # same contexts / positions as Phase 1
ctx = []
for _, text in splits["hess"][: cfg["n_ctx"]]:
    ids = encode(tok, text, "cpu")
    ctx.append((ids, sorted(rng.sample(valid_positions(ids.shape[1]), cfg["S"]))))
pi = torch.tensor([p[0] for p in pairs])
pj = torch.tensor([p[1] for p in pairs])


def compute(l, c):
    path = os.path.join(wdir, f"L{l}_c{c:03d}.pt")
    if os.path.exists(path):
        return
    ids, ts = ctx[c]
    cache = tail.clean(ids.cuda())
    Ul = sub["U"][l].cuda().T                                          # [k, d]
    out = torch.empty(len(ts), len(pairs), d)
    for si, t in enumerate(ts):
        for b0 in range(0, len(pairs), cfg["batch"]):
            sl = slice(b0, b0 + cfg["batch"])
            out[si, sl] = tail.hess(l, t, cache, Ul[pi[sl]], Ul[pj[sl]]).cpu()
    torch.save(out, path + ".tmp")
    os.replace(path + ".tmp", path)


if args.bench:
    res = {}
    for l in layers:
        compute(l, 0)
        os.remove(os.path.join(wdir, f"L{l}_c000.pt"))
        with Timer(f"layer {l}") as tm:
            compute(l, 0)
        res[l] = tm.dt
    tot = sum(res.values()) * cfg["n_ctx"]
    print(f"bench {res}; full ~ {tot/3600:.2f} h; peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB", flush=True)
    save_json({"per_context_seconds": res, "extrapolated_hours": tot / 3600, "config": cfg},
              os.path.join(out_dir, f"bench_k{k}.json"))
    raise SystemExit

if "hessian" in stages:
    with Timer("hessian") as tm:
        for c in range(cfg["n_ctx"]):
            for l in layers:
                compute(l, c)
            if c % 8 == 7:
                print(f"context {c+1}/{cfg['n_ctx']}", flush=True)
    meta["hessian_seconds"] = tm.dt

if "eval" not in stages:
    save_json(meta, os.path.join(out_dir, f"meta_k{k}_{args.stage}.json"))
    raise SystemExit

# ------------------------------------------------------------------ assemble Hbar
Hfull, Hsd = {}, {}
for l in layers:
    X = torch.cat([torch.load(os.path.join(wdir, f"L{l}_c{c:03d}.pt")) for c in range(cfg["n_ctx"])])
    Hb = X.mean(0)                                                       # [n_pairs, d]
    M = torch.zeros(k, k, d)
    M[pi, pj] = Hb
    M[pj, pi] = Hb
    Hfull[l] = M.cuda()
    signs = (torch.randint(0, 2, (100, X.shape[0])) * 2 - 1).float()
    null = torch.stack([(s[:, None, None] * X).mean(0).norm(dim=-1) for s in signs])
    Hsd[l] = {"pair_norm_mean": Hb.norm(dim=-1).mean().item(), "pair_null_mean": null.mean().item()}
meta["Hbar_pairs_vs_null"] = Hsd
Us = {l: sub["U"][l].cuda() for l in layers}
other = {l: layers[(i + len(layers) // 2) % len(layers)] for i, l in enumerate(layers)}
meta["C2_other_layer"] = other


def quad(l, delta, H_of=None):
    z = delta @ Us[H_of or l]
    return 0.5 * torch.einsum("ni,nj,ijd->nd", z, z, Hfull[H_of or l])


gC1 = torch.Generator(device="cuda").manual_seed(cfg["seed"] + 50)


def v2(l, h):
    dl = h - mu[l]
    return dl @ J[l].T + quad(l, dl)


def c1(l, h):
    dl = h - mu[l]
    q = quad(l, dl)
    r = torch.randn(q.shape, device="cuda", generator=gC1)
    return dl @ J[l].T + r * (q.norm(dim=-1, keepdim=True) / r.norm(dim=-1, keepdim=True))


def c2(l, h):
    dl = h - mu[l]
    q = quad(l, dl)
    qo = quad(l, dl, H_of=other[l])
    return dl @ J[l].T + qo * (q.norm(dim=-1, keepdim=True) / qo.norm(dim=-1, keepdim=True).clamp_min(1e-12))


variants = {
    "V0": lambda l, h: h @ J[l].T,
    "V1": lambda l, h: (h - mu[l]) @ J[l].T,
    "V2": v2, "C1": c1, "C2": c2,
}

# ---- exact mean-ablation response (fp32, before switching to bf16)
with Timer("delta_f") as tm:
    rng2 = random.Random(cfg["seed"] + 60)
    dres = {v: {l: {"cos": [], "rel": []} for l in layers} for v in variants if v != "V0"}
    dres["quad_vs_residual"] = {l: [] for l in layers}
    for _, text in splits["heldout"]:
        ids = encode(tok, text, "cuda")
        cache = tail.clean(ids)
        ts = rng2.sample(valid_positions(ids.shape[1]), cfg["deltaf_per_seq"])
        for l in layers:
            for t in ts:
                x = cache.hidden[l][:, t:].clone()
                with torch.no_grad():
                    f_h = tail.f(x, l, t, cache)
                    x[:, 0] = mu[l]
                    f_mu = tail.f(x, l, t, cache)
                target = (f_h - f_mu)[0]
                h = cache.hidden[l][:, t]
                lin = ((h - mu[l]) @ J[l].T)[0]
                for v in dres:
                    if v == "quad_vs_residual":
                        continue
                    fh = variants[v](l, h)[0]
                    dres[v][l]["cos"].append(torch.cosine_similarity(fh, target, dim=0).item())
                    dres[v][l]["rel"].append(((fh - target).norm() / target.norm()).item())
                q = quad(l, h - mu[l])[0]
                # does the quadratic term point along the part the linear term misses?
                dres["quad_vs_residual"][l].append(torch.cosine_similarity(q, target - lin, dim=0).item())
meta["delta_f_seconds"] = tm.dt
torch.save(dres, os.path.join(out_dir, f"deltaf_raw_k{k}.pt"))

# ---- bf16 evaluation (held-out + lens-eval)
model.to(torch.bfloat16)
held_ids = [encode(tok, t, "cuda") for _, t in splits["heldout"]]
sets = load_sets(cfg["jlens_repo"], tok)
with Timer("eval") as tm:
    held = eval_heldout(model, held_ids, layers, variants, st["ybar"], {})
    le = eval_lenseval(model, sets, layers, variants, tuple(cfg["band"]))
meta["eval_seconds"] = tm.dt
torch.save({"heldout": held, "lenseval": le}, os.path.join(out_dir, f"eval_raw_k{k}.pt"))
save_json(meta, os.path.join(out_dir, f"meta_k{k}.json"))
print("phase2 eval done", flush=True)
