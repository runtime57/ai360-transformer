"""Phase 0 correctness checks on a small model (Qwen3-0.6B).

1. functional Qwen3 stack == HF forward (block outputs);
2. prefix-reuse tail == full recomputation (f, J u, H[u,v]);
3. nested forward-mode H[u,u] vs central finite differences (fp64): rel err < 1e-3;
4. TF32 / fp32 Hessian vs fp64 (how much precision we lose in production mode).
"""

import argparse
import os
import sys

import torch
import yaml

from qlens.core import (QwenTail, encode, get_splits, git_hash, load_model, save_json, seed_all,
                        valid_positions)

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
seed_all(cfg["seed"])
out_dir = os.path.join(os.environ["QLENS_ROOT"], cfg["out_dir"])
res = {"config": cfg, "git": git_hash(), "seed": cfg["seed"]}

model, tok = load_model(cfg["model"], torch.float64)
tail = QwenTail(model)
splits = get_splits(tok, cfg["data"])
ids = encode(tok, splits["heldout"][0][1], "cuda")
T = ids.shape[1]
L = tail.n_layers

# 1. functional == HF
with torch.no_grad():
    hf = model.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
    cache = tail.clean(ids)
# hidden_states[j+1] is the output of block j, except the last which is post-norm.
errs = [((cache.hidden[j] - hf[j + 1]).norm() / hf[j + 1].norm()).item() for j in range(L - 1)]
res["functional_vs_hf_max_rel_err"] = max(errs)
print("functional vs HF max rel err:", max(errs), flush=True)


def full_f(x_row, layer, t):
    """Reference without prefix reuse: run all positions 0..T-1 from `layer` with row t replaced."""
    h = cache.hidden[layer].expand(x_row.shape[0], -1, -1).clone()
    h[:, t] = x_row
    out = tail.run(h, layer, 0, cache)
    return out[:, t:T - 1].sum(1)


checks = []
gen = torch.Generator(device="cuda").manual_seed(cfg["seed"])
for layer in cfg["check_layers"]:
    for t in cfg["check_positions"]:
        base = cache.hidden[layer][:, t]  # [1, d]
        scale = base.norm()
        u = torch.randn(2, base.shape[-1], device="cuda", dtype=torch.float64, generator=gen)
        u = u / u.norm(dim=-1, keepdim=True) * scale  # directions with the size of a residual
        v = u.flip(0)

        # 2. prefix reuse vs full recompute
        from torch.func import jvp

        H_pre, J_pre = tail.hess(layer, t, cache, u, v, with_jvp=True)
        f_pre = tail.f(cache.hidden[layer][:, t:].clone(), layer, t, cache)
        f_full = full_f(base, layer, t)

        def inner(x):
            return jvp(lambda z: full_f(z, layer, t), (x,), (u,))[1]

        J_full, H_full = jvp(inner, (base.expand(2, -1).contiguous(),), (v,))
        rel = lambda a, b: ((a - b).norm() / b.norm()).item()
        c = {"layer": layer, "t": t,
             "prefix_vs_full_f": rel(f_pre, f_full), "prefix_vs_full_J": rel(J_pre, J_full),
             "prefix_vs_full_H": rel(H_pre, H_full)}

        # 3. H[u,u] vs central finite differences (fp64)
        Huu = tail.hess(layer, t, cache, u, u)
        fd = {}
        with torch.no_grad():
            f0 = full_f(base.expand(2, -1), layer, t)
            for eps in cfg["fd_eps"]:
                fp = full_f(base + eps * u, layer, t)
                fm = full_f(base - eps * u, layer, t)
                Hfd = (fp - 2 * f0 + fm) / eps ** 2
                fd[str(eps)] = rel(Huu, Hfd)
        c["fd_rel_err"] = fd
        c["fd_rel_err_best"] = min(fd.values())
        c["H_norm_over_J_norm"] = (Huu.norm(dim=-1) / J_pre.norm(dim=-1)).tolist()
        checks.append(c)
        print(c, flush=True)

res["checks"] = checks
res["max_prefix_vs_full_H"] = max(c["prefix_vs_full_H"] for c in checks)
res["max_fd_rel_err_best"] = max(c["fd_rel_err_best"] for c in checks)

# 4. production precision (fp32 weights + TF32 matmuls) vs fp64
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
m32 = model.float()
tail32 = QwenTail(m32)
ids32 = ids
with torch.no_grad():
    cache32 = tail32.clean(ids32)
prec = []
for layer in cfg["check_layers"]:
    t = cfg["check_positions"][0]
    base = cache32.hidden[layer][:, t]
    u = torch.randn(4, base.shape[-1], device="cuda", generator=gen) * base.norm() / base.shape[-1] ** 0.5
    H32 = tail32.hess(layer, t, cache32, u, u)
    model.double()
    H64 = tail.hess(layer, t, tail.clean(ids), u.double(), u.double())
    model.float()
    prec.append({"layer": layer, "tf32_vs_fp64_rel_err": ((H32.double() - H64).norm() / H64.norm()).item()})
    print(prec[-1], flush=True)
res["tf32_precision"] = prec

# HF eager attention takes softmax in fp32 even for fp64 models -> ~1e-7 floor.
ok = (res["functional_vs_hf_max_rel_err"] < 1e-5 and res["max_prefix_vs_full_H"] < 1e-8
      and res["max_fd_rel_err_best"] < 1e-3)
res["PASS"] = bool(ok)
save_json(res, os.path.join(out_dir, "checks.json"))
print("PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
