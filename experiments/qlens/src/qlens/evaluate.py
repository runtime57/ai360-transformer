"""Lens metrics shared by Phase 0 (baseline) and Phase 2 (variants).

A *variant* is a callable  (layer, h[n, d] fp32) -> fhat[n, d] fp32  in the final-block basis;
its readout is  softmax(W_U . norm(fhat))  (``core.unembed``).

Held-out (per sequence, valid positions t in [16, T-2]):
  kl      KL(p_model || p_lens)
  top1    argmax lens == argmax model
  cos / rel   cos(fhat, y), ||fhat - y|| / ||y||, with y[t] = sum_{t'=t}^{T-2} h_final[t']
              (raw y for uncentred V0, y - ybar(t) for centred variants; both are stored)
  SHL     (Alibaba eq. 21) per position: the length of the run of layers, from the top, whose
          top-1 equals the true next token; "with last" starts the run at the model output
          (block L-1), "without" at the top lens layer. The run only continues over *consecutive*
          layer indices that were evaluated.
lens-eval (per prompt, all positions):
  icr_raw   #(layer, position) cells whose top-10 contains a concept token (Alibaba eq. 22)
  norm_any  fraction of the prompt's concepts in top-10 at >= 1 cell in the band layers
  norm_ro   same, at the readout position only (the Anthropic readout)
"""

from __future__ import annotations

import torch

from qlens.core import record, unembed, valid_positions


def _shl(correct: torch.Tensor, layers: list[int], final_correct: torch.Tensor | None):
    """correct: [n_layers, n] bool for ascending `layers`. Returns per-position run length."""
    order = sorted(range(len(layers)), key=lambda i: -layers[i])
    n = correct.shape[1]
    run = torch.zeros(n, dtype=torch.long, device=correct.device)
    alive = torch.ones(n, dtype=torch.bool, device=correct.device)
    prev = None
    if final_correct is not None:
        alive &= final_correct
        run += alive.long()
        prev = layers[order[0]] + 1  # model output sits right above the top lens layer
    for i in order:
        if prev is not None and layers[i] != prev - 1:
            break
        alive &= correct[i]
        run += alive.long()
        prev = layers[i]
    return run.float()


@torch.no_grad()
def eval_heldout(model, texts_ids, layers, variants, ybar, centred: dict[str, bool]):
    L = model.config.num_hidden_layers
    out = {v: {"kl": [], "top1": [], "cos": [], "rel": [], "cos_c": [], "rel_c": [],
               "shl_with": [], "shl_without": [], "next_top1": []} for v in variants}
    ybar = ybar.cuda()
    for ids in texts_ids:
        T = ids.shape[1]
        acts = record(model, ids, layers)
        pos = torch.tensor(valid_positions(T), device="cuda")
        nxt = ids[0, pos + 1]
        fin = acts[L - 1][0].float()
        logp_m = torch.log_softmax(unembed(model, fin[pos]), -1)
        top_m = logp_m.argmax(-1)
        y = fin[: T - 1].flip(0).cumsum(0).flip(0)[pos]
        yc = y - ybar[pos]
        for v, fn in variants.items():
            per = {k: [] for k in ("kl", "top1", "cos", "rel", "cos_c", "rel_c")}
            correct = []
            for l in layers:
                fh = fn(l, acts[l][0, pos].float())
                logq = torch.log_softmax(unembed(model, fh), -1)
                per["kl"].append((logp_m.exp() * (logp_m - logq)).sum(-1).mean().item())
                tq = logq.argmax(-1)
                per["top1"].append((tq == top_m).float().mean().item())
                correct.append(tq == nxt)
                for tgt, sfx in ((y, ""), (yc, "_c")):
                    per["cos" + sfx].append(torch.cosine_similarity(fh, tgt, dim=-1).mean().item())
                    per["rel" + sfx].append(((fh - tgt).norm(dim=-1) / tgt.norm(dim=-1)).mean().item())
            correct = torch.stack(correct)
            out[v]["next_top1"].append(correct.float().mean(-1).tolist())  # per layer
            for k, vals in per.items():
                out[v][k].append(vals)
            out[v]["shl_with"].append(_shl(correct, layers, top_m == nxt).mean().item())
            out[v]["shl_without"].append(_shl(correct, layers, None).mean().item())
    out["_layers"] = list(layers)
    return out


@torch.no_grad()
def eval_lenseval(model, sets, layers, variants, band: tuple[float, float], k: int = 10):
    L = model.config.num_hidden_layers
    band_idx = [i for i, l in enumerate(layers) if band[0] * L <= l <= band[1] * L]
    res = {}
    for name, rows in sets.items():
        res[name] = {v: [] for v in variants}
        for row in rows:
            ids = torch.tensor([row["ids"]], device="cuda")
            acts = record(model, ids, layers)
            for v, fn in variants.items():
                tops = torch.stack([unembed(model, fn(l, acts[l][0].float())).topk(k, -1).indices
                                    for l in layers])               # [n_layers, T, k]
                hits = []
                for c in row["concepts"]:
                    tk = torch.tensor(c["tokens"], device="cuda")
                    hits.append(torch.isin(tops, tk).any(-1))      # [n_layers, T]
                hits = torch.stack(hits)                            # [n_c, n_layers, T]
                any_c = hits.any(0)
                hb = hits[:, band_idx]
                res[name][v].append({
                    "icr_raw": int(any_c.sum()),
                    "icr_per_layer": any_c.sum(-1).tolist(),
                    "norm_any": hb.flatten(1).any(-1).float().mean().item(),
                    "norm_ro": hb[:, :, row["readout"]].any(-1).float().mean().item(),
                    "concept_any": hb.flatten(1).any(-1).tolist(),
                    "concept_multi": [c["multi_token"] for c in row["concepts"]],
                })
    res["_layers"] = list(layers)
    res["_band_layers"] = [layers[i] for i in band_idx]
    return res
