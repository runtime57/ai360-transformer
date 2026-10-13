"""Fit the Jacobian lens J_l = E_{prompt, t, t'>=t}[d h_final,t' / d h_l,t].

Per prompt, one backward pass with cotangent e_d placed at *every* position t' gives
sum_{t'} d h_final,t'[d] / d h_l,t for all t and all layers at once (causality zeroes t' < t);
summing over t yields row d of the per-prompt sum. Pairs are pooled across prompts, so
J_l = (sum over prompts of the row sums) / (sum over prompts of T(T+1)/2).

Layers: index l = residual after block l-1 (0 = embeddings, n_layers = h_final, where J = I).

Usage: python fit_jlens.py --n 1000 --out artifacts/jlens_n1000.pt [--per-prompt artifacts/jlens_pp.pt]
"""

import argparse
import time

import pandas as pd
import torch

from common import ART, BPE, VAL_PARQUET, Runner, load_model


def prompt_jacobian_sums(R, ids, chunk):
    """Return [n_layers+1, d, d] tensor S with S[l][i, j] = sum_{t<=t'} d h_final,t'[i] / d h_l,t[j]."""
    d, L = R.d, ids.shape[1]
    dev = ids.device
    out = torch.zeros(R.n_layers + 1, d, d, device=dev)
    for c0 in range(0, d, chunk):
        c = min(chunk, d - c0)
        h0 = R.embed(ids).expand(c, L, d).detach().clone().requires_grad_(True)
        hs, h = [h0], h0
        for blk in R.dec.transformer_blocks:
            h = blk(h)
            hs.append(h)
        go = torch.zeros(c, L, d, device=dev)
        go[torch.arange(c), :, c0 + torch.arange(c)] = 1.0
        grads = torch.autograd.grad(h, hs[:-1], grad_outputs=go)
        for l, g in enumerate(grads):
            out[l, c0:c0 + c] = g.sum(1)
    out[R.n_layers] = torch.eye(d, device=dev) * L  # h_final,t' depends on h_final,t only for t' = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--out", default=str(ART / "jlens_n1000.pt"))
    ap.add_argument("--checkpoints", default="100,200,1000", help="also store J at these n")
    ap.add_argument("--per-prompt", default=None, help="store per-prompt sums (fp16) here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=22)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    tok = BPE()
    R = Runner(load_model(args.device))
    for p in R.dec.parameters():
        p.requires_grad_(False)
    texts = pd.read_parquet(VAL_PARQUET).text.tolist()[: args.n]
    ckpts = {int(x) for x in args.checkpoints.split(",") if x}

    S = torch.zeros(R.n_layers + 1, R.d, R.d, dtype=torch.float64, device=args.device)
    n_pairs, saved, per_prompt, t0 = 0, {}, [], time.time()
    for i, text in enumerate(texts):
        ids = torch.tensor([tok.encode(text, bos=True)[: args.seq_len]], device=args.device)
        L = ids.shape[1]
        s = prompt_jacobian_sums(R, ids, args.chunk)
        S += s.double()
        n_pairs += L * (L + 1) // 2
        if args.per_prompt:
            per_prompt.append(((s / (L * (L + 1) / 2)).half().cpu(), L))
        if i + 1 in ckpts:
            saved[i + 1] = (S / n_pairs).float().cpu()
        if (i + 1) % 10 == 0 or i == 0:
            el = time.time() - t0
            print(f"{i + 1}/{len(texts)}  {el:.0f}s  eta {el / (i + 1) * (len(texts) - i - 1):.0f}s", flush=True)
    saved[len(texts)] = (S / n_pairs).float().cpu()
    torch.save({"J": saved[len(texts)], "J_at_n": saved, "n": len(texts), "seq_len": args.seq_len,
                "layer_convention": "index l = residual after block l-1; 0 = embeddings; last = h_final"}, args.out)
    if args.per_prompt:
        torch.save({"J": torch.stack([j for j, _ in per_prompt]), "L": torch.tensor([L for _, L in per_prompt])},
                   args.per_prompt)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
