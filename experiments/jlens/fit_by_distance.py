"""Distance-resolved J: E[d h_final,t' / d h_l,t] bucketed by d = t' - t.

One backward pass with a one-hot cotangent on h_final at a single position t' gives
d h_final,t' / d h_l,s for every s <= t' at once, i.e. one Jacobian sample for every
distance d = t' - s. Sequences are story-aligned (<bos> + story, as in training) and
truncated at t' + 1 (causality makes later tokens irrelevant). With t' ~ U[1, T-1]
the distance distribution matches the all-pairs weighting used by fit.py.

Saves per-bucket sums/counts for two halves (split-half reliability); lenses are
assembled in exp6_future.py:  J_same = bucket d=0,  J_fut = all d >= 1,  J_all = all d.
"""
import argparse
import time

import torch

from core import ROOT, ByteBPE, load_model, residuals

BUCKETS = [(0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 10**9)]


def stories(path, line_range=(None, None)):
    """Stories are separated by a single newline, paragraphs inside a story by a blank line."""
    lines = path.read_text().splitlines()[slice(*line_range)]
    out, cur, prev_blank = [], [], True
    for ln in lines:
        if ln.strip() and not prev_blank and cur:
            out.append("\n".join(cur))
            cur = []
        cur.append(ln)
        prev_blank = not ln.strip()
    if cur:
        out.append("\n".join(cur))
    return [s.strip() for s in out if s.strip()]


def story_ids(tk, texts, max_len):
    return [torch.tensor(tk.encode(s)[:max_len]) for s in texts]


def bucket_of(d):
    return next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= d <= hi)


def fit(dec, seqs, n_batches, B, T, seed, chunk=64, log=print, save_cb=None):
    D, L = dec.d_model, len(dec.transformer_blocks)
    nb = len(BUCKETS)
    J = torch.zeros(2, nb, L + 1, D, D, dtype=torch.float64)   # [half, bucket, layer, D, D]
    cnt = torch.zeros(2, nb, dtype=torch.float64)
    bidx = torch.tensor([bucket_of(d) for d in range(T)])
    g = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for bi in range(n_batches):
        half = bi % 2
        tp = int(torch.randint(1, T, (1,), generator=g))
        pool = [s for s in seqs if len(s) > tp]
        pick = torch.randperm(len(pool), generator=g)[:B]
        seq = torch.stack([pool[i][:tp + 1] for i in pick])     # [B, tp+1]
        d_of_s = tp - torch.arange(tp + 1)                      # distance for each source position
        b_of_s = bidx[d_of_s]
        for start in range(0, D, chunk):
            k = min(chunk, D - start)
            hs, _ = residuals(dec, seq.repeat(k, 1), grad=True)
            for h in hs[1:]:
                h.retain_grad()
            out = hs[-1][:, -1].view(k, B, D)
            cot = torch.zeros_like(out)
            cot[torch.arange(k), :, start + torch.arange(k)] = 1.0
            (out * cot).sum().backward()
            for l, h in enumerate(hs):
                gr = h.grad.view(k, B, tp + 1, D).sum(1).double()   # [k, S, D] summed over batch
                for b in range(nb):
                    m = b_of_s == b
                    if m.any():
                        J[half, b, l, start:start + k] += gr[:, m].sum(1)
        for b in range(nb):
            cnt[half, b] += B * int((b_of_s == b).sum())
        log(f"batch {bi + 1}/{n_batches}  t'={tp}  {time.time() - t0:.0f}s")
        if save_cb and ((bi + 1) % 10 == 0 or bi + 1 == n_batches):
            save_cb(J, cnt, bi + 1)
    return J, cnt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", type=int, default=120)
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    tk = ByteBPE(args.vocab)
    dec = load_model(args.ckpt, len(tk.token_to_id))
    # J is a property of the model; fit on train-split stories, evaluate on held-out val stories
    seqs = story_ids(tk, stories(ROOT / "data/tinystory/processed/train.txt")[:4000], args.T)

    def save(J, cnt, n):
        torch.save({"J": J.float(), "count": cnt, "buckets": BUCKETS, "batches_done": n,
                    "B": args.B, "T": args.T}, args.out)
        print("saved", args.out, "after", n, "batches", flush=True)

    fit(dec, seqs, args.batches, args.B, args.T, seed=0, log=lambda m: print(m, flush=True), save_cb=save)
