"""Fit J-lens matrices on TinyStories validation text and save them."""
import argparse
import time

import torch

from core import ROOT, ByteBPE, fit_jacobians, load_model


def corpus_windows(tk, path, n, T, seed, line_range):
    lines = path.read_text().splitlines()[slice(*line_range)]
    stream = tk.encode("\n".join(lines), bos=False)
    g = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, len(stream) - T, (n,), generator=g)
    return torch.stack([torch.tensor(stream[s:s + T]) for s in starts])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    tk = ByteBPE(args.vocab)
    dec = load_model(args.ckpt, len(tk.token_to_id))
    data = corpus_windows(tk, ROOT / "data/tinystory/processed/val.txt", args.n, args.T, 0, (0, 1200))
    batches = list(data.split(args.batch))
    t0 = time.time()
    # two halves fitted separately -> split-half reliability; the full J is their mean
    half = len(batches) // 2
    JA = fit_jacobians(dec, batches[:half], log=lambda m: print("A", m, f"{time.time() - t0:.0f}s", flush=True))
    JB = fit_jacobians(dec, batches[half:], log=lambda m: print("B", m, f"{time.time() - t0:.0f}s", flush=True))
    torch.save({"J": (JA + JB) / 2, "JA": JA, "JB": JB, "n": args.n, "T": args.T}, args.out)
    print("saved", args.out)
