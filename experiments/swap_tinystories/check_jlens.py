"""Verify fit_jlens against brute-force Jacobians (all layers, real text) and run stage-3 sanity checks."""

import sys

import pandas as pd
import torch

from common import ART, BPE, VAL_PARQUET, Runner, load_model
from fit_jlens import prompt_jacobian_sums


def main(jpath):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, R = BPE(), Runner(load_model(dev))
    for p in R.dec.parameters():
        p.requires_grad_(False)
    texts = pd.read_parquet(VAL_PARQUET).text.tolist()

    # 1) brute force on a real prompt, every layer: full [t', out, t, in] Jacobian
    L = 24
    ids = torch.tensor([tok.encode(texts[0], bos=True)[:L]], device=dev)
    S = prompt_jacobian_sums(R, ids, 512)
    with torch.no_grad():
        _, hs = R.forward(ids, return_resid=True)
    tp, t = torch.meshgrid(torch.arange(L), torch.arange(L), indexing="ij")
    print("brute-force check (real text, L=%d):" % L)
    for l in range(R.n_layers + 1):
        def f(x, l=l):
            h = x.unsqueeze(0)
            for b in R.dec.transformer_blocks[l:]:
                h = b(h)
            return h[0]
        Jf = torch.func.jacrev(f)(hs[l][0])  # [t', out, t, in]
        acausal = Jf.permute(0, 2, 1, 3)[tp < t].abs().max().item()
        ref = Jf.sum((0, 2))
        err = (S[l] - ref).abs().max().item()
        print(f"  layer {l}: max|fit - brute|={err:.2e}  (scale {ref.abs().max().item():.1f}), max acausal entry={acausal:.1e}")

    # 2) sanity checks on the fitted lens
    ck = torch.load(jpath)
    J = ck["J"].to(dev)
    I = torch.eye(R.d, device=dev)
    print("\nfitted J, n=%d:" % ck["n"])
    for l in range(R.n_layers + 1):
        rel = ((J[l] - I).norm() / I.norm()).item()
        cos = torch.nn.functional.cosine_similarity(J[l].flatten(), I.flatten(), 0).item()
        print(f"  layer {l}: ||J-I||/||I||={rel:.3f}  cos(J,I)={cos:.3f}")

    # readout of logit lens at the last layer == model output
    ids = torch.tensor([tok.encode(texts[1], bos=True)[:128]], device=dev)
    with torch.no_grad():
        logits, hs = R.forward(ids, return_resid=True)
        lens_last = R.unembed(hs[-1] @ J[-1].T)
        print("\nlast-layer readout vs model logits: max abs diff = %.2e" % (lens_last - logits).abs().max().item())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ART / "jlens_n1000.pt"))
