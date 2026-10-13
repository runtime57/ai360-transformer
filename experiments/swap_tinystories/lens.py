"""Lens dictionaries, readouts and nonnegative k-sparse decomposition.

Layer index l follows common.Runner: l = residual after block l-1 (1..n_layers); 0 = embeddings.
"""

import numpy as np
import torch

from common import ART

N_SPECIAL = 4  # <pad> <unk> <bos> <eos> are never dictionary atoms


def load_J(device="cpu"):
    return torch.load(ART / "jlens_n1000.pt")["J"].to(device)


def lens_matrix(J, lens, l):
    return J[l] if lens == "J" else torch.eye(J.shape[-1], device=J.device)


def readout(R, J, lens, l, h):
    """Lens logits W_U · ln_f(J_l h) (J = I for the logit lens)."""
    return R.unembed(h @ lens_matrix(J, lens, l).T)


def dictionary(R, J, lens, l):
    """Unit-norm token directions in h_l space: rows J_lᵀ (g ⊙ W_U[tok])."""
    base = R.W_U.detach() * R.g.detach()  # [V, d]
    D = base @ lens_matrix(J, lens, l)  # row tok: (J_lᵀ base_tok)ᵀ
    D = D / D.norm(dim=1, keepdim=True)
    D[:N_SPECIAL] = 0
    return D


def nnls(A, b, iters=200):
    """Lawson–Hanson NNLS: min ||A x - b||, x >= 0. A: [d, k] float64."""
    k = A.shape[1]
    x = np.zeros(k)
    P = np.zeros(k, bool)
    w = A.T @ (b - A @ x)
    for _ in range(iters):
        if P.all() or w[~P].max(initial=-np.inf) <= 1e-12:
            break
        j = np.where(~P)[0][np.argmax(w[~P])]
        P[j] = True
        while True:
            z = np.zeros(k)
            z[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if (z[P] > 0).all():
                x = z
                break
            neg = P & (z <= 0)
            alpha = np.min(x[neg] / (x[neg] - z[neg]))
            x = x + alpha * (z - x)
            P &= x > 1e-12
            x[~P] = 0
        w = A.T @ (b - A @ x)
    return x


def nn_omp(D, p, k=16):
    """Nonnegative OMP: greedily add the atom most positively correlated with the residual,
    refit all coefficients with NNLS. Returns (atom ids, coefs, approximation c, residual r)."""
    Dn = D.double().cpu().numpy()
    pn = p.double().cpu().numpy()
    S, coef, r = [], np.zeros(0), pn.copy()
    for _ in range(k):
        corr = Dn @ r
        corr[S] = -np.inf
        j = int(np.argmax(corr))
        if corr[j] <= 0:
            break
        S.append(j)
        coef = nnls(Dn[S].T, pn)
        r = pn - Dn[S].T @ coef
    keep = coef > 0
    S, coef = [s for s, kk in zip(S, keep) if kk], coef[keep]
    c = Dn[S].T @ coef if S else np.zeros_like(pn)
    order = np.argsort(-coef)
    return ([S[i] for i in order], coef[order], torch.tensor(c, dtype=p.dtype, device=p.device),
            torch.tensor(pn - c, dtype=p.dtype, device=p.device))
