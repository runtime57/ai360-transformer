"""Stages 6–8: clamped two-direction swaps s→t, dev grid search, held-out test evaluation.

h' = h + α · V (swap(c) − c),  V = [u_s u_t],  c = pinv(V) h, applied at every prompt position
on every layer of the band (hook after each block in the band, so edits compound = "clamped").
"""

import itertools
import time

import numpy as np
import torch

from common import ART, BPE, DATA, Runner, jdump, jload, load_model, pad_batch
from lens import dictionary, load_J

ALPHAS = [0.5, 1, 2, 4]
LENS_CONDS = ["C1", "C2", "C3", "C5", "C6"]  # lens-dependent conditions; C4 is lens-free
EXCLUDED_PAIRS = {("frog", "lion"), ("lion", "frog")}  # identical first answer token 'R'


def bands(n_layers):
    out = []
    for w in range(1, 5):
        for start in range(1, n_layers - w + 2):
            out.append(tuple(range(start, start + w)))
    return out


class Swapper:
    def __init__(self, device):
        self.dev = device
        self.tok = BPE()
        self.R = Runner(load_model(device))
        self.J = load_J(device)
        meta = jload(DATA / "meta.json")
        self.animals = meta["animals"]
        self.gold = meta["gold"]
        self.cand_ids = [self.tok.encode(self.gold[a]) for a in self.animals]
        self.first_tok = {a: self.tok.encode(self.gold[a])[0] for a in self.animals}
        self.animal_tok = {a: self.tok.encode(" " + a)[0] for a in self.animals}
        store = torch.load(ART / "probes_decomp.pt")
        L_all = range(1, self.R.n_layers + 1)
        unit = lambda v: (v / v.norm()).to(device)
        # dirs[(cond, lens)][layer][animal] -> unit vector
        self.dirs = {}
        for lens in ["J", "logit"]:
            D = {l: dictionary(self.R, self.J, lens, l) for l in L_all}
            self.dirs[("C1", lens)] = {l: {a: D[l][self.animal_tok[a]] for a in self.animals} for l in L_all}
            self.dirs[("C6", lens)] = {l: {a: D[l][self.first_tok[a]] for a in self.animals} for l in L_all}
            self.dirs[("C2", lens)] = {l: {a: unit(store["c"][lens][(a, l)]) for a in self.animals} for l in L_all}
            self.dirs[("C3", lens)] = {l: {a: unit(store["r"][lens][(a, l)]) for a in self.animals} for l in L_all}
        self.dirs[("C4", None)] = {l: {a: unit(store["p"][(a, l)]) for a in self.animals} for l in L_all}
        self.dirs[("C5", "J")] = self.dirs[("C2", "J")]  # C5 borrows ‖Δh‖ from C2 of the same lens
        self.dirs[("C5", "logit")] = self.dirs[("C2", "logit")]

    # ---------- trials
    def make_trials(self, recs):
        trials = []
        for r in recs:
            for t in self.animals:
                if t != r["animal"] and (r["animal"], t) not in EXCLUDED_PAIRS:
                    trials.append(dict(s=r["animal"], t=t, prompt=r["prompt"], idx=r["idx"]))
        return trials

    def batch(self, trials):
        rows, P = [], []
        for tr in trials:
            p = self.tok.encode(tr["prompt"], bos=True)
            P.append(len(p))
            rows += [p + c for c in self.cand_ids]
        seq = pad_batch(rows, self.tok.pad).to(self.dev)
        P = torch.tensor(P, device=self.dev).repeat_interleave(len(self.cand_ids))
        mask = (torch.arange(seq.shape[1], device=self.dev)[None] < P[:, None]).float()
        return seq, P, mask

    # ---------- one forward with edits
    @torch.no_grad()
    def run(self, trials, cond, lens, band, alpha, beta=None, ref_norms=None, seed=0):
        """Returns per-trial metrics and per-row Δh norms [B, len(band), L] (prompt positions)."""
        nc = len(self.cand_ids)
        seq, P, mask = self.batch(trials)
        B, Lseq = seq.shape
        norms = torch.zeros(B, len(band), Lseq, device=self.dev)
        edit = {}
        if cond is not None:
            for bi, l in enumerate(band):
                if cond == "C5":
                    g = torch.Generator(device="cpu").manual_seed(seed * 1000 + l)
                    rnd = torch.randn(len(trials), Lseq, self.R.d, generator=g).to(self.dev)
                    rnd = (rnd / rnd.norm(dim=-1, keepdim=True)).repeat_interleave(nc, 0)
                    delta = rnd * ref_norms[:, bi, :, None] * mask[..., None]
                    edit[l] = (lambda d: (lambda h: h + d))(delta)
                    norms[:, bi] = ref_norms[:, bi]
                    continue
                D = self.dirs[(cond, lens)][l]
                V = torch.stack([torch.stack([D[tr["s"]], D[tr["t"]]], -1) for tr in trials]).repeat_interleave(nc, 0)
                Pinv = torch.linalg.pinv(V)  # [B, 2, d]
                scale = alpha * (beta.repeat_interleave(nc) if beta is not None else torch.ones(len(trials), device=self.dev).repeat_interleave(nc))

                def f(h, V=V, Pinv=Pinv, scale=scale, bi=bi):
                    c = torch.einsum("bkd,bld->blk", Pinv, h)
                    delta = torch.einsum("bdk,blk->bld", V, c.flip(-1) - c) * (scale[:, None, None] * mask[..., None])
                    norms[:, bi] = delta.norm(dim=-1)
                    return h + delta
                edit[l] = f
        logits = self.R.forward(seq, edit=edit)
        lp = torch.log_softmax(logits.float(), -1)
        # candidate log-probs (teacher forcing), row j of trial i scores candidate j
        cand_lp = torch.zeros(B, device=self.dev)
        for j, ids in enumerate(self.cand_ids):
            rows = torch.arange(j, B, nc, device=self.dev)
            for k, tid in enumerate(ids):
                cand_lp[rows] += lp[rows, P[rows] - 1 + k, tid]
        cand_lp = cand_lp.view(len(trials), nc)
        top1 = lp[torch.arange(0, B, nc, device=self.dev), P[::nc] - 1].argmax(-1)
        ti = torch.tensor([self.animals.index(tr["t"]) for tr in trials], device=self.dev)
        si = torch.tensor([self.animals.index(tr["s"]) for tr in trials], device=self.dev)
        other = cand_lp.clone()
        other[torch.arange(len(trials)), ti] = -np.inf
        res = dict(
            success_cat=(cand_lp.argmax(-1) == ti).cpu().numpy(),
            success_top1=(top1 == torch.tensor([self.first_tok[tr["t"]] for tr in trials], device=self.dev)).cpu().numpy(),
            margin=(cand_lp[torch.arange(len(trials)), ti] - other.max(-1).values).cpu().numpy(),
            still_source=(cand_lp.argmax(-1) == si).cpu().numpy(),
            pred=[self.animals[i] for i in cand_lp.argmax(-1).tolist()],
            cand_lp=cand_lp.cpu().numpy(),
        )
        Pn = P[::nc]
        mean_norm = torch.stack([norms[::nc][i, :, : Pn[i]].mean() for i in range(len(trials))])
        return res, norms[::nc].repeat_interleave(nc, 0), mean_norm

    def run_condition(self, trials, cond, lens, band, alpha):
        """Run one (condition, lens, band, α) setting, resolving C3/C5 dependencies on C2."""
        if cond in ("C3", "C5"):
            res2, norms2, mean2 = self.run(trials, "C2", lens, band, alpha)
            if cond == "C5":
                res, _, mean = self.run(trials, "C5", lens, band, alpha, ref_norms=norms2, seed=hash(band) % 9973)
                res["norm_ratio_to_C2"] = (mean / mean2).cpu().numpy()
                return res
            # edits compound across band layers (‖Δh‖ can grow exponentially in β), so solve
            # log mean‖Δh_C3‖(β) = log mean‖Δh_C2‖ per trial with secant steps in log β
            tgt = mean2.clamp_min(1e-12).log()
            x0 = torch.zeros_like(tgt)
            y0 = self.run(trials, "C3", lens, band, alpha, beta=x0.exp())[2].clamp_min(1e-12).log() - tgt
            x1 = x0 - y0
            for _ in range(8):
                y1 = self.run(trials, "C3", lens, band, alpha, beta=x1.exp())[2].clamp_min(1e-12).log() - tgt
                if y1.abs().max() < 0.01:
                    break
                slope = ((y1 - y0) / (x1 - x0).where((x1 - x0).abs() > 1e-9, torch.ones_like(x1))).clamp(0.2, 20)
                x0, y0, x1 = x1, y1, x1 - y1 / slope
            res, _, mean = self.run(trials, "C3", lens, band, alpha, beta=x1.exp())
            res["norm_ratio_to_C2"] = (mean / mean2).cpu().numpy()
            return res
        res, _, mean = self.run(trials, cond, lens, band, alpha)
        res["mean_dh_norm"] = mean.cpu().numpy()
        return res


def settings():
    out = [(c, L) for L in ["J", "logit"] for c in LENS_CONDS]
    return out + [("C4", None)]


def summarize(res):
    return {k: float(np.mean(v)) for k, v in res.items() if k not in ("pred", "cand_lp")}


def main(device):
    t0 = time.time()
    S = Swapper(device)
    dev_trials = S.make_trials(jload(DATA / "dev.json"))
    test_trials = S.make_trials(jload(DATA / "test.json"))
    print(f"dev trials {len(dev_trials)}, test trials {len(test_trials)}")
    clean_dev, _, _ = S.run(dev_trials, None, None, (), 0)
    clean_test, _, _ = S.run(test_trials, None, None, (), 0)
    print("clean success@cat dev/test (should be 0):", clean_dev["success_cat"].mean(), clean_test["success_cat"].mean())

    grid = {}
    all_bands = bands(S.R.n_layers)
    for (cond, lens), band, alpha in itertools.product(settings(), all_bands, ALPHAS):
        res = S.run_condition(dev_trials, cond, lens, band, alpha)
        grid[f"{cond}|{lens}|{'-'.join(map(str, band))}|{alpha}"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res.items()}
    print(f"dev grid done: {len(grid)} settings, {time.time() - t0:.0f}s")

    chosen, test = {}, {}
    for cond, lens in settings():
        keys = [k for k in grid if k.startswith(f"{cond}|{lens}|")]
        best = max(keys, key=lambda k: (np.mean(grid[k]["success_cat"]), np.mean(grid[k]["margin"])))
        _, _, b, a = best.split("|")
        band, alpha = tuple(int(x) for x in b.split("-")), float(a)
        chosen[f"{cond}|{lens}"] = dict(band=band, alpha=alpha, dev=summarize({k: np.array(v) for k, v in grid[best].items() if k != "pred"}))
        res = S.run_condition(test_trials, cond, lens, band, alpha)
        test[f"{cond}|{lens}"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res.items()}
        print(f"{cond:3s} {str(lens):5s} band={band} α={alpha}: dev@cat={chosen[f'{cond}|{lens}']['dev']['success_cat']:.3f} "
              f"test@cat={np.mean(res['success_cat']):.3f} test@top1={np.mean(res['success_top1']):.3f}")

    jdump(dict(dev_trials=dev_trials, test_trials=test_trials, alphas=ALPHAS, bands=all_bands,
               clean_dev={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in clean_dev.items()},
               clean_test={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in clean_test.items()},
               grid=grid, chosen=chosen, test=test), ART / "swap_results.json")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import sys

    torch.set_num_threads(22)
    main(sys.argv[1] if len(sys.argv) > 1 else ("cuda" if torch.cuda.is_available() else "cpu"))
