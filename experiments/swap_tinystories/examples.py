"""Greedy continuations before/after the swap for 5 successful and 5 failed test trials of (C2, J)."""

import torch

from common import ART, DATA, jdump, jload
from swap import Swapper


@torch.no_grad()
def generate(S, tr, cond, lens, band, alpha, n_new=8):
    ids = S.tok.encode(tr["prompt"], bos=True)
    P = len(ids)
    edit = {}
    if cond is not None:
        for l in band:
            D = S.dirs[(cond, lens)][l]
            V = torch.stack([D[tr["s"]], D[tr["t"]]], -1)
            Pinv = torch.linalg.pinv(V)

            def f(h, V=V, Pinv=Pinv):
                c = h @ Pinv.T
                delta = alpha * (c.flip(-1) - c) @ V.T
                delta[:, P:] = 0  # prompt positions only
                return h + delta
            edit[l] = f
    for _ in range(n_new):
        logits = S.R.forward(torch.tensor([ids], device=S.dev), edit=edit)
        ids.append(int(logits[0, -1].argmax()))
    return S.tok.decode(ids[P:])


@torch.no_grad()
def target_logprob(S, tr, cond, lens, band, alpha):
    """log p(gold answer of t | prompt) under the (optional) swap."""
    one = dict(tr)
    res, _, _ = S.run([one], cond, lens, band, alpha) if cond else S.run([one], None, None, (), 0)
    j = S.animals.index(tr["t"])
    return float(res["cand_lp"][0][j])


@torch.no_grad()
def clean_stats(S):
    """How strongly does the clean model want to make the sound on the kept prompts?"""
    recs = jload(DATA / "dev.json") + jload(DATA / "test.json")
    top1_is_gold, p_gold = 0, []
    for r in recs:
        ids = S.tok.encode(r["prompt"], bos=True)
        lp = torch.log_softmax(S.R.forward(torch.tensor([ids], device=S.dev))[0, -1], -1)
        top1_is_gold += int(lp.argmax()) == S.first_tok[r["animal"]]
        p_gold.append(float(lp[S.first_tok[r["animal"]]].exp()))
    return dict(n_prompts=len(recs), top1_is_gold_first_token=top1_is_gold, median_p_gold_first_token=float(torch.tensor(p_gold).median()),
                max_p_gold_first_token=max(p_gold))


def spread(idx, trials, n=5):
    """Pick n trials round-robin over source animals (first available, deterministic)."""
    by_src = {}
    for i in idx:
        by_src.setdefault(trials[i]["s"], []).append(i)
    out = []
    while len(out) < n and any(by_src.values()):
        for s in list(by_src):
            if by_src[s] and len(out) < n:
                out.append(by_src[s].pop(0))
    return out


def main(device="cpu", cond="C2", lens="J"):
    S = Swapper(device)
    sw = jload(ART / "swap_results.json")
    k = f"{cond}|{lens}"
    ch, res = sw["chosen"][k], sw["test"][k]
    band, alpha = tuple(ch["band"]), ch["alpha"]
    trials = sw["test_trials"]
    succ = spread([i for i, s in enumerate(res["success_cat"]) if s], trials)
    fail = spread([i for i, s in enumerate(res["success_cat"]) if not s], trials)
    clean = sw["clean_test"]

    def pack(i):
        tr = trials[i]
        lp0 = target_logprob(S, tr, None, None, (), 0)
        lp1 = target_logprob(S, tr, cond, lens, band, alpha)
        return dict(s=tr["s"], t=tr["t"], prompt=tr["prompt"], before=generate(S, tr, None, None, (), 0),
                    after=generate(S, tr, cond, lens, band, alpha), pred_before=clean["pred"][i], pred_after=res["pred"][i],
                    p_target_before=float(torch.tensor(lp0).exp()), p_target_after=float(torch.tensor(lp1).exp()))
    jdump(dict(condition=f"{cond}, {lens}-lens, band {'-'.join(map(str, band))}, α={alpha:g}",
               success=[pack(i) for i in succ], fail=[pack(i) for i in fail], clean_stats=clean_stats(S)), ART / "examples.json")


if __name__ == "__main__":
    main()
