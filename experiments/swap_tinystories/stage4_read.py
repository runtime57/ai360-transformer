"""Stage 4: reading the unspoken animal with both lenses, plus the §4.1 lens-quality metrics."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from common import ART, BPE, DATA, FIG, VAL_PARQUET, Runner, jdump, jload, load_model
from lens import load_J, readout

LENSES = ["J", "logit"]
KS = [1, 5, 10, 16]


def rank_of(logits, tok_id):
    return int((logits > logits[tok_id]).sum())


@torch.no_grad()
def main(device="cpu", n_corpus=200):
    torch.manual_seed(0)
    tok, R = BPE(), Runner(load_model(device))
    J = load_J(device)
    meta = jload(DATA / "meta.json")
    animal_id = {a: tok.encode(" " + a)[0] for a in meta["animals"]}
    layers = list(range(R.n_layers + 1))
    test = jload(DATA / "test.json")

    # --- ranks of the animal token at the last prompt position
    ranks = {L: np.zeros((len(test), len(layers)), int) for L in LENSES}
    for i, rec in enumerate(test):
        ids = torch.tensor([tok.encode(rec["prompt"], bos=True)], device=device)
        _, hs = R.forward(ids, return_resid=True)
        for L in LENSES:
            for l in layers:
                ranks[L][i, l] = rank_of(readout(R, J, L, l, hs[l][0, -1]), animal_id[rec["animal"]])

    # --- corpus statistics on held-out val texts (J was fit on the first 1000)
    texts = pd.read_parquet(VAL_PARQUET).text.tolist()[1000:1000 + n_corpus]
    sum_h = [torch.zeros(R.d, device=device) for _ in layers]
    n_pos = 0
    kurt = {L: np.zeros(len(layers)) for L in LENSES}
    top10 = {L: np.zeros(len(layers)) for L in LENSES}
    auto = {L: np.zeros(len(layers)) for L in LENSES}
    auto_null = {L: np.zeros(len(layers)) for L in LENSES}
    n_adj = 0
    gen = torch.Generator().manual_seed(0)
    for text in texts:
        ids = torch.tensor([tok.encode(text, bos=True)[:128]], device=device)
        logits, hs = R.forward(ids, return_resid=True)
        model_top1 = logits[0, 1:].argmax(-1)  # skip <bos> position
        T = model_top1.shape[0]
        n_pos += T
        n_adj += T - 1
        perms = [torch.randperm(T, generator=gen) for _ in range(10)]
        for l in layers:
            h = hs[l][0, 1:]
            sum_h[l] += h.sum(0)
            for L in LENSES:
                lg = readout(R, J, L, l, h)
                z = (lg - lg.mean(-1, keepdim=True)) / lg.std(-1, keepdim=True)
                kurt[L][l] += ((z ** 4).mean(-1) - 3).sum().item()
                top10[L][l] += (lg.topk(10, -1).indices == model_top1[:, None]).any(-1).sum().item()
                t1 = lg.argmax(-1)
                auto[L][l] += (t1[1:] == t1[:-1]).sum().item()
                auto_null[L][l] += np.mean([(t1[p][1:] == t1[p][:-1]).sum().item() for p in perms])
    mean_h = [s / n_pos for s in sum_h]
    freq_rank = {L: {a: [rank_of(readout(R, J, L, l, mean_h[l]), animal_id[a]) for l in layers] for a in animal_id}
                 for L in LENSES}

    out = dict(layers=layers, n_test_prompts=len(test))
    for L in LENSES:
        rk = ranks[L]
        mn = rk.min(1)
        out[L] = dict(
            median_rank_by_layer=np.median(rk, 0).tolist(),
            min_rank_per_prompt=mn.tolist(),
            argmin_layer_per_prompt=rk.argmin(1).tolist(),
            pass_at_k_minrank={k: float((mn < k).mean()) for k in KS},
            pass_at_k_by_layer={k: (rk < k).mean(0).tolist() for k in KS},
            freq_control_rank_by_layer=freq_rank[L],
            freq_control_median_by_layer=np.median([freq_rank[L][r["animal"]] for r in test], 0).tolist(),
            excess_kurtosis=(kurt[L] / n_pos).tolist(),
            top10_contains_model_top1=(top10[L] / n_pos).tolist(),
            top1_autocorr=(auto[L] / n_adj).tolist(),
            top1_autocorr_shuffled=(auto_null[L] / n_adj).tolist(),
        )
    out["per_prompt"] = [dict(animal=r["animal"], prompt=r["prompt"], J_ranks=ranks["J"][i].tolist(),
                              logit_ranks=ranks["logit"][i].tolist()) for i, r in enumerate(test)]
    jdump(out, ART / "stage4.json")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for L, c in zip(LENSES, ["C0", "C1"]):
        ax[0].plot(layers, out[L]["median_rank_by_layer"], "o-", c=c, label=f"{L}-lens: test prompts")
        ax[0].plot(layers, out[L]["freq_control_median_by_layer"], "x--", c=c, alpha=.6, label=f"{L}-lens: mean-residual control")
    ax[0].set_yscale("symlog")
    ax[0].set_xlabel("layer (0 = embeddings, l = after block l-1)")
    ax[0].set_ylabel("median rank of animal token (0 = top)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Reading the unnamed animal")
    for L, c in zip(LENSES, ["C0", "C1"]):
        ax[1].plot(layers, [out[L]["pass_at_k_by_layer"][10][l] for l in layers], "o-", c=c, label=f"{L}: pass@10")
        ax[1].plot(layers, [out[L]["pass_at_k_by_layer"][1][l] for l in layers], "s:", c=c, label=f"{L}: pass@1")
    ax[1].set_xlabel("layer")
    ax[1].set_ylabel("fraction of test prompts")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "stage4_ranks.png", dpi=130)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for L, c in zip(LENSES, ["C0", "C1"]):
        ax[0].plot(layers, out[L]["excess_kurtosis"], "o-", c=c, label=L)
        ax[1].plot(layers, out[L]["top10_contains_model_top1"], "o-", c=c, label=L)
        ax[2].plot(layers, out[L]["top1_autocorr"], "o-", c=c, label=f"{L}")
        ax[2].plot(layers, out[L]["top1_autocorr_shuffled"], "x--", c=c, alpha=.6, label=f"{L} shuffled null")
    for a, t in zip(ax, ["excess kurtosis of readout", "model top-1 in lens top-10", "top-1 same at t, t+1"]):
        a.set_title(t)
        a.set_xlabel("layer")
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "stage4_quality.png", dpi=130)

    for L in LENSES:
        o = out[L]
        print(f"{L}: median rank by layer {o['median_rank_by_layer']}; pass@k(min over layers) {o['pass_at_k_minrank']}")
        print(f"   freq-control median {o['freq_control_median_by_layer']}")
        print(f"   kurtosis {np.round(o['excess_kurtosis'], 1).tolist()}  top10∋top1 {np.round(o['top10_contains_model_top1'], 3).tolist()}")
        print(f"   autocorr {np.round(o['top1_autocorr'], 3).tolist()}  null {np.round(o['top1_autocorr_shuffled'], 3).tolist()}")


if __name__ == "__main__":
    import sys

    torch.set_num_threads(22)
    main(sys.argv[1] if len(sys.argv) > 1 else "cpu")
