"""Five J-lens experiments on the TinyStories GPT. Writes figures/*.png and results.json."""
import argparse
import json
from pathlib import Path

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from core import ROOT, ByteBPE, lens_logits, load_model, residuals  # noqa: E402
from fit import corpus_windows  # noqa: E402

# ---------------------------------------------------------------- style
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
C_LOGIT, C_J = "#2a78d6", "#eb6834"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)
LENSES = [("logit lens", C_LOGIT), ("J-lens", C_J)]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10.5, "text.color": INK,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.spines.left": False, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6.5,
})


def layer_names(n):
    return ["emb"] + [f"L{i}" for i in range(1, n)]


def suptitle(fig, title, subtitle):
    fig.text(0.012, 0.985, title, fontsize=15, fontweight="bold", va="top", color=INK)
    fig.text(0.012, 0.935, subtitle, fontsize=10.5, va="top", color=INK2)


def save(fig, out, name):
    fig.savefig(out / name, dpi=170, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", out / name)


def all_lens_logits(dec, hs, J):
    """-> {lens_name: [L+1, B, T, V]}"""
    return {
        "logit lens": torch.stack([lens_logits(dec, h) for h in hs]),
        "J-lens": torch.stack([lens_logits(dec, h, J[l]) for l, h in enumerate(hs)]),
    }


# ---------------------------------------------------------------- E1 faithfulness
def exp1_faithfulness(dec, data, J, JA, JB, out):
    hs, final = residuals(dec, data)
    logp_final = final.log_softmax(-1)
    top_final = final.argmax(-1)
    res = {}
    for name, LG in all_lens_logits(dec, hs, J).items():
        lp = LG.log_softmax(-1)
        kl = (logp_final.exp() * (logp_final - lp)).sum(-1).mean((1, 2))
        agree = (LG.argmax(-1) == top_final).float().mean((1, 2))
        res[name] = {"kl": kl.tolist(), "top1_agree": agree.tolist()}
    cos = [F.cosine_similarity(a.flatten(), b.flatten(), 0).item() for a, b in zip(JA, JB)]
    res["split_half_cos"] = cos

    names = layer_names(len(hs))
    x = range(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.15, 1.15, 0.8]})
    for (name, c) in LENSES:
        axes[0].plot(list(x)[:-1], res[name]["kl"][:-1], "-o", color=c, label=name)
        axes[1].plot(x, [100 * v for v in res[name]["top1_agree"]], "-o", color=c, label=name)
    axes[0].set_yscale("log")
    axes[0].set_title("KL(final ‖ lens), nats · lower = better")
    axes[1].set_title("Top-1 agrees with final output, %")
    axes[1].set_ylim(0, 105)
    for ax in axes[:2]:
        ax.legend(loc="upper right" if ax is axes[0] else "lower right")
    axes[2].bar(x, cos, color=C_J, width=0.6)
    axes[2].set_ylim(0.98, 1.0)
    axes[2].set_title("J stability: cos(J half A, J half B)")
    for ax in axes:
        ax.set_xticks(list(x), names)
        ax.set_xlabel("layer")
    suptitle(fig, "Experiment 1 · How faithful is each lens?",
             f"{data.numel():,} held-out TinyStories tokens. J-lens maps each layer through the average Jacobian "
             "to the final residual before unembedding.")
    fig.subplots_adjust(top=0.78, wspace=0.28)
    save(fig, out, "exp1_faithfulness.png")
    return res


# ---------------------------------------------------------------- E2 thought grid
def exp2_thought_grid(dec, tk, J, out, text, fname, title):
    ids = torch.tensor([tk.encode(text)])
    hs, _ = residuals(dec, ids)
    toks = [tk.tok(i) for i in ids[0]]
    LGs = all_lens_logits(dec, hs, J)
    n_layers, T = len(hs), ids.shape[1]
    fig, axes = plt.subplots(2, 1, figsize=(max(12, 1.02 * T), 2 * (0.55 * n_layers + 1.4)))
    res = {"tokens": toks}
    for ax, (name, c) in zip(axes, LENSES):
        P = LGs[name][:, 0].softmax(-1)
        p, top = P.max(-1)
        ax.imshow(p.numpy()[::-1], cmap=CMAP, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        grid_txt = []
        for li in range(n_layers):
            row = []
            for t in range(T):
                s = tk.tok(top[li, t]).strip() or repr(tk.tok(top[li, t]))
                row.append(s)
                v = p[li, t].item()
                ax.text(t, n_layers - 1 - li, s[:9], ha="center", va="center", fontsize=8.6,
                        color="white" if v > 0.45 else INK, fontweight="bold" if li == n_layers - 1 else None)
            grid_txt.append(row)
        res[name] = grid_txt
        ax.set_yticks(range(n_layers), layer_names(n_layers)[::-1])
        ax.set_xticks(range(T), [t.strip() or "␣" for t in toks], rotation=45, ha="right", fontsize=9)
        ax.tick_params(length=0)
        ax.set_title(f"{name}  ·  top-1 guess for the NEXT token (shade = probability)", color=c)
        for s in ax.spines.values():
            s.set_visible(False)
    cax = fig.add_axes([0.92, 0.94, 0.07, 0.012])
    fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP), cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cax.tick_params(labelsize=8)
    suptitle(fig, title, "Each column is an input position; each row is what that layer 'thinks' comes next. "
             "With tied embeddings, the logit lens on early layers mostly echoes the current token.")
    fig.subplots_adjust(top=0.86, hspace=0.55)
    save(fig, out, fname)
    return res


# ---------------------------------------------------------------- E3 answer trajectories
CLOZE = [
    ("Lily had a red ball. She threw the ball to her dog. The dog ran and brought back the", " ball"),
    ("Tom and his mom went to the park. When it was time to go home, Tom held hands with his", " mom"),
    ("Once upon a time, there was a little girl named", " Lily"),
    ("Sam lost his toy and could not find it. He was very sad and started to", " cry"),
    ("Anna saw a big, scary dog. She was very", " scared"),
    ("The cat climbed up the tall tree but could not get down. The cat was stuck in the", " tree"),
]


def exp3_answer_trajectories(dec, tk, J, out):
    res = []
    for prompt, ans in CLOZE:
        ids = torch.tensor([tk.encode(prompt)])
        aid = tk.encode(ans, bos=False)
        assert len(aid) == 1, ans
        aid = aid[0]
        hs, final = residuals(dec, ids)
        row = {"prompt": prompt, "answer": ans, "final_top1": tk.tok(final[0, -1].argmax())}
        for name, LG in all_lens_logits(dec, hs, J).items():
            P = LG[:, 0, -1].softmax(-1)
            row[name] = {"p": P[:, aid].tolist(),
                         "rank": [(P[l] > P[l, aid]).sum().item() + 1 for l in range(len(hs))],
                         "top1": [tk.tok(P[l].argmax()) for l in range(len(hs))]}
        res.append(row)

    n = len(res)
    cols = 3
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(15, 4.1 * ((n + cols - 1) // cols)), sharey=True)
    names = layer_names(len(res[0]["J-lens"]["p"]))
    for ax, r in zip(axes.flat, res):
        for name, c in LENSES:
            ax.plot(names, r[name]["p"], "-o", color=c, label=name)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 1.5)
        short = r["prompt"] if len(r["prompt"]) < 58 else "…" + r["prompt"][-56:]
        ax.set_title(f"P(“{r['answer'].strip()}”)", fontsize=11.5, pad=20)
        ax.text(0, 1.025, short + " ___", transform=ax.transAxes, fontsize=8.6, color=INK2, va="bottom")
    axes.flat[0].legend(loc="lower right")
    for ax in axes[:, 0]:
        ax.set_ylabel("probability of the answer (log)")
    suptitle(fig, "Experiment 3 · When does the model 'know' the answer?",
             "Probability each lens assigns to the correct next word at the last position, layer by layer (values below 1e-6 run off the axis).")
    fig.subplots_adjust(top=0.82, hspace=0.62, wspace=0.08)
    save(fig, out, "exp3_answer_trajectories.png")
    return res


# ---------------------------------------------------------------- E4 lookahead
def exp4_lookahead(dec, data, J, out, K=(1, 2, 3), topk=10):
    hs, final = residuals(dec, data)
    T = data.shape[1]
    res = {}
    for name, LG in all_lens_logits(dec, hs, J).items():
        top = LG.topk(topk, -1).indices  # [L+1, B, T, k]
        res[name] = {}
        for k in K:
            tgt = data[:, k:]                         # token at t+k for t in [0, T-k)
            hit = (top[:, :, :T - k] == tgt[None, :, :, None]).any(-1).float().mean((1, 2))
            res[name][k] = hit.tolist()
    names = layer_names(len(hs))
    fig, axes = plt.subplots(1, len(K), figsize=(15, 4.4), sharey=False)
    for ax, k in zip(axes, K):
        for name, c in LENSES:
            ax.plot(names, [100 * v for v in res[name][k]], "-o", color=c, label=name)
        ax.set_title(f"token t+{k} in top-{topk}" + ("  (= next token)" if k == 1 else ""))
        ax.set_xlabel("layer")
    axes[0].set_ylabel("% of positions")
    axes[0].legend(loc="upper left")
    suptitle(fig, "Experiment 4 · Do intermediate layers look further ahead than the next token?",
             "Hit rate of the actual future token among each lens's top-10 readout. "
             "J-lens averages effects on all later positions, so it can surface 'plans'.")
    fig.subplots_adjust(top=0.78, wspace=0.22)
    save(fig, out, "exp4_lookahead.png")
    return {n: {str(k): v for k, v in d.items()} for n, d in res.items()}


# ---------------------------------------------------------------- E5 crystallization
FUNC = set("the a an to and was of in on he she it they his her that with for is at but so as said had were "
           "you i we be not all very up out then".split())


def token_category(s):
    w = s.strip()
    if not w or not any(ch.isalnum() for ch in w):
        return "punctuation"
    if w.lower() in FUNC:
        return "function word"
    if w[0].isupper() and s.startswith(" "):
        return "Capitalised (names)"
    return "content word"


def exp5_crystallization(dec, tk, data, J, out):
    hs, final = residuals(dec, data)
    top_final = final.argmax(-1)
    correct = top_final[:, :-1] == data[:, 1:]
    cats = [[token_category(tk.tok(i)) for i in row[1:]] for row in data]
    order = ["punctuation", "function word", "content word", "Capitalised (names)"]
    res = {}
    nL = len(hs)
    for name, LG in all_lens_logits(dec, hs, J).items():
        same = (LG.argmax(-1) == top_final[None])[:, :, :-1]       # [L+1, B, T-1]
        # earliest layer from which the lens agrees with the final answer at every later layer
        stays = torch.flip(torch.cumprod(torch.flip(same.int(), [0]), 0), [0]).bool()
        depth = (nL - stays.sum(0)).float()                         # [B, T-1]
        res[name] = {}
        for cat in order:
            mask = torch.tensor([[c == cat for c in row] for row in cats]) & correct
            res[name][cat] = {"mean_depth": depth[mask].mean().item(), "n": int(mask.sum()),
                              "hist": torch.bincount(depth[mask].long(), minlength=nL).tolist()}

    names = layer_names(nL)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    y = torch.arange(len(order)).float()
    h = 0.36
    for j, (name, c) in enumerate(LENSES):
        vals = [res[name][cat]["mean_depth"] for cat in order]
        ax.barh(y + (j - 0.5) * (h + 0.04), vals, height=h, color=c, label=name)
        for yi, v in zip(y, vals):
            ax.text(v + 0.05, yi + (j - 0.5) * (h + 0.04), f"{v:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [f"{c}\n(n={res['J-lens'][c]['n']})" for c in order])
    ax.invert_yaxis()
    ax.set_xlim(0, nL - 0.5)
    ax.set_xticks(range(nL), names)
    ax.grid(axis="y", visible=False)
    ax.set_title("Mean layer where the final answer locks in", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)

    ax = axes[1]
    for name, c in LENSES:
        tot = torch.zeros(nL)
        for cat in order:
            tot += torch.tensor(res[name][cat]["hist"], dtype=torch.float)
        cum = (tot.cumsum(0) / tot.sum() * 100).tolist()
        ax.plot(names, cum, "-o", color=c, label=name)
        res[name]["cumulative_pct"] = cum
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of correct predictions already locked in")
    ax.set_title("Cumulative share of decided tokens by layer")
    ax.legend(loc="lower right")
    suptitle(fig, "Experiment 5 · Which words are decided early, which late?",
             "Only positions the model predicts correctly. 'Locks in' = from this layer on, the lens top-1 "
             "equals the final output.")
    fig.subplots_adjust(top=0.74, wspace=0.35)
    save(fig, out, "exp5_crystallization.png")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--J", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent / "figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    tk = ByteBPE(args.vocab)
    dec = load_model(args.ckpt, len(tk.token_to_id))
    Jd = torch.load(args.J)
    J = Jd["J"]
    held_out = corpus_windows(tk, ROOT / "data/tinystory/processed/val.txt", 64, 64, 1, (1200, None))

    results = {"J_fit": {"n_windows": Jd["n"], "T": Jd["T"]}}
    results["exp1"] = exp1_faithfulness(dec, held_out, J, Jd["JA"], Jd["JB"], out)
    results["exp2a"] = exp2_thought_grid(
        dec, tk, J, out, "Once upon a time, there was a little girl named Lily. She loved to play with her",
        "exp2_thought_grid_lily.png", "Experiment 2 · Layer-by-layer 'thoughts' on a story opening")
    results["exp2b"] = exp2_thought_grid(
        dec, tk, J, out, "Tim was hungry. His mom gave him a big red apple. Tim said, \"Thank you, Mom!\" and",
        "exp2_thought_grid_tim.png", "Experiment 2b · Layer-by-layer 'thoughts' on dialogue")
    results["exp3"] = exp3_answer_trajectories(dec, tk, J, out)
    results["exp4"] = exp4_lookahead(dec, held_out, J, out)
    results["exp5"] = exp5_crystallization(dec, tk, held_out, J, out)
    (out / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("wrote", out / "results.json")
