"""Experiment 6 · Does the J-lens read out the *future* better than the logit lens?

Unlike exp4 (top-10 hit of x_{t+2}, x_{t+3} on random windows), this test
  * uses story-aligned held-out sequences (<bos> + story): TinyStories validation stories absent from val.txt,
  * scores only NOVEL future tokens: not in the context x_0..x_t and != the next token,
    so copying and next-token prediction cannot explain a win,
  * uses rank metrics only (temperature-free): the percentile of a future token among
    "negative" tokens = vocab minus context minus every token that occurs later in the story,
  * resamples stories for paired bootstrap CIs,
  * adds a mechanism ablation with distance-resolved Jacobians (fit_by_distance.py):
    J_same = E[dh_L,t/dh_l,t]  (no future averaging) vs  J_fut = E[dh_L,t'/dh_l,t], t' > t.

Primary endpoint (fixed before running): future-window AUC with W = 16, i.e. mean
percentile of the distinct novel tokens in x_{t+2..t+16}, per layer l < L,
J-lens (J.pt from fit.py) minus logit lens; 95% paired-bootstrap CI over stories.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from core import ROOT, ByteBPE, lens_logits, load_model, residuals
from fit_by_distance import stories, story_ids
from run_experiments import INK, INK2, MUTED, layer_names, plt, save, suptitle

C_LENS = {"logit lens": "#2a78d6", "J-lens": "#eb6834", "J_same (t'=t)": "#1baf7a", "J_fut (t'>t)": "#eda100"}
DMAX, W, T_MIN, MAX_LEN, N_BINS = 32, 16, 8, 128, 50


def heldout_stories(n, seed=0):
    """TinyStories validation stories that are NOT in data/tinystory/processed/val.txt (J.pt was fit on
    val.txt); the model was trained on the train split only, so these are unseen by model and lens fits."""
    import os
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset
    used = (ROOT / "data/tinystory/processed/val.txt").read_text()
    ds = load_dataset("roneneldan/TinyStories", split="validation").shuffle(seed=seed)
    out = []
    for ex in ds:
        txt = ex["text"].strip()
        if txt and txt[:100] not in used:
            out.append(txt)
            if len(out) == n:
                return out
    return out


def build_lenses(J_repo, Jd):
    J, cnt = Jd["J"].double().sum(0), Jd["count"].sum(0)        # sum halves -> [bucket, L+1, D, D]
    lens = {"logit lens": None, "J-lens": J_repo,
            "J_same (t'=t)": (J[0] / cnt[0]).float(),
            "J_fut (t'>t)": (J[1:].sum(0) / cnt[1:].sum()).float()}
    # dh_L,t'/dh_L,t = 0 for t' != t, so J_fut is undefined at the last layer; all lenses = output there
    lens["J_fut (t'>t)"][-1] = torch.eye(J.shape[-1])
    refit_all = (J.sum(0) / cnt.sum()).float()
    halves = Jd["J"].double()
    hc = Jd["count"]
    diag = {
        "cos(J_repo, J_all refit)": [F.cosine_similarity(a.flatten(), b.flatten(), 0).item()
                                     for a, b in zip(J_repo, refit_all)],
        "split_half_cos J_same": [F.cosine_similarity(a.flatten(), b.flatten(), 0).item() for a, b in
                                  zip(halves[0, 0] / hc[0, 0], halves[1, 0] / hc[1, 0])],
        "split_half_cos J_fut": [F.cosine_similarity(a.flatten(), b.flatten(), 0).item() for a, b in
                                 zip(halves[0, 1:].sum(0) / hc[0, 1:].sum(), halves[1, 1:].sum(0) / hc[1, 1:].sum())],
        "fit_pairs_per_bucket": cnt.tolist(), "buckets": Jd["buckets"], "fit_batches": Jd["batches_done"],
    }
    return lens, diag


def percentiles(scores, neg, targets, target_scores=None):
    """scores [K, T, V], neg [T, V] bool, targets [T, M] token ids
    -> share of negatives scored below each target (ties count 1/2), [K, T, M]."""
    st = scores.gather(-1, targets.expand(scores.shape[0], -1, -1)) if target_scores is None else target_scores
    s = scores.masked_fill(~neg, float("inf")).sort(-1).values
    n_neg = neg.sum(-1)[None, :, None]
    lo = torch.searchsorted(s, st, side="left")
    hi = torch.searchsorted(s, st, side="right").minimum(n_neg)
    return (lo + hi).double() / 2 / n_neg


def percentiles_matched(scores, neg, targets, bins):
    """Same, but each target is compared only with negatives from its own unigram-frequency bin."""
    st = scores.gather(-1, targets.expand(scores.shape[0], -1, -1))
    out = torch.full(st.shape, float("nan"), dtype=torch.float64)
    tb = bins[targets]
    for b in tb.unique().tolist():
        idx = (bins == b).nonzero()[:, 0]
        sel = (tb == b).expand_as(st)
        out = torch.where(sel, percentiles(scores[..., idx], neg[:, idx], targets, st), out)
    return out


def eval_story(dec, ids, lenses, unigram, bins, special):
    """-> {"raw" | "freq-matched": (pct [K, L+1, n, DMAX], uni [1, n, DMAX])}: percentile of x_{t+d}
          (nan if not a novel token), and first [n, DMAX]: x_{t+d} is its first occurrence in x_{t+2..t+d}."""
    T = len(ids)
    V = unigram.numel()
    hs, _ = residuals(dec, ids[None])
    pos = torch.arange(T_MIN, T - 2)                         # need at least x_{t+2}
    n = len(pos)
    seen = torch.zeros(T, V, dtype=torch.bool)               # token occurred at <= t
    seen[torch.arange(T), ids] = True
    seen = seen.cummax(0).values
    later = torch.zeros(T, V, dtype=torch.bool)              # token occurs at > t (anywhere later in story)
    later[torch.arange(T - 1), ids[1:]] = True
    later = later.flip(0).cummax(0).values.flip(0)
    neg = ~(seen | later)[pos]
    neg[:, special] = False

    d = torch.arange(1, DMAX + 1)
    tgt_pos = pos[:, None] + d[None]                         # [n, DMAX]
    valid = tgt_pos < T
    tgt = ids[tgt_pos.clamp(max=T - 1)]
    novel = valid & ~seen[pos[:, None], tgt]                  # not in context x_0..x_t
    novel[:, 1:] &= tgt[:, 1:] != tgt[:, :1]                  # d >= 2: also != next token
    first = torch.ones_like(novel)
    for j in range(2, DMAX):                                  # dedupe inside x_{t+2..t+d}
        first[:, j] = ~(tgt[:, 1:j] == tgt[:, j:j + 1]).any(1)

    Ss = [torch.stack([lens_logits(dec, h[0, pos], None if J is None else J[l]) for l, h in enumerate(hs)])
          for J in lenses.values()]                           # K x [L+1, n, V]
    # how much of each lens readout is a frequency prior: Spearman(score, unigram) over the vocab
    ur = unigram.argsort().argsort().double()
    ur = (ur - ur.mean()) / ur.std()
    sub = slice(None, None, 8)
    fc = []
    for S in Ss:
        r = S[:, sub].argsort(-1).argsort(-1).double()
        r = (r - r.mean(-1, keepdim=True)) / r.std(-1, keepdim=True)
        fc.append((r * ur).mean(-1).mean(-1) * V / (V - 1))   # [L+1]
    out = {"freq_corr": torch.stack(fc)}
    for metric, fn in [("raw", percentiles), ("freq-matched", lambda *a: percentiles_matched(*a, bins))]:
        rows = [fn(S, neg, tgt) for S in Ss]                  # K x [L+1, n, DMAX]
        uni = fn(unigram.expand(1, n, V), neg, tgt)           # [1, n, DMAX]
        out[metric] = (torch.stack(rows).masked_fill(~novel, float("nan")), uni.masked_fill(~novel, float("nan")))
    return out, first


def window_auc(pct, first, w, start=2):
    """mean percentile over distinct novel tokens whose first occurrence after the next token lies in
    x_{t+start..t+w}, per position -> [..., n] (nan where a position has no such token)"""
    m = first[:, start - 1:w]
    x = pct[..., start - 1:w]
    ok = ~torch.isnan(x) & m
    per_pos = torch.where(ok, x, 0).sum(-1) / ok.sum(-1)
    return per_pos


def bootstrap_ci(per_story, n_boot=10000, seed=0):
    """per_story: [S, ...] story means -> mean, lo, hi over stories."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, per_story.shape[0], (n_boot, per_story.shape[0]), generator=g)
    boots = per_story[idx].nanmean(1)
    return per_story.nanmean(0), boots.quantile(0.025, 0), boots.quantile(0.975, 0)


def qualitative(dec, tk, ids, lenses, special, layer, t, k=12):
    hs, _ = residuals(dec, ids[None])
    ctx = set(ids[:t + 1].tolist()) | set(special)
    fut = set(ids[t + 2:].tolist()) - ctx - {int(ids[t + 1])}
    out = {"context": tk.decode(ids[:t + 1]), "next": tk.tok(ids[t + 1]),
           "rest_of_story": tk.decode(ids[t + 2:t + 30]), "layer": layer}
    for name, J in lenses.items():
        s = lens_logits(dec, hs[layer][0, t], None if J is None else J[layer])
        s[list(ctx)] = -float("inf")
        top = s.topk(k).indices.tolist()
        out[name] = [(tk.tok(i), i in fut) for i in top]
    return out


def ci_dict(x):
    m, lo, hi = bootstrap_ci(x)
    return {"mean": m.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}


def summarize(A, UA, FA, UFA, dsum, dn, udsum, udn, lens_names):
    """A/FA: [S, K, L+1] story-mean window AUC (W=16 / far window); UA/UFA: [S] unigram."""
    r = {"auc": {}, "auc_diff_vs_logit": {}, "far_auc": {}, "far_auc_diff_vs_logit": {}, "dist": {}}
    for k, ln in enumerate(lens_names):
        r["auc"][ln] = ci_dict(A[:, k])
        r["auc_diff_vs_logit"][ln] = ci_dict(A[:, k] - A[:, 0])
        r["far_auc"][ln] = ci_dict(FA[:, k])
        r["far_auc_diff_vs_logit"][ln] = ci_dict(FA[:, k] - FA[:, 0])
        r["dist"][ln] = (dsum[k] / dn[k]).tolist()            # [L+1][DMAX]
    r["auc"]["unigram"] = {k: v[0] for k, v in ci_dict(UA[:, None]).items()}
    r["far_auc"]["unigram"] = {k: v[0] for k, v in ci_dict(UFA[:, None]).items()}
    r["dist"]["unigram"] = (udsum[0] / udn[0]).tolist()
    k_fut, k_same, k_j = (lens_names.index(n) for n in ("J_fut (t'>t)", "J_same (t'=t)", "J-lens"))
    r["auc_diff_fut_minus_same"] = ci_dict(A[:, k_fut] - A[:, k_same])
    r["far_auc_diff_fut_minus_same"] = ci_dict(FA[:, k_fut] - FA[:, k_same])
    r["auc_diff_jlens_minus_output"] = ci_dict(A[:, k_j] - A[:, k_j, -1:])
    r["far_auc_diff_jlens_minus_output"] = ci_dict(FA[:, k_j] - FA[:, k_j, -1:])
    return r


def plot_main(res, names, out):
    x = list(range(len(names)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.4), gridspec_kw={"width_ratios": [1.25, 1]})
    for row, (metric, label) in enumerate([("raw", "all never-appearing tokens"),
                                           ("freq-matched", "same-frequency never-appearing tokens")]):
        r = res[metric]
        ax = axes[row, 0]
        for lens, c in C_LENS.items():
            ax.fill_between(x, r["auc"][lens]["lo"], r["auc"][lens]["hi"], color=c, alpha=0.18, lw=0)
            ax.plot(x, r["auc"][lens]["mean"], "-o", color=c, label=lens)
        u = r["auc"]["unigram"]["mean"]
        ax.axhline(u, color=MUTED, ls="--", lw=1.5)
        ax.text(0, u + 0.004, "unigram frequency", color=INK2, fontsize=9, va="bottom")
        ax.axhline(0.5, color=MUTED, ls=":", lw=1)
        ax.set_xticks(x, names)
        ax.set_ylabel(f"future-window AUC (W={W})")
        ax.set_title(f"{'A' if row == 0 else 'C'} · novel x[t+2…t+{W}] vs {label}", fontsize=11.5)
        if row == 0:
            ax.legend(loc="lower right", ncol=2)

        ax = axes[row, 1]
        for lens, off in [("J-lens", 0), ("J_same (t'=t)", -0.2), ("J_fut (t'>t)", 0.2)]:
            d = r["auc_diff_vs_logit"][lens]
            m, lo, hi = (torch.tensor(d[k][:-1]) for k in ("mean", "lo", "hi"))
            ax.errorbar([i + off for i in x[:-1]], m, yerr=[m - lo, hi - m], fmt="o", color=C_LENS[lens],
                        capsize=3, label=lens)
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(x[:-1], names[:-1])
        ax.set_ylabel("Δ AUC vs logit lens")
        ax.set_title(f"{'B' if row == 0 else 'D'} · paired difference, 95% bootstrap CI", fontsize=11.5)
        if row == 0:
            ax.legend(loc="lower left", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("layer")
    suptitle(fig, "Experiment 6 · Which lens sees the upcoming words?",
             f"{res['n_stories']} held-out stories, {res['n_positions']:,} positions. Only tokens absent from the "
             "context and different from the next token are scored; rank metric, so lens temperature is irrelevant.")
    fig.subplots_adjust(top=0.88, wspace=0.22, hspace=0.32)
    save(fig, out, "exp6_future_auc.png")


def plot_distance(res, names, out):
    layers = list(range(1, len(names)))
    fig, axes = plt.subplots(2, len(layers), figsize=(16, 8), sharey="row")
    d = list(range(1, DMAX + 1))
    for row, metric in enumerate(["raw", "freq-matched"]):
        for ax, l in zip(axes[row], layers):
            for lens, c in C_LENS.items():
                ax.plot(d, res[metric]["dist"][lens][l], "-", color=c, label=lens, lw=1.8)
            ax.plot(d, res[metric]["dist"]["unigram"], "--", color=MUTED, lw=1.3, label="unigram")
            ax.set_xscale("log", base=2)
            ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
            if row == 0:
                ax.set_title(names[l] + (" (= output)" if l == len(names) - 1 else ""))
            else:
                ax.set_xlabel("distance d")
        axes[row, 0].set_ylabel(("" if row == 0 else "frequency-matched\n") + "mean percentile of x[t+d]")
    axes[0, 0].legend(loc="lower left", fontsize=8.5)
    suptitle(fig, "Experiment 6b · How far ahead does each lens look?",
             "Percentile of the actual token at t+d among never-appearing tokens (top row) or never-appearing tokens "
             f"of the same frequency bin, 1/{N_BINS} of vocab (bottom row). Novel tokens only; d=1 is the next token.")
    fig.subplots_adjust(top=0.86, wspace=0.08, hspace=0.25)
    save(fig, out, "exp6_future_distance.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--J", required=True)
    ap.add_argument("--Jdist", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent / "figures"))
    ap.add_argument("--min_len", type=int, default=48)
    ap.add_argument("--n_stories", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)

    tk = ByteBPE(args.vocab)
    dec = load_model(args.ckpt, len(tk.token_to_id))
    lenses, diag = build_lenses(torch.load(args.J)["J"], torch.load(args.Jdist))
    lens_names = list(lenses)
    special = [tk.token_to_id[s] for s in ("<pad>", "<unk>", "<bos>", "<eos>")]
    V = len(tk.token_to_id)

    # unigram baseline from the training corpus (log-count; only the ranking matters) and its deciles
    cnt = Counter()
    for s in stories(ROOT / "data/tinystory/processed/train.txt")[:4000]:
        cnt.update(tk.encode(s, bos=False))
    unigram = torch.zeros(V)
    for i, c in cnt.items():
        unigram[i] = c
    unigram = (unigram + 0.5).log() + 1e-6 * torch.rand(V, generator=torch.Generator().manual_seed(0))
    bins = unigram.argsort().argsort() * N_BINS // V

    seqs = [s for s in story_ids(tk, heldout_stories(args.n_stories), MAX_LEN) if len(s) >= args.min_len]
    acc = {m: {"A": [], "UA": [], "FA": [], "UFA": [], "dsum": 0, "dn": 0, "udsum": 0, "udn": 0}
           for m in ("raw", "freq-matched")}
    n_pos, fcorr = 0, []
    for si, ids in enumerate(seqs):
        per_metric, first = eval_story(dec, ids, lenses, unigram, bins, special)
        fcorr.append(per_metric.pop("freq_corr"))
        for m, (pct, uni) in per_metric.items():
            a = acc[m]
            a["A"].append(window_auc(pct, first, W).nanmean(-1))
            a["UA"].append(window_auc(uni, first, W).nanmean(-1)[0])
            a["FA"].append(window_auc(pct, first, DMAX, start=8).nanmean(-1))
            a["UFA"].append(window_auc(uni, first, DMAX, start=8).nanmean(-1)[0])
            ok, uok = ~torch.isnan(pct), ~torch.isnan(uni)
            a["dsum"] = a["dsum"] + torch.where(ok, pct, 0).sum(2)
            a["dn"] = a["dn"] + ok.sum(2)
            a["udsum"] = a["udsum"] + torch.where(uok, uni, 0).sum(1)
            a["udn"] = a["udn"] + uok.sum(1)
        n_pos += first.shape[0]
        if si % 20 == 0:
            print(f"story {si + 1}/{len(seqs)}", flush=True)

    names = layer_names(len(dec.transformer_blocks) + 1)
    res = {"n_stories": len(seqs), "n_positions": n_pos, "W": W, "DMAX": DMAX, "far_window": [8, DMAX],
           "lenses": lens_names, "layers": names, "diagnostics": diag,
           "dist_n": acc["raw"]["dn"][0, 0].tolist(),
           "freq_corr": dict(zip(lens_names, torch.stack(fcorr).mean(0).tolist()))}
    for m, a in acc.items():
        res[m] = summarize(torch.stack(a["A"]), torch.stack(a["UA"]), torch.stack(a["FA"]), torch.stack(a["UFA"]),
                           a["dsum"], a["dn"], a["udsum"], a["udn"], lens_names)
    res["qualitative"] = [qualitative(dec, tk, seqs[i], lenses, special, 3, t)
                          for i, t in [(0, 30), (5, 40), (11, 25)] if i < len(seqs) and t + 3 < len(seqs[i])]

    plot_main(res, names, out)
    plot_distance(res, names, out)
    (out / "exp6_results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("wrote", out / "exp6_results.json")
