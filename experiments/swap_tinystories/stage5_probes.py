"""Stage 5: concept probes p_X and their k=16 nonnegative decompositions in the J and logit dictionaries."""

import torch

from common import ART, BPE, DATA, Runner, jdump, jload, load_model
from lens import dictionary, load_J, nn_omp

K = 16


@torch.no_grad()
def main(device="cpu"):
    tok, R = BPE(), Runner(load_model(device))
    J = load_J(device)
    meta = jload(DATA / "meta.json")
    animals = meta["animals"]
    probe = jload(DATA / "probe.json")
    layers = list(range(1, R.n_layers + 1))

    last = {a: [] for a in animals}  # per animal: [n_prompts, n_layers+1, d]
    for rec in probe:
        ids = torch.tensor([tok.encode(rec["prompt"], bos=True)], device=device)
        _, hs = R.forward(ids, return_resid=True)
        last[rec["animal"]].append(torch.stack([h[0, -1] for h in hs]))
    means = {a: torch.stack(v).mean(0) for a, v in last.items()}
    grand = torch.stack(list(means.values())).mean(0)
    P = {a: means[a] - grand for a in animals}  # [n_layers+1, d]

    store = {"layers": layers, "animals": animals, "p": {}, "c": {}, "r": {}}
    summary = {}
    for L in ["J", "logit"]:
        store["c"][L], store["r"][L] = {}, {}
        summary[L] = {}
        for l in layers:
            D = dictionary(R, J, L, l)
            for a in animals:
                p = P[a][l]
                ids_, coef, c, r = nn_omp(D, p, K)
                store["p"][(a, l)] = p.cpu()
                store["c"][L][(a, l)] = c.cpu()
                store["r"][L][(a, l)] = r.cpu()
                summary[L].setdefault(a, {})[l] = dict(
                    tokens=[tok.tok_str(i) for i in ids_], token_ids=ids_, coefs=[round(float(x), 4) for x in coef],
                    frac_var_explained=float(1 - r.norm() ** 2 / p.norm() ** 2),
                    animal_token_in_top16=animal_in(tok, a, ids_))
    torch.save(store, ART / "probes_decomp.pt")
    jdump(summary, ART / "stage5.json")

    mid = layers[len(layers) // 2 - 1]
    for a in animals[:3]:
        for L in ["J", "logit"]:
            s = summary[L][a][mid]
            print(f"[{a} | {L} | layer {mid}] var expl {s['frac_var_explained']:.3f}: "
                  + ", ".join(f"{t!r}:{w:.2f}" for t, w in zip(s["tokens"], s["coefs"])))
    print("\nfraction of probe variance explained by c_X (k=16):")
    for L in ["J", "logit"]:
        for a in animals:
            print(f"  {L:5s} {a:5s}", " ".join(f"{summary[L][a][l]['frac_var_explained']:.2f}" for l in layers))


def animal_in(tok, a, ids_):
    return tok.encode(" " + a)[0] in ids_


if __name__ == "__main__":
    import sys

    torch.set_num_threads(22)
    main(sys.argv[1] if len(sys.argv) > 1 else "cpu")
