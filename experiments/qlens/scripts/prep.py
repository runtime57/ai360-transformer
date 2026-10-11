"""Per-model preparation (bf16): J-lens (fit with jlens, or load a pre-fitted one),
layer means mu_l and the mean aggregated target ybar(t), all on the fit corpus.

mu_l      = mean over fit sequences and valid positions of h_l[t]
ybar(t)   = mean over fit sequences of y[t],  y[t] = sum_{t'=t}^{T-2} h_final[t']
"""

import argparse
import logging
import os

import torch
import yaml

from qlens.core import (WORK, Timer, encode, get_splits, git_hash, load_lens, load_model, record,
                        save_json, seed_all, valid_positions)

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
seed_all(cfg["seed"])
tag = cfg["tag"]
os.makedirs(os.path.join(WORK, tag), exist_ok=True)
model, tok = load_model(cfg["model"], torch.bfloat16)
L, d = model.config.num_hidden_layers, model.config.hidden_size
splits = get_splits(tok, cfg["data"])
fit_texts = [t for _, t in splits["fit"]]
meta = {"config": cfg, "git": git_hash(), "seed": cfg["seed"]}

lens_path = os.path.join(WORK, tag, "lens.pt")
if not os.path.exists(lens_path):
    if cfg.get("pretrained_lens"):
        from huggingface_hub import hf_hub_download

        src = hf_hub_download(cfg["pretrained_lens"]["repo"], cfg["pretrained_lens"]["file"])
        os.symlink(src, lens_path)
        meta["lens"] = {"source": "pretrained", **cfg["pretrained_lens"]}
    else:
        import jlens

        jlens.configure_logging()
        logging.getLogger("jlens").setLevel(logging.INFO)
        lm = jlens.from_hf(model, tok)
        with Timer("jlens.fit") as tm:
            lens = jlens.fit(lm, fit_texts, dim_batch=cfg["dim_batch"], max_seq_len=128,
                             checkpoint_path=os.path.join(WORK, tag, "fit_ckpt.pt"))
        lens.save(lens_path)
        meta["lens"] = {"source": "fit", "n_prompts": lens.n_prompts, "seconds": tm.dt,
                        "dim_batch": cfg["dim_batch"]}
J = load_lens(lens_path)
meta["lens_layers"] = sorted(J)

# mu_l and ybar(t)
with Timer("means") as tm:
    mu = torch.zeros(L, d, dtype=torch.float64, device="cuda")
    ysum, n_pos, n_seq = None, 0, 0
    for text in fit_texts:
        ids = encode(tok, text, "cuda")
        T = ids.shape[1]
        acts = record(model, ids, range(L))
        pos = torch.tensor(valid_positions(T), device="cuda")
        for l in range(L):
            mu[l] += acts[l][0, pos].double().sum(0)
        n_pos += len(pos)
        fin = acts[L - 1][0, : T - 1].double()            # targets t' <= T-2
        y = fin.flip(0).cumsum(0).flip(0)                  # y[t] = sum_{t'>=t}^{T-2}
        ysum = y if ysum is None else ysum + y
        n_seq += 1
    mu = (mu / n_pos).float().cpu()
    ybar = (ysum / n_seq).float().cpu()                    # [T-1, d], rows t < 16 unused
torch.save({"mu": mu, "ybar": ybar, "n_pos": n_pos, "n_seq": n_seq}, os.path.join(WORK, tag, "stats.pt"))
meta["stats_seconds"] = tm.dt
save_json(meta, os.path.join(os.environ["QLENS_ROOT"], "results", "phase0", tag, "prep.json"))
print("done", meta)
