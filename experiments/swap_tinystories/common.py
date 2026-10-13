"""Shared utilities for the TinyStories concept-swap experiment.

The `tokenizers` package is not installed in this environment and there is
no network, so the byte-level BPE from the saved tokenizer JSON is
reimplemented here (GPT-2 style pre-tokenization + merge ranks).
"""

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
DATA = EXP / "data"
ART = EXP / "artifacts"
FIG = EXP / "figures"
for _d in (DATA, ART, FIG):
    _d.mkdir(exist_ok=True)

CKPT = ROOT / "src" / "exp-data" / "tiny_stories_baseline.pth"
TOK_PATH = ROOT / "src" / "exp-data" / "tokenizer_bpe_tiny_stories.json"
VAL_PARQUET = Path(os.environ["VAL_PARQUET"]) if "VAL_PARQUET" in os.environ else next((Path.home() / ".cache/huggingface/hub/datasets--roneneldan--TinyStories").rglob("validation-*.parquet"), None)

sys.path.insert(0, str(ROOT))

ANIMALS = {
    "cat": "meow", "dog": "woof", "cow": "moo", "duck": "quack", "sheep": "baa", "pig": "oink",
    "frog": "ribbit", "lion": "roar", "owl": "hoot", "bee": "buzz", "horse": "neigh", "snake": "hiss",
}


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


# GPT-2 / HF ByteLevel(use_regex=True) pattern; \p{L} ~ [^\W\d_], \p{N} ~ \d (enough for ASCII stories)
_PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")


class BPE:
    def __init__(self, path=TOK_PATH):
        state = json.load(open(path))
        tj = json.loads(state["model_state"]["tokenizer_json"])
        self.token_to_id = tj["model"]["vocab"]
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.ranks = {tuple(m): i for i, m in enumerate(tj["model"]["merges"])}
        self.b2u = _bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.bos = self.token_to_id["<bos>"]
        self.eos = self.token_to_id["<eos>"]
        self.pad = self.token_to_id["<pad>"]

    @lru_cache(maxsize=None)
    def _bpe(self, word):
        parts = list(word)
        while len(parts) > 1:
            pairs = [(self.ranks.get((parts[i], parts[i + 1]), 1 << 30), i) for i in range(len(parts) - 1)]
            r, i = min(pairs)
            if r == 1 << 30:
                break
            a, b = parts[i], parts[i + 1]
            merged, j = [], 0
            while j < len(parts):
                if j < len(parts) - 1 and parts[j] == a and parts[j + 1] == b:
                    merged.append(a + b)
                    j += 2
                else:
                    merged.append(parts[j])
                    j += 1
            parts = merged
        return tuple(parts)

    def encode(self, text, bos=False):
        ids = [self.bos] if bos else []
        for piece in _PAT.findall(text):
            word = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            ids.extend(self.token_to_id[t] for t in self._bpe(word))
        return ids

    def decode(self, ids):
        s = "".join(self.id_to_token[i] for i in ids if i > 3)
        return bytes(self.u2b[c] for c in s).decode("utf-8", errors="replace")

    def tok_str(self, i):
        return self.decode([i]).replace("\n", "\\n")


def load_model(device="cpu"):
    from src.model.gpt import TransformerDecoder

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ck["config"]["model"]["decoder"]
    dec = TransformerDecoder(d_model=cfg["d_model"], vocab_size=4096, num_layers=cfg["num_layers"], num_heads=cfg["num_heads"],
                             seq_dropout=0.0, ffn_dropout=0.0, attn_dropout=0.0)
    sd = {k[len("decoder."):]: v for k, v in ck["state_dict"].items()}
    dec.load_state_dict(sd)
    return dec.eval().to(device)


class Runner:
    """Thin functional wrapper around TransformerDecoder exposing the residual stream.

    Layer indexing: h[0] = embeddings (input of block 0), h[l] = residual after block l-1,
    so h[n_layers] = h_final (before ln_f). "Layer ℓ" in the task (= after block ℓ) is h[ℓ+1];
    we use the 1..n_layers convention throughout and also keep 0 (embeddings) for readouts.
    """

    def __init__(self, dec):
        self.dec = dec
        self.n_layers = len(dec.transformer_blocks)
        self.d = dec.d_model
        self.W_U = dec.token_embedding.weight  # tied
        self.g = dec.final_norm.weight
        self.b = dec.final_norm.bias

    def embed(self, seq):
        d = self.d
        L = seq.shape[1]
        pos = torch.arange(L, device=seq.device)
        dims = torch.arange(d, device=seq.device)
        angles = pos.unsqueeze(-1) / 10000 ** (2 * (dims // 2) / d)
        pe = torch.where(dims % 2 == 0, torch.sin(angles), torch.cos(angles))
        return d ** 0.5 * self.dec.token_embedding(seq) + pe

    def forward(self, seq, edit=None, return_resid=False, start=0, h=None):
        """edit: dict layer -> fn(h) applied to residual after block layer (1-based); 0 = embeddings."""
        if h is None:
            h = self.embed(seq)
            if edit and 0 in edit:
                h = edit[0](h)
        resid = [h] if return_resid else None
        for i in range(start, self.n_layers):
            h = self.dec.transformer_blocks[i](h)
            if edit and (i + 1) in edit:
                h = edit[i + 1](h)
            if return_resid:
                resid.append(h)
        logits = self.unembed(h)
        return (logits, resid) if return_resid else logits

    def ln_f(self, h):
        return self.dec.final_norm(h)

    def unembed(self, h):
        return self.ln_f(h) @ self.W_U.T


def pad_batch(seqs, pad):
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def jload(path):
    return json.loads(Path(path).read_text())
