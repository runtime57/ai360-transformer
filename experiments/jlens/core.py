"""J-lens (Jacobian lens) for the TinyStories GPT.

J_l = E_{prompt, t, t' >= t} [ d h_final,t' / d h_l,t ]
lens(h_l) = softmax(W_U · LN_final(J_l h_l))

h_0 is the (scaled) token + positional embedding, h_l is the residual stream after
block l, h_L is the residual stream before the final LayerNorm. The logit lens is
the special case J_l = I.
"""
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.model import GPT, TransformerDecoder  # noqa: E402


# ---------------------------------------------------------------- tokenizer
# Pure-python byte-level BPE that reproduces the HF `tokenizers` pipeline the
# model was trained with (ByteLevel pre-tokenizer, no prefix space, BPE merges).

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


_GPT2_PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""")


class ByteBPE:
    def __init__(self, path):
        state = json.loads(Path(path).read_text())
        self.token_to_id = state["token_to_id"]
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        merges = json.loads(state["model_state"]["tokenizer_json"])["model"]["merges"]
        self.ranks = {tuple(m): r for r, m in enumerate(merges)}
        self.b2u = _bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.bos, self.eos = self.token_to_id["<bos>"], self.token_to_id["<eos>"]

    @lru_cache(maxsize=100_000)
    def _bpe(self, word):
        parts = list(word)
        while len(parts) > 1:
            pairs = [(self.ranks.get(p, 1e18), i) for i, p in enumerate(zip(parts, parts[1:]))]
            rank, i = min(pairs)
            if rank == 1e18:
                break
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        return tuple(parts)

    def encode(self, text, bos=True):
        ids = [self.bos] if bos else []
        for piece in _GPT2_PAT.findall(text):
            word = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            ids += [self.token_to_id.get(p, self.token_to_id["<unk>"]) for p in self._bpe(word)]
        return ids

    def tok(self, i):
        """Readable form of a single token id."""
        t = self.id_to_token[int(i)]
        if t.startswith("<"):
            return t
        s = bytes(self.u2b[c] for c in t).decode("utf-8", errors="replace")
        return s.replace("\n", "↵")

    def decode(self, ids):
        return "".join(self.tok(i) for i in ids if self.id_to_token[int(i)] not in ("<bos>", "<eos>", "<pad>"))


# ---------------------------------------------------------------- model

def load_model(ckpt_path, vocab_size):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]["model"]["decoder"]
    dec = TransformerDecoder(
        d_model=cfg["d_model"], vocab_size=vocab_size, num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], use_trainable_pos_embeds=cfg["use_trainable_pos_embeds"],
        max_seq_len=cfg["max_seq_len"],
    )
    model = GPT(dec)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.decoder


def residuals(dec, seq, grad=False):
    """Run the decoder and return [h_0, ..., h_L] (each [B, T, D]) and final logits."""
    B, T = seq.shape
    pos = torch.arange(T)
    dims = torch.arange(dec.d_model)
    angles = pos.unsqueeze(-1) / 10000 ** (2 * (dims // 2) / dec.d_model)
    pos_embed = torch.where(dims % 2 == 0, torch.sin(angles), torch.cos(angles))
    h = dec.d_model ** 0.5 * dec.token_embedding(seq) + pos_embed
    if grad:
        h.requires_grad_(True)
    hs = [h]
    for block in dec.transformer_blocks:
        h = block(h)
        hs.append(h)
    logits = dec.final_norm(h) @ dec.token_embedding.weight.T
    return hs, logits


def unembed(dec, h):
    return dec.final_norm(h) @ dec.token_embedding.weight.T


# ---------------------------------------------------------------- J fitting

def fit_jacobians(dec, batches, chunk=64, log=print):
    """Estimate J_l for every layer l (including L, where it is a scaled identity).

    For a batch, S = sum_{t'} h_L,t' ; d S_i / d h_l,t  summed over (b, t) equals row i
    of sum_{pairs t' >= t} dh_L,t'/dh_l,t (causality zeroes t' < t). Rows are computed
    `chunk` at a time by replicating the batch and sending a different one-hot
    cotangent through each replica in a single backward pass.
    """
    D, L = dec.d_model, len(dec.transformer_blocks)
    J = torch.zeros(L + 1, D, D, dtype=torch.float64)
    n_pairs = 0
    for bi, seq in enumerate(batches):
        B, T = seq.shape
        n_pairs += B * T * (T + 1) // 2
        for start in range(0, D, chunk):
            k = min(chunk, D - start)
            rep = seq.repeat(k, 1)                       # [k*B, T]
            hs, _ = residuals(dec, rep, grad=True)
            for h in hs[1:]:
                h.retain_grad()
            S = hs[-1].sum(1).view(k, B, D)              # [k, B, D]
            cot = torch.zeros_like(S)
            cot[torch.arange(k), :, start + torch.arange(k)] = 1.0
            (S * cot).sum().backward()
            for l, h in enumerate(hs):
                g = h.grad.view(k, B, T, D).sum((1, 2))  # [k, D] -> rows start..start+k
                J[l, start:start + k] += g.double()
        log(f"  J fit: batch {bi + 1}/{len(batches)}")
    return (J / n_pairs).float()


def lens_logits(dec, h, J=None):
    """h: [..., D] residual at some layer; J: [D, D] or None (logit lens)."""
    if J is not None:
        h = h @ J.T
    return unembed(dec, h)
