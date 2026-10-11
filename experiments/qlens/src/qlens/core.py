"""Shared pieces: model loading, data sampling, layer choice, functional Qwen3 tail.

Conventions follow ``jlens`` (anthropics/jacobian-lens):
  * "layer l" = output of decoder block l (what ``ActivationRecorder`` hooks);
  * target = output of the last block (n_layers - 1), *before* the final norm;
  * valid positions = [SKIP_FIRST, T-2] (``jlens.fitting.valid_position_mask``);
  * the J estimator is  J_l = mean_{t valid} sum_{t' >= t, t' valid} dh_final[t'] / dh_l[t].
Every new quantity (Hessian, target y) uses the same aggregation:
  f_{c,t}(x) = sum_{t' >= t, t' valid} h_final[t'](h_l[t] := x).
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

SKIP_FIRST = 16  # jlens.fitting.SKIP_FIRST_N_POSITIONS
SEQ_LEN = 128

ROOT = os.environ.get("QLENS_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
WORK = os.environ.get("QLENS_WORK", os.path.join(ROOT, "work"))


# ----------------------------------------------------------------------------- utils

def git_hash() -> str:
    """Repo HEAD + content hash of src/scripts/configs (experiments/ is not committed yet)."""
    try:
        return open(os.path.join(ROOT, "CODE_VERSION")).read().strip()
    except OSError:
        return "unknown"


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
    os.replace(tmp, path)


def atomic_save(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class Timer:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.dt = time.perf_counter() - self.t0
        print(f"[timer] {self.name}: {self.dt:.1f}s", flush=True)


# ----------------------------------------------------------------------------- layers

def pct_layer(p: float, n_layers: int) -> int:
    """Layer index at p * depth, clipped to the fitted range [0, n_layers-2]."""
    return min(max(int(round(p * n_layers)), 0), n_layers - 2)


def layers_for(pcts, n_layers: int, last_k: int = 0) -> list[int]:
    out = {pct_layer(p, n_layers) for p in pcts}
    out |= {n_layers - 2 - i for i in range(last_k)}
    return sorted(out)


# ----------------------------------------------------------------------------- model

def load_model(name: str, dtype: torch.dtype, device: str = "cuda"):
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, attn_implementation="eager"
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    # jlens.from_hf(force_bos=True) does the same; no-op for Qwen3 (no BOS token).
    if getattr(tok, "bos_token_id", None) is not None and hasattr(tok, "add_bos_token"):
        tok.add_bos_token = True
    return model, tok


def encode(tok, text: str, device, max_length: int = SEQ_LEN) -> torch.Tensor:
    """Same tokenisation as ``jlens.HFLensModel.encode``."""
    return tok(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)


# ----------------------------------------------------------------------------- data

def _wikitext(split: str):
    import datasets

    return datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)


def sample_texts(tok, split: str, n: int, seed: int, exclude: set[int] = frozenset(),
                 min_tokens: int = SEQ_LEN) -> list[tuple[int, str]]:
    """Random WikiText-103 paragraphs (no headings) that tokenize to >= min_tokens.

    Returns (row_index, text); texts are later truncated to SEQ_LEN tokens."""
    ds = _wikitext(split)
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    out = []
    for i in order:
        if i in exclude:
            continue
        text = ds[i]["text"]
        s = text.strip()
        if len(s) < 400 or s.startswith("="):
            continue
        if len(tok(text, truncation=True, max_length=min_tokens).input_ids) < min_tokens:
            continue
        out.append((i, text))
        if len(out) == n:
            break
    return out


def get_splits(tok, cfg: dict) -> dict[str, list[tuple[int, str]]]:
    """fit (train), hess (train, disjoint from fit), heldout (validation). Cached to disk."""
    path = os.path.join(WORK, "data", f"splits_s{cfg['seed']}_{cfg['n_fit']}_{cfg['n_hess']}_{cfg['n_heldout']}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    fit = sample_texts(tok, "train", cfg["n_fit"], cfg["seed"])
    hess = sample_texts(tok, "train", cfg["n_hess"], cfg["seed"] + 1, exclude={i for i, _ in fit})
    held = sample_texts(tok, "validation", cfg["n_heldout"], cfg["seed"] + 2)
    splits = {"fit": fit, "hess": hess, "heldout": held}
    save_json(splits, path)
    return splits


def valid_positions(seq_len: int) -> list[int]:
    return list(range(SKIP_FIRST, seq_len - 1))


# ----------------------------------------------------------------------------- lens

def load_lens(path: str) -> dict[int, torch.Tensor]:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return {int(l): J.float() for l, J in ck["J"].items()}


@torch.no_grad()
def record(model, input_ids: torch.Tensor, layers) -> dict[int, torch.Tensor]:
    """Residual stream after each requested block (and the last block), via hooks."""
    blocks = model.model.layers
    want = sorted(set(layers) | {len(blocks) - 1})
    acts, handles = {}, []
    for i in want:
        def hook(_m, _inp, out, i=i):
            acts[i] = (out if torch.is_tensor(out) else out[0]).detach()
        handles.append(blocks[i].register_forward_hook(hook))
    try:
        model.model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return acts


def unembed(model, x: torch.Tensor) -> torch.Tensor:
    """Final RMSNorm + LM head (``jlens.HFLensModel.unembed``)."""
    w = model.lm_head.weight
    return model.lm_head(model.model.norm(x.to(w.dtype))).float()


# ----------------------------------------------------------------------------- functional Qwen3 tail

def _rms(x, w, eps):
    dt = x.dtype
    xf = x.float() if x.dtype in (torch.bfloat16, torch.float16) else x
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(dt)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@dataclass
class TailCache:
    """Per-block post-RoPE keys / values of a clean pass, plus RoPE tables."""
    k: list[torch.Tensor]  # per block: [1, n_kv, T, hd]
    v: list[torch.Tensor]
    cos: torch.Tensor      # [T, hd]
    sin: torch.Tensor
    hidden: list[torch.Tensor]  # per block output: [1, T, d]


class QwenTail:
    """Functional re-implementation of a Qwen3 decoder stack that can start at
    block l+1 for positions >= t only, with keys/values of positions < t taken
    from a clean pass (they do not depend on the perturbation at t: causal)."""

    def __init__(self, model):
        cfg = model.config
        self.model = model
        self.blocks = model.model.layers
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.eps = cfg.rms_norm_eps
        self.scaling = self.hd ** -0.5
        assert not getattr(cfg, "use_sliding_window", False)

    def rope(self, T: int, device, dtype):
        pos = torch.arange(T, device=device)[None]
        dummy = torch.zeros(1, T, self.hd, device=device, dtype=dtype)
        cos, sin = self.model.model.rotary_emb(dummy, pos)
        return cos[0].to(dtype), sin[0].to(dtype)

    def _block(self, j, x, cos, sin, k_pre=None, v_pre=None, return_kv=False):
        """x: [B, n, d] for absolute positions [t, t+n); k_pre/v_pre: [1|B, n_kv, t, hd]."""
        blk = self.blocks[j]
        at = blk.self_attn
        B, n, _ = x.shape
        h = _rms(x, blk.input_layernorm.weight, self.eps)
        q = at.q_proj(h).view(B, n, self.n_heads, self.hd)
        k = at.k_proj(h).view(B, n, self.n_kv, self.hd)
        v = at.v_proj(h).view(B, n, self.n_kv, self.hd).transpose(1, 2)
        q = _rms(q, at.q_norm.weight, self.eps).transpose(1, 2)
        k = _rms(k, at.k_norm.weight, self.eps).transpose(1, 2)
        c, s = cos[None, None], sin[None, None]
        q = q * c + _rotate_half(q) * s
        k = k * c + _rotate_half(k) * s
        new_k, new_v = k, v
        t = 0
        if k_pre is not None and k_pre.shape[2] > 0:
            t = k_pre.shape[2]
            k = torch.cat([k_pre.expand(B, -1, -1, -1), k], dim=2)
            v = torch.cat([v_pre.expand(B, -1, -1, -1), v], dim=2)
        g = self.n_heads // self.n_kv
        qg = q.reshape(B, self.n_kv, g, n, self.hd)
        scores = torch.einsum("bkgnh,bkmh->bkgnm", qg, k) * self.scaling
        qpos = torch.arange(n, device=x.device)[:, None] + t
        kpos = torch.arange(t + n, device=x.device)[None, :]
        scores = scores.masked_fill(kpos > qpos, float("-inf"))
        up = scores.float() if scores.dtype in (torch.bfloat16, torch.float16) else scores
        probs = F.softmax(up, dim=-1).to(q.dtype)
        o = torch.einsum("bkgnm,bkmh->bkgnh", probs, v).reshape(B, self.n_heads, n, self.hd)
        o = at.o_proj(o.transpose(1, 2).reshape(B, n, self.n_heads * self.hd))
        x = x + o
        h = _rms(x, blk.post_attention_layernorm.weight, self.eps)
        m = blk.mlp
        x = x + m.down_proj(F.silu(m.gate_proj(h)) * m.up_proj(h))
        if return_kv:
            return x, new_k, new_v
        return x

    @torch.no_grad()
    def clean(self, input_ids: torch.Tensor) -> TailCache:
        """Full clean pass (functional) recording per-block K, V and outputs."""
        x = self.model.model.embed_tokens(input_ids)
        T = input_ids.shape[1]
        cos, sin = self.rope(T, x.device, x.dtype)
        ks, vs, hs = [], [], []
        for j in range(self.n_layers):
            x, k, v = self._block(j, x, cos, sin, return_kv=True)
            ks.append(k)
            vs.append(v)
            hs.append(x)
        return TailCache(ks, vs, cos, sin, hs)

    def run(self, x: torch.Tensor, layer: int, t: int, cache: TailCache) -> torch.Tensor:
        """Run blocks layer+1 .. L-1 on rows x = h_layer[t:T] (B, n, d); returns final rows."""
        n = x.shape[1]
        cos, sin = cache.cos[t:t + n], cache.sin[t:t + n]
        for j in range(layer + 1, self.n_layers):
            x = self._block(j, x, cos, sin, cache.k[j][:, :, :t], cache.v[j][:, :, :t])
        return x

    def f(self, x: torch.Tensor, layer: int, t: int, cache: TailCache) -> torch.Tensor:
        """f_{c,t}: sum of final-block outputs over valid targets t' in [t, T-2]."""
        T = cache.cos.shape[0]
        out = self.run(x, layer, t, cache)
        return out[:, : T - 1 - t].sum(1)

    def hess(self, layer: int, t: int, cache: TailCache, U: torch.Tensor, V: torch.Tensor,
             with_jvp: bool = False):
        """H_{c,t}[u_b, v_b] for each row b of U, V ([B, d]) via nested forward-mode AD."""
        from torch.func import jvp

        base = cache.hidden[layer][:, t:]  # [1, n, d]
        B, n, d = U.shape[0], base.shape[1], base.shape[2]
        X = base.expand(B, n, d).contiguous()
        TU = torch.zeros(B, n, d, device=U.device, dtype=X.dtype)
        TV = torch.zeros_like(TU)
        TU[:, 0] = U.to(X.dtype)
        TV[:, 0] = V.to(X.dtype)

        def inner(y):
            return jvp(lambda z: self.f(z, layer, t, cache), (y,), (TU,))[1]

        if with_jvp:
            ju, huv = jvp(inner, (X,), (TV,))
            return huv, ju
        return jvp(inner, (X,), (TV,))[1]
