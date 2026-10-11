"""The six lens-eval sets of anthropics/jacobian-lens (data/evaluations/), read as documented
in their README: items[*] = {prompt, intermediates, [target]}; readout position = last prompt
token (the token preceding `target`), except poetry = the last newline token.

Concept tokens for a word w: the first token of "w" and of " w". A concept is
"multi-token" if neither form is a single token (we still use the first token, as asked).
order-ops: each key is expanded to a synonym set (README: numbers -> digit and word forms,
operations -> symbol and word forms); our expansion is listed in ORDER_OPS_SYNONYMS.
"""

from __future__ import annotations

import json
import os

SETS = ["association", "multihop", "multilingual", "order-ops", "poetry", "typo"]

_NUM = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
        "nineteen", "twenty", "twenty-one", "twenty-two", "twenty-three", "twenty-four"]
ORDER_OPS_SYNONYMS = {
    "addition": ["addition", "add", "plus", "sum", "+"],
    "subtraction": ["subtraction", "subtract", "minus", "difference", "-"],
    "multiplication": ["multiplication", "multiply", "times", "product", "*", "×"],
    "division": ["division", "divide", "divided", "quotient", "/", "÷"],
    "mod": ["mod", "modulo", "remainder", "%"],
    "squared": ["squared", "square", "^", "**", "²"],
}


def _forms(word: str) -> list[str]:
    return [word, " " + word]


def concept_tokens(tok, word: str, set_name: str):
    """Returns (token_ids, is_multi_token). token_ids = first token of every form."""
    if set_name == "order-ops":
        if word.isdigit():
            words = [word] + ([_NUM[int(word)]] if int(word) < len(_NUM) else [])
        else:
            words = ORDER_OPS_SYNONYMS.get(word, [word])
    else:
        words = [word]
    ids, single = set(), False
    for w in words:
        for form in _forms(w):
            t = tok(form, add_special_tokens=False).input_ids
            if not t:
                continue
            if set_name == "order-ops" and len(t) > 1:
                continue  # README: "min over single-token synonyms"
            ids.add(t[0])
            single |= len(t) == 1
    if not ids:  # every synonym is multi-token: fall back to first token of the key
        ids = {tok(" " + word, add_special_tokens=False).input_ids[0]}
    return sorted(ids), not single


def load_sets(repo_dir: str, tok) -> dict[str, list[dict]]:
    out = {}
    for name in SETS:
        items = json.load(open(os.path.join(repo_dir, "data", "evaluations", f"lens-eval-{name}.json")))["items"]
        rows = []
        for it in items:
            ids = tok(it["prompt"], add_special_tokens=True).input_ids
            if name == "poetry":
                nl = [i for i, t in enumerate(ids) if "\n" in tok.decode([t])]
                readout = nl[-1]
            else:
                readout = len(ids) - 1
            concepts = []
            for w in it["intermediates"]:
                toks, multi = concept_tokens(tok, w, name)
                concepts.append({"word": w, "tokens": toks, "multi_token": multi})
            rows.append({"name": it["name"], "ids": ids, "readout": readout, "concepts": concepts})
        out[name] = rows
    return out
