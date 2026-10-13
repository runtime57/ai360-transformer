"""Stage 1: which animal -> sound mappings does the model know?"""

import torch

from common import ANIMALS, ART, BPE, Runner, jdump, load_model, pad_batch


def art(word):
    return ("an " if word[0] in "aeiou" else "a ") + word


TEMPLATES = [
    lambda a: f'Once upon a time, there was {art(a)}. Every morning the {a} said, "',
    lambda a: f'One day, a little {a} went for a walk. When it saw its friend, the {a} said, "',
    lambda a: f'There was a happy {a} who lived on a farm. Every day the {a} said, "',
    lambda a: f'Tim had {art(a)}. When Tim came home, the {a} looked at him and said, "',
    lambda a: f'Lily heard a sound outside. She looked and saw {art(a)}. The {a} said, "',
]


def variants(sound):
    return [sound.capitalize(), sound, " " + sound.capitalize(), " " + sound]


@torch.no_grad()
def answer_logprobs(R, tok, prompt, answers):
    """Sum log p(answer tokens | prompt), teacher-forced; prompt starts with <bos>."""
    p = tok.encode(prompt, bos=True)
    seqs = [p + tok.encode(a) for a in answers]
    lp = torch.log_softmax(R.forward(pad_batch(seqs, tok.pad)), -1)
    out = []
    for i, a in enumerate(answers):
        ids = tok.encode(a)
        pos = torch.arange(len(p) - 1, len(p) - 1 + len(ids))
        out.append(lp[i, pos, torch.tensor(ids)].sum().item())
    return out


def main():
    torch.set_num_threads(22)
    tok, R = BPE(), Runner(load_model())
    cands = [(s, v) for s in ANIMALS.values() for v in variants(s)]
    strings = [v for _, v in cands]
    results = {}
    for animal, sound in ANIMALS.items():
        single = len(tok.encode(" " + animal)) == 1
        scores = [answer_logprobs(R, tok, T(animal), strings) for T in TEMPLATES]  # [5][48]
        mean = torch.tensor(scores).mean(0)
        own = [i for i, (s, _) in enumerate(cands) if s == sound]
        gold_i = max(own, key=lambda i: mean[i])
        gold = strings[gold_i]
        wins, top = 0, []
        for sc in scores:
            best_other = max(sc[i] for i, (s, _) in enumerate(cands) if s != sound)
            wins += sc[gold_i] > best_other
            top.append(strings[max(range(len(sc)), key=lambda i: sc[i])])
        results[animal] = dict(sound=sound, gold=gold, gold_tokens=[tok.tok_str(i) for i in tok.encode(gold)],
                               gold_ids=tok.encode(gold), animal_token_single=single,
                               animal_tokens=[tok.tok_str(i) for i in tok.encode(" " + animal)],
                               wins=int(wins), top_per_template=top, mean_gold_logprob=float(mean[gold_i]))
    golds = [r["gold"].strip().lower() for r in results.values()]
    keep = [a for a, r in results.items() if r["wins"] >= 4 and r["animal_token_single"]]
    dup = {g for g in golds if golds.count(g) > 1}
    keep = [a for a in keep if results[a]["gold"].strip().lower() not in dup]
    jdump(dict(templates=[T("{animal}") for T in TEMPLATES], results=results, kept=keep), ART / "stage1.json")

    print("| animal | animal tok | single | gold | tokens | wins/5 | top-1 per template |")
    print("|---|---|---|---|---|---|---|")
    for a, r in results.items():
        print(f"| {a} | {r['animal_tokens']} | {r['animal_token_single']} | `{r['gold']}` | {r['gold_tokens']} | "
              f"{r['wins']}/5 | {', '.join(repr(t) for t in r['top_per_template'])} |")
    print("kept:", keep, len(keep))


if __name__ == "__main__":
    main()
