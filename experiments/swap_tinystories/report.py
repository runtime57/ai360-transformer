"""Build REPORT.md from saved artifacts. Decision rules are fixed here, before looking at swap results:

- H1, H2, H3 (C2 > C5): exact one-sided McNemar test on paired test trials, success@cat, α = 0.05.
- H3 (C5 ≈ 0): upper end of the 95% Wilson CI of success@cat(C5) < 0.10.
- H4: on the dev grid of single-layer bands, effect(layer) = max over α of success@cat (ties broken by mean
  target margin); peak layer of C1 and C2 must be earlier than that of C6. One-sided bootstrap over dev
  trials (2000 resamples): p = P*(peak_C ≥ peak_C6).
"""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import ART, DATA, EXP, FIG, jload

LENSES = ["J", "logit"]
CONDS = ["C1", "C2", "C3", "C4", "C5", "C6"]
COND_NAMES = {"C1": "raw token", "C2": "k16 component", "C3": "residual (norm-matched)", "C4": "whole probe",
              "C5": "random (norm-matched)", "C6": "answer swap"}
COND_RU = {"C1": "токен животного", "C2": "k16-компонента", "C3": "остаток (норма как у C2)", "C4": "проба целиком",
           "C5": "случайный (норма как у C2)", "C6": "answer swap"}
YES = lambda ok: "подтверждена" if ok else "не подтверждена"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0, c - h), min(1, c + h)


def mcnemar_greater(a, b):
    """One-sided exact McNemar: H_A: P(a=1,b=0) > P(a=0,b=1)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    p = sum(math.comb(n, k) for k in range(n10, n + 1)) / 2 ** n if n else 1.0
    return p, n10, n01


def key(cond, lens):
    return f"{cond}|{None if cond == 'C4' else lens}"


def single_layer_curves(sw, cond, lens, idx=None):
    """success@cat and margin per single layer, max over α (on dev trials idx subset)."""
    layers = sorted({b[0] for b in sw["bands"] if len(b) == 1})
    succ, marg = [], []
    for l in layers:
        best = None
        for a in sw["alphas"]:
            g = sw["grid"][f"{cond}|{None if cond == 'C4' else lens}|{l}|{a}"]
            s, m = np.array(g["success_cat"], float), np.array(g["margin"], float)
            if idx is not None:
                s, m = s[idx], m[idx]
            cand = (s.mean(), m.mean())
            best = cand if best is None or cand > best else best
        succ.append(best[0])
        marg.append(best[1])
    return layers, np.array(succ), np.array(marg)


def peak(layers, succ, marg):
    return layers[max(range(len(layers)), key=lambda i: (succ[i], marg[i]))]


def hypotheses(sw):
    n_test, n_dev = len(sw["test_trials"]), len(sw["dev_trials"])
    T = lambda c, L: np.array(sw["test"][key(c, L)]["success_cat"], bool)
    H = {}
    p, a, b = mcnemar_greater(T("C2", "J"), T("C2", "logit"))
    H["H1"] = dict(x=T("C2", "J").mean(), y=T("C2", "logit").mean(), p=p, n10=a, n01=b, ok=p < 0.05)
    for L in LENSES:
        p, a, b = mcnemar_greater(T("C2", L), T("C3", L))
        H[f"H2|{L}"] = dict(x=T("C2", L).mean(), y=T("C3", L).mean(), p=p, n10=a, n01=b, ok=p < 0.05)
        p, a, b = mcnemar_greater(T("C2", L), T("C5", L))
        k5 = int(T("C5", L).sum())
        _, lo, hi = wilson(k5, n_test)
        H[f"H3|{L}"] = dict(x=T("C2", L).mean(), y=T("C5", L).mean(), p=p, n10=a, n01=b, k5=k5, lo=lo, hi=hi,
                            zero=hi < 0.10, ok=p < 0.05 and hi < 0.10)
    rng = np.random.default_rng(0)
    for L in LENSES:
        lay, s6, m6 = single_layer_curves(sw, "C6", L)
        pk6 = peak(lay, s6, m6)
        for c in ["C1", "C2"]:
            lay, s, m = single_layer_curves(sw, c, L)
            pk = peak(lay, s, m)
            boot = []
            for _ in range(2000):
                idx = rng.integers(0, n_dev, n_dev)
                boot.append(peak(*single_layer_curves(sw, c, L, idx)) >= peak(*single_layer_curves(sw, "C6", L, idx)))
            p = float(np.mean(boot))
            H[f"H4|{L}|{c}"] = dict(pk=pk, pk6=pk6, s=s.max(), s6=s6.max(), p=p, ok=pk < pk6 and p < 0.05)
    return H, T


def cluster_boot(sw, T):
    prompts = [tr["prompt"] for tr in sw["test_trials"]]
    groups = [np.array([i for i, p in enumerate(prompts) if p == u]) for u in sorted(set(prompts))]
    rng = np.random.default_rng(1)
    out = []
    for name, (ca, la), (cb, lb) in [("H1: J C2 − logit C2", ("C2", "J"), ("C2", "logit")),
                                      ("H2: J C2 − J C3", ("C2", "J"), ("C3", "J")),
                                      ("H2: logit C2 − logit C3", ("C2", "logit"), ("C3", "logit")),
                                      ("H3: J C2 − J C5", ("C2", "J"), ("C5", "J")),
                                      ("H3: logit C2 − logit C5", ("C2", "logit"), ("C5", "logit"))]:
        d = T(ca, la).astype(float) - T(cb, lb).astype(float)
        boots = np.array([d[np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])].mean()
                          for _ in range(5000)])
        out.append((name, d.mean(), np.quantile(boots, .025), np.quantile(boots, .975), np.mean(boots <= 0)))
    return out, len(groups)


def main():
    s1 = jload(ART / "stage1.json")
    meta = jload(DATA / "meta.json")
    s4 = jload(ART / "stage4.json")
    s5 = jload(ART / "stage5.json")
    sw = jload(ART / "swap_results.json")
    ex = jload(ART / "examples.json")
    jl = torch.load(ART / "jlens_n1000.pt")
    test_all, dev, test = jload(DATA / "test_all.json"), jload(DATA / "dev.json"), jload(DATA / "test.json")
    animals = meta["animals"]
    n_test, n_dev = len(sw["test_trials"]), len(sw["dev_trials"])
    H, T = hypotheses(sw)
    cb, n_src_prompts = cluster_boot(sw, T)
    cbd = {name: r for name, *r in cb}
    cs = ex["clean_stats"]
    top1 = lambda c, L: np.mean(sw["test"][key(c, L)]["success_top1"])
    out = []
    w = out.append

    # ---------------- summary
    w("# Подмена невысказанного концепта в TinyStories-GPT: J-lens против logit lens (k=16)\n")
    w("## Коротко\n")
    w(f"- **H1 — {YES(H['H1']['ok'])}.** Подмена по k16-компоненте из словаря J-lens срабатывает чаще, чем из словаря logit lens: "
      f"success@cat {H['H1']['x']:.1%} против {H['H1']['y']:.1%} (McNemar p = {H['H1']['p']:.3f}; "
      f"с учётом того, что испытания сгруппированы по промптам, p* = {cbd['H1: J C2 − logit C2'][3]:.3f}).")
    w(f"- **H2 — для J {YES(H['H2|J']['ok'])}, для logit {YES(H['H2|logit']['ok'])}.** J: k16-компонента {H['H2|J']['x']:.1%} против "
      f"остатка {H['H2|J']['y']:.1%} (p = {H['H2|J']['p']:.3f}, но кластерный бутстрап p* = {cbd['H2: J C2 − J C3'][3]:.3f} — на грани); "
      f"logit: {H['H2|logit']['x']:.1%} против {H['H2|logit']['y']:.1%}, разницы нет.")
    w(f"- **H3 — {YES(H['H3|J']['ok'] and H['H3|logit']['ok'])}.** C2 сильно лучше случайного контроля "
      f"(p < 1e-6 для обеих линз), но сам случайный контроль не ≈0: C5 = {H['H3|J']['k5']}/{n_test} = {H['H3|J']['y']:.1%}, "
      f"95% CI [{H['H3|J']['lo']:.1%}, {H['H3|J']['hi']:.1%}] при заранее заданном пороге < 10%. Сильная случайная правка "
      "сбивает модель с исходного ответа, и цель иногда угадывается случайно (избирательность C5 ≈ 1/5, см. §7).")
    w(f"- **H4 — для J {YES(H['H4|J|C1']['ok'] or H['H4|J|C2']['ok'])}, для logit {YES(H['H4|logit|C1']['ok'] and H['H4|logit|C2']['ok'])}.** "
      f"Пик эффекта подмены концепта на одиночном слое раньше, чем у answer swap, в обеих линзах "
      f"(J: C1 слой {H['H4|J|C1']['pk']}, C2 слой {H['H4|J|C2']['pk']}, C6 слой {H['H4|J|C1']['pk6']}; "
      f"logit: C1 слой {H['H4|logit|C1']['pk']}, C2 слой {H['H4|logit|C2']['pk']}, C6 слой {H['H4|logit|C1']['pk6']}), "
      f"но для J это незначимо на {n_dev} dev-испытаниях (бутстрап p = {H['H4|J|C1']['p']:.2f} и {H['H4|J|C2']['p']:.2f}).")
    w(f"- **Главная оговорка.** Модель вообще не произносит звук: ни на одном из {cs['n_prompts']} отобранных промптов первый токен "
      f"звука не стоит на 1-м месте (медианная вероятность {cs['median_p_gold_first_token']:.1%}). success@cat меряет только "
      f"порядок среди шести маловероятных звуков. По реальному ответу (success@top1) подмена концепта почти ничего не меняет: "
      f"C1/C2 — {top1('C1', 'J'):.0%}–{top1('C2', 'J'):.0%}, тогда как answer swap (C6, J) — {top1('C6', 'J'):.0%}.")
    w("- **Чтение (этап 4) — нулевой результат.** Токен животного на последней позиции не попадает в top-16 ни на одном слое "
      "ни в одной из линз (pass@16 = 0); медианный ранг ~700–2000 из 4096.")
    w(f"- **Маленькая выборка.** Из 60 звуковых промптов чистый фильтр прошли 30; у frog — 0, поэтому frog бывает только целью. "
      f"Испытаний: dev {n_dev}, test {n_test} из {n_src_prompts} test-промптов. α=4 (максимум сетки) выбран почти везде.\n")

    # ---------------- 1. setup
    w("## 1. Модель, данные, испытания\n")
    w("- Модель: `src/exp-data/tiny_stories_baseline.pth` — GPT-декодер, 6 блоков, d_model=512, 8 голов, MLP 2048, "
      "синусоидальные позиции, связанные эмбеддинги/unembedding, финальный LayerNorm `ln_f`; byte-level BPE на 4096 токенов; "
      "10 эпох на TinyStories, вход `<bos> история <eos>`. Наш val loss на TinyStories validation: 1.00 нат/токен. "
      "Токенизатор переписан без библиотеки `tokenizers` и сверен с ней: 0 расхождений на всех 21 990 val-текстах.")
    w("- Слои: слой l = residual после блока l (1..6); слой 6 = h_final (до `ln_f`); слой 0 = эмбеддинги.")
    w(f"- J-lens: первые 1000 историй TinyStories validation (≤128 токенов), среднее ∂h_final,t'/∂h_l,t по всем парам t'≥t. "
      f"Сходимость cos(J_200, J_1000) по слоям 0..6: {', '.join(f'{x:.4f}' for x in conv(jl))}. "
      "На слое 6 J = 2/(L+1)·I ровно (масштаб не важен — `ln_f` и нормировка его убирают), cos(J, I) = 1; "
      "readout logit lens на слое 6 совпадает с логитами модели (max |Δ| = 0). "
      f"Подгонка: 1000 историй за 358 с на A40 (100 историй на CPU — 87 мин).")
    w("- Проверка перебором (`check_jlens.py`): суммы Якобианов совпадают с `torch.func.jacrev` на всех слоях "
      "(max |Δ| ≤ 1.5e-4 при масштабе 30–70), элементы с t'<t ровно 0; J по одним и тем же 100 текстам на CPU и GPU совпадает до 7e-7.\n")
    w("**Этап 1 — 5 «названных» шаблонов на животное, log-prob ответа целиком, 4 варианта написания:**\n")
    w("| животное | золотой ответ | токены | побед | оставлено |")
    w("|---|---|---|---|---|")
    for a, r in s1["results"].items():
        w(f"| {a} | `{r['gold']}` | {' · '.join(r['gold_tokens'])} | {r['wins']}/5 | {'да' if a in s1['kept'] else 'нет'} |")
    w(f"\nОставлены: {', '.join(animals)}. Кандидаты категории во всех метриках: "
      f"{', '.join('`' + meta['gold'][a] + '`' for a in animals)}. "
      "У `Ribbit` и `Roar` общий первый токен `R`, поэтому пары frog→lion и lion→frog исключены из всех условий.\n")
    w("**Этап 2 — данные.** По 20 описаний на животное вручную (10 probe с концовками `It liked to eat` / `It lived in a`, "
      "10 test с концовкой `Every morning it said, \"`). Автопроверка: ни названий, ни детёнышей, ни звуков ни одного из 12 "
      "животных-кандидатов (0 вхождений). Фильтр чистой моделью:\n")
    w("| животное | звуковых промптов | чистая модель выбирает золотой | dev | test |")
    w("|---|---|---|---|---|")
    for a in animals:
        w(f"| {a} | 10 | {sum(r['animal'] == a and r['clean_pred'] == a for r in test_all)} | "
          f"{sum(r['animal'] == a for r in dev)} | {sum(r['animal'] == a for r in test)} |")
    w(f"\nИспытание = (промпт источника s, цель t≠s): **dev {n_dev}, test {n_test}**. Чистый success@cat = 0 по построению "
      f"({np.mean(sw['clean_dev']['success_cat']):.2f}/{np.mean(sw['clean_test']['success_cat']):.2f}). "
      "Сетка dev: 18 полос (6 одиночных слоёв + окна 2–4) × α ∈ {0.5, 1, 2, 4}; лучшая (полоса, α) для каждой пары "
      "(условие, линза) — по success@cat на dev, при равенстве — по среднему отрыву log-prob цели. "
      "C3: масштаб подбирается по каждому испытанию (метод секущих), чтобы средняя ‖Δh‖ совпадала с C2 (отношение 1.00±0.01). "
      "C5: случайное направление на каждой позиции и слое с ‖Δh‖ из прогона C2 того же испытания.\n")

    # ---------------- 2. main table
    w("## 2. Результаты на test (настройки выбраны на dev)\n")
    w("success@cat — ответ цели имеет наибольший полный log-prob среди 6 кандидатов; success@top1 — первый токен ответа цели "
      "является top-1 по всему словарю. 95% CI Вильсона; испытания с общим промптом зависимы, поэтому интервалы оптимистичны "
      "(кластерный бутстрап — в §7).\n")
    w("| условие | J: полоса, α | J success@cat | J success@top1 | logit: полоса, α | logit success@cat | logit success@top1 |")
    w("|---|---|---|---|---|---|---|")
    for c in CONDS:
        cells = []
        for L in LENSES:
            if c == "C4" and L == "logit":
                cells += ["(то же, что J)", "", ""]
                continue
            k = key(c, L)
            ch = sw["chosen"][k]
            cells += [f"{'-'.join(map(str, ch['band']))}, {ch['alpha']:g}", fmt_ci(sw["test"][k]["success_cat"]),
                      fmt_ci(sw["test"][k]["success_top1"])]
        w(f"| {c} {COND_RU[c]} | " + " | ".join(cells) + " |")
    w("")

    # ---------------- 3. hypotheses
    w("## 3. Гипотезы\n")
    w("Правила решения зафиксированы в `report.py` до просмотра результатов подмены: H1–H3 — точный односторонний тест Макнемара "
      "на парных test-испытаниях (α = 0.05); «C5 ≈ 0» — верхняя граница 95% CI Вильсона < 0.10; H4 — пик на dev-сетке "
      "одиночных слоёв (максимум по α), C1/C2 раньше C6 и односторонний бутстрап по dev-испытаниям p < 0.05.\n")
    h = H["H1"]
    w(f"- **H1** success@cat(J,C2) > success@cat(logit,C2): {h['x']:.3f} против {h['y']:.3f}; несовпадающих пар {h['n10']}/{h['n01']}, "
      f"p = {h['p']:.3g} → **{YES(h['ok'])}**.")
    for L in LENSES:
        h = H[f"H2|{L}"]
        w(f"- **H2 ({L})** C2 > C3: {h['x']:.3f} против {h['y']:.3f}; несовпадающих {h['n10']}/{h['n01']}, p = {h['p']:.3g} → **{YES(h['ok'])}**.")
    for L in LENSES:
        h = H[f"H3|{L}"]
        w(f"- **H3 ({L})** C2 > C5 и C5 ≈ 0: {h['x']:.3f} против {h['y']:.3f}; несовпадающих {h['n10']}/{h['n01']}, p = {h['p']:.3g}; "
          f"C5 = {h['k5']}/{n_test}, 95% CI [{h['lo']:.3f}, {h['hi']:.3f}] → C5≈0: {'да' if h['zero'] else 'нет'} → **{YES(h['ok'])}**.")
    for L in LENSES:
        for c in ["C1", "C2"]:
            h = H[f"H4|{L}|{c}"]
            w(f"- **H4 ({L}, {c} против C6)** пик {c} — слой {h['pk']} (dev success@cat {h['s']:.3f}), C6 — слой {h['pk6']} "
              f"({h['s6']:.3f}); бутстрап p = P*(пик {c} ≥ пик C6) = {h['p']:.3f} → **{YES(h['ok'])}**.")
    w("")

    # ---------------- 4. figures
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for i, L in enumerate(LENSES):
        for c, st in zip(["C1", "C2", "C6", "C3", "C5"], ["o-", "s-", "^-", "x:", "d:"]):
            lay, s, _ = single_layer_curves(sw, c, L)
            ax[i].plot(lay, s, st, label=f"{c} {COND_NAMES[c]}")
        lay, s, _ = single_layer_curves(sw, "C4", None)
        ax[i].plot(lay, s, "k--", alpha=.5, label="C4 whole probe")
        ax[i].set_title(f"{L}-lens dictionaries: dev success@cat, single-layer swap (max over α)", fontsize=10)
        ax[i].set_xlabel("layer (after block l)")
        ax[i].legend(fontsize=8)
    ax[0].set_ylabel("success@cat (dev)")
    fig.tight_layout()
    fig.savefig(FIG / "success_by_layer.png", dpi=130)
    heat_fig(sw)

    w("## 4. Графики\n")
    w("Успех подмены по одиночным слоям (dev, максимум по α) — C1, C2, C6 и контроли, обе линзы:\n")
    w("![success by layer](figures/success_by_layer.png)\n")
    w("C2 для двух словарей на одном графике (dev по слоям с 95% CI; ромбы — test на выбранной настройке):\n")
    w("![C2 J vs logit](figures/c2_j_vs_logit.png)\n")
    w("Вся dev-сетка (строки — полосы, столбцы — α):\n")
    w("![grid](figures/dev_grid.png)\n")
    w("### Этап 4: чтение\n")
    w("Медианный ранг токена ` <животное>` (из 4096) на последней позиции test-промптов; пунктир — тот же токен в readout "
      "среднего по корпусу residual (контроль частотного перекоса):\n")
    w("![ranks](figures/stage4_ranks.png)\n")
    w("| линза | медианный ранг, слои 0..6 | контроль (средний residual) | pass@1/5/10/16 минимального ранга по слоям |")
    w("|---|---|---|---|")
    for L in LENSES:
        o = s4[L]
        w(f"| {L} | {', '.join(f'{x:g}' for x in o['median_rank_by_layer'])} | "
          f"{', '.join(f'{x:g}' for x in o['freq_control_median_by_layer'])} | "
          f"{' / '.join(f'{o['pass_at_k_minrank'][str(k)]:.2f}' for k in [1, 5, 10, 16])} |")
    w(f"\nМинимальный ранг по слоям для каждого test-промпта — J: {s4['J']['min_rank_per_prompt']}; "
      f"logit: {s4['logit']['min_rank_per_prompt']}.\n")
    w("Метрики §4.1 на 200 отложенных историях validation (позиции 1..127): excess kurtosis readout, "
      "доля позиций, где top-1 модели входит в top-10 линзы, и доля соседних позиций с одинаковым top-1 линзы "
      "против перемешанного нуля:\n")
    w("![quality](figures/stage4_quality.png)\n")
    w("| линза | метрика | слой 0 | 1 | 2 | 3 | 4 | 5 | 6 |")
    w("|---|---|---|---|---|---|---|---|---|")
    for L in LENSES:
        o = s4[L]
        for name, kk in [("excess kurtosis", "excess_kurtosis"), ("top-1 модели ∈ top-10 линзы", "top10_contains_model_top1"),
                         ("автокорр. top-1 (t, t+1)", "top1_autocorr"), ("… перемешанный ноль", "top1_autocorr_shuffled")]:
            w(f"| {L} | {name} | " + " | ".join(f"{x:.3f}" for x in o[kk]) + " |")
    w("\nJ-lens усредняет влияние на все будущие позиции t'≥t, поэтому хуже logit lens предсказывает *следующий* токен "
      "(top-10 ∋ top-1 ниже на всех слоях 0–5), но даёт более устойчивый по позициям top-1 на слоях 4–5 "
      "(автокорреляция выше, чем у logit lens и у перемешанного нуля).\n")

    # ---------------- 5. k16 tokens
    w("## 5. k=16 компоненты на выбранном слое\n")
    w("Проба p_X = средний residual на последнем токене 10 probe-промптов животного минус среднее по всем животным; "
      "неотрицательный OMP по 16 атомам единичной длины. Слой — первый слой полосы, выбранной на dev для C2 этой линзы. "
      "Доля объяснённой дисперсии = 1 − ‖r‖²/‖p‖².\n")
    for L in LENSES:
        lay = str(sw["chosen"][key("C2", L)]["band"][0])
        w(f"**{L}-lens, слой {lay}**\n")
        w("| животное | доля дисп. | 16 токенов (коэффициент) |")
        w("|---|---|---|")
        for a in animals:
            s = s5[L][a][lay]
            w(f"| {a} | {s['frac_var_explained']:.3f} | " + ", ".join(f"`{t.strip() or repr(t)}` {c:.2f}" for t, c in zip(s["tokens"], s["coefs"])) + " |")
        w("")
    w("Доля дисперсии пробы, объяснённая c_X, по слоям 1..6:\n")
    w("| линза | животное | " + " | ".join(f"L{l}" for l in range(1, 7)) + " |")
    w("|---|---|" + "---|" * 6)
    for L in LENSES:
        for a in animals:
            w(f"| {L} | {a} | " + " | ".join(f"{s5[L][a][str(l)]['frac_var_explained']:.2f}" for l in range(1, 7)) + " |")
    w("")

    # ---------------- 6. examples
    w(f"## 6. Примеры ({ex['condition']}, test)\n")
    w("Жадное продолжение на 8 токенов (правка только на позициях промпта); «выбор» — argmax среди 6 звуков-кандидатов; "
      "p(t) — вероятность полного ответа цели. По 5 испытаний, по кругу по животным-источникам.\n")
    esc = lambda s: s.replace("\n", "\\n").replace("|", "\\|")
    for title, items in [("Удачные", ex["success"]), ("Неудачные", ex["fail"])]:
        w(f"**{title}**\n")
        w("| s → t | промпт | до: выбор / продолжение | после: выбор / продолжение | p(t) до → после |")
        w("|---|---|---|---|---|")
        for e in items:
            w(f"| {e['s']} → {e['t']} | {e['prompt']} | {e['pred_before']} / `{esc(e['before'])}` | "
              f"{e['pred_after']} / `{esc(e['after'])}` | {e['p_target_before']:.1e} → {e['p_target_after']:.1e} |")
        w("")

    # ---------------- 7. exploratory
    w("## 7. Разведочный анализ (post hoc, в проверке H1–H4 не используется)\n")
    w("Сбой = доля испытаний, где выбор ушёл с источника; избирательность = P(выбор = t | выбор ≠ s), случайный уровень ≈ 1/5.\n")
    w("| условие | линза | success@cat | сбой | избирательность | средний отрыв цели (нат) |")
    w("|---|---|---|---|---|---|")
    for c in CONDS:
        for L in (["J"] if c == "C4" else LENSES):
            r = sw["test"][key(c, L)]
            s, src = np.array(r["success_cat"], bool), np.array(r["still_source"], bool)
            dis = ~src
            sel = s[dis].mean() if dis.any() else float("nan")
            w(f"| {c} {COND_RU[c]} | {'—' if c == 'C4' else L} | {s.mean():.3f} | {dis.mean():.3f} | {sel:.3f} (n={dis.sum()}) | "
              f"{np.mean(r['margin']):.2f} |")
    w("\nВсе неслучайные условия направляют ответ именно в цель (избирательность 0.63–0.90), а случайный контроль — нет (≈0.16). "
      "То есть успех C1–C4/C6 — не только «сбить модель», хотя частично и это.\n")
    w(f"Зависимость испытаний: {n_test} test-испытаний построены из {n_src_prompts} промптов. Кластерный бутстрап по промптам "
      "(5000 повторов) для разностей success@cat; p* = P*(разность ≤ 0):\n")
    w("| сравнение | разность | 95% CI | p* |")
    w("|---|---|---|---|")
    for name, d, lo, hi, p in cb:
        w(f"| {name} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} |")
    w("")
    w("## 8. Воспроизведение\n")
    w("```\ncd experiments/swap_tinystories\npython stage1_capabilities.py      # этап 1\npython stage2_data.py              # этап 2 → data/*.json\n"
      "python fit_jlens.py --n 1000 --chunk 512 --checkpoints 100,200,1000   # этап 3 (GPU: ~6 мин)\npython check_jlens.py              # проверки J\n"
      "python stage4_read.py              # этап 4\npython stage5_probes.py            # этап 5\npython swap.py cuda                # этапы 6–8 (L4: ~5 мин)\n"
      "python examples.py && python report.py\n```\n")
    (EXP / "REPORT.md").write_text("\n".join(out))
    print("\n".join(out))


def conv(jl):
    J2, J1 = jl["J_at_n"][200], jl["J_at_n"][1000]
    return [torch.nn.functional.cosine_similarity(J2[l].flatten().double(), J1[l].flatten().double(), 0).item() for l in range(J1.shape[0])]


def fmt_ci(x):
    p, lo, hi = wilson(int(np.sum(x)), len(x))
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}] ({int(np.sum(x))}/{len(x)})"


def heat_fig(sw):
    bands = ["-".join(map(str, b)) for b in sw["bands"]]
    panels = [(c, L) for L in LENSES for c in ["C1", "C2", "C3", "C5", "C6"]] + [("C4", None)]
    fig, axes = plt.subplots(2, 6, figsize=(20, 8), sharey=True)
    for ax, (c, L) in zip(axes.flat, panels):
        M = np.array([[np.mean(sw["grid"][f"{c}|{L}|{b}|{a}"]["success_cat"]) for a in sw["alphas"]] for b in bands])
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(f"{c} {L or ''}", fontsize=9)
        ax.set_xticks(range(len(sw["alphas"])), [f"{a:g}" for a in sw["alphas"]])
        ax.set_yticks(range(len(bands)), bands, fontsize=7)
    axes.flat[-1].axis("off")
    fig.colorbar(im, ax=axes, shrink=.6, label="dev success@cat")
    fig.savefig(FIG / "dev_grid.png", dpi=110)


if __name__ == "__main__":
    main()
