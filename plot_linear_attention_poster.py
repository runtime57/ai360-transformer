"""Build Russian poster figures and a point-by-point report from saved runs."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from plot_attention_results import (
    VARIANTS,
    load_histories,
    load_report,
    training_batch_times,
    validate_training_generation,
    variant,
)


NAMES = {
    ("softmax", False): "softmax",
    ("softmax", True): "softmax + RoPE",
    ("linear", False): "linear",
    ("linear", True): "linear + RoPE",
}
COLORS = {"softmax": "#4453a6", "linear": "#f06445"}
CONTEXT_COLORS = {512: "#aeb9e8", 8192: "#4054a5"}


def epoch_rows(history, split):
    rows = sorted(
        (r for r in history["rows"] if r["split"] == split and r["scope"] == "epoch"),
        key=lambda r: int(r["epoch"]),
    )
    if [int(r["epoch"]) for r in rows] != list(range(1, 11)):
        raise ValueError(f"Expected ten complete {split} epochs: {history['source']}")
    for row in rows:
        for field in ("loss", "perplexity", "token_accuracy"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Invalid {field}: {history['source']}")
        if row.get("nll") and not math.isclose(
            math.exp(float(row["nll"])), float(row["perplexity"]), rel_tol=1e-9
        ):
            raise ValueError(f"PPL does not equal exp(NLL): {history['source']}")
    return rows


def load_bundle(runs_dir, generation_path, quality_path, expected_context):
    histories = load_histories(runs_dir)
    if [h["key"] for h in histories] != VARIANTS:
        raise ValueError(f"Need exactly four attention variants: {runs_dir}")
    generation, metadata = load_report(generation_path)
    measured, configurations = validate_training_generation(histories, generation, metadata)
    timing = {variant(row): row for row in training_batch_times(histories)}
    for row in measured:
        key = variant(row)
        timing[key].update(
            generation_mean_ms=float(row["mean_ms"]),
            generation_tokens_per_second=float(row["tokens_per_second"]),
            generation_stdev_ms=float(row["stdev_ms"]),
        )
    cfg = configurations[0]
    if cfg["seq_length"] != expected_context:
        raise ValueError(f"Expected context {expected_context}: {runs_dir}")
    if len({row["training_batches"] for row in timing.values()}) != 1:
        raise ValueError(f"Different training budgets: {runs_dir}")
    quality, quality_metadata = load_report(quality_path)
    quality_by_key = {variant(row): row for row in quality}
    if set(quality_by_key) != set(VARIANTS) or len(quality) != 4:
        raise ValueError(f"Need test quality for all four variants: {quality_path}")
    if quality_metadata.get("split") != "test" or quality_metadata["seq_length"] != expected_context:
        raise ValueError(f"Wrong quality split or context: {quality_path}")
    if len({float(row["evaluated_tokens"]) for row in quality}) != 1:
        raise ValueError(f"Test variants evaluated different token counts: {quality_path}")
    for row in quality:
        checkpoint = Path(row["checkpoint"])
        if checkpoint.parent.parent.resolve() != runs_dir.resolve():
            raise ValueError(f"Quality belongs to another run: {checkpoint}")
        if not math.isclose(math.exp(float(row["nll"])), float(row["perplexity"]), rel_tol=1e-9):
            raise ValueError(f"Test PPL does not equal exp(NLL): {checkpoint}")
    return {
        "runs": runs_dir,
        "generation_source": generation_path,
        "quality_source": quality_path,
        "histories": {h["key"]: h for h in histories},
        "val": {h["key"]: epoch_rows(h, "val") for h in histories},
        "train": {h["key"]: epoch_rows(h, "train") for h in histories},
        "speed": timing,
        "config": cfg,
        "environment": metadata.get("environment", {}),
        "quality": quality_by_key,
        "quality_metadata": quality_metadata,
    }


def speedup(bundle, rope, field):
    soft = bundle["speed"][("softmax", rope)][field]
    linear = bundle["speed"][("linear", rope)][field]
    return linear / soft if field == "generation_tokens_per_second" else soft / linear


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dfe3eb", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#384151")


def save_figure(fig, output_dir, name):
    for extension in ("svg", "png"):
        fig.savefig(output_dir / f"{name}.{extension}", dpi=240, facecolor="white")
    plt.close(fig)


def comparison_captions(bundles):
    short, long = bundles
    long_training = [100 * (1 - 1 / speedup(long, rope, "training_ms_per_batch")) for rope in (False, True)]
    long_generation = [100 * (speedup(long, rope, "generation_tokens_per_second") - 1) for rope in (False, True)]
    short_training = [100 * (1 / speedup(short, rope, "training_ms_per_batch") - 1) for rope in (False, True)]
    short_generation = [100 * (1 - speedup(short, rope, "generation_tokens_per_second")) for rope in (False, True)]

    def span(values):
        return f"{min(values):.0f}–{max(values):.0f}%"

    return [
        (
            "Размер контекста 512: linear проигрывает по скорости",
            f"Время батча больше примерно на {np.mean(short_training):.0f}%; "
            f"генерация — на {span(short_generation)} токенов/с меньше.\n"
            "Контекст 512 короткий, квадратичные вычисления softmax ещё дешёвые.\n"
            "Softmax лучше оптимизирован.",
        ),
        (
            "Размер контекста 8192: получаем ускорение с linear",
            f"Время батча меньше на {span(long_training)}; "
            f"генерация — на {span(long_generation)} токенов/с больше.\n"
            "Сложность ядра: softmax O(T²d), linear O(Td²). При фиксированном d\n"
            "затраты linear растут линейно, а softmax — квадратично по контексту T.",
        ),
    ]


def plot_comparison(bundles, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 6.25), gridspec_kw={"width_ratios": [1.55, 1, 1]})
    fig.subplots_adjust(left=0.046, right=0.974, bottom=0.37, top=0.80, wspace=0.37)
    fig.suptitle("Linear Attention", x=0.046, y=0.967, ha="left", color=COLORS["softmax"], fontsize=25, weight="bold")
    ax = axes[0]
    for key in VARIANTS:
        rows = bundles[0]["val"][key]
        ax.plot(
            [int(r["epoch"]) for r in rows], [float(r["loss"]) for r in rows],
            color=COLORS[key[0]], linestyle="--" if key[1] else "-", linewidth=2.4, label=NAMES[key],
        )
    ax.set_title("Validation loss · контекст 512", loc="left", fontsize=14, pad=15)
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Validation loss ↓")
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_xlim(0.7, 10.3)
    ax.set_ylim(1.4, 2.7)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    style_axis(ax)
    for ax, field, title in zip(
        axes[1:], ("training_ms_per_batch", "generation_tokens_per_second"),
        ("Скорость обучения", "Скорость генерации"),
    ):
        x = np.arange(2)
        for index, bundle in enumerate(bundles):
            context = bundle["config"]["seq_length"]
            values = [speedup(bundle, rope, field) for rope in (False, True)]
            bars = ax.bar(x + (index - 0.5) * 0.33, values, width=0.30, color=CONTEXT_COLORS[context], zorder=3)
            ax.bar_label(bars, labels=[f"{v:.2f}×".replace(".", ",") for v in values], padding=5, fontsize=12, weight="bold")
        ax.axhline(1, color="#707887", linewidth=1.2, linestyle="--", zorder=2)
        ax.set_title(title, loc="left", fontsize=14, pad=15)
        ax.set_xticks(x, ["Без RoPE", "+ RoPE"])
        ax.set_xlabel("Linear против соответствующего softmax", fontsize=9.5, labelpad=10)
        ax.set_ylabel("Скорость linear / softmax, × ↑")
        ax.set_ylim(0, 1.55)
        ax.set_yticks([0, 0.5, 1, 1.5])
        style_axis(ax)
    fig.legend(
        handles=[Patch(facecolor=CONTEXT_COLORS[t], label=f"Контекст {t}") for t in (512, 8192)],
        loc="upper right", bbox_to_anchor=(0.972, 0.972), ncol=2, frameon=False, fontsize=12,
    )
    fig.add_artist(plt.Line2D([0.046, 0.974], [0.22, 0.22], transform=fig.transFigure, color="#dfe3eb", linewidth=1))
    for x, (title, body) in zip((0.046, 0.523), comparison_captions(bundles)):
        fig.text(x, 0.182, title, ha="left", va="top", fontsize=13.5, weight="bold", color=COLORS["softmax"])
        fig.text(x, 0.130, body, ha="left", va="top", fontsize=11, color="#384151", linespacing=1.55)

    save_figure(fig, output_dir, "linear_attention_comparison")


def plot_absolute(bundles, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 7.0))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.82, hspace=0.75, wspace=0.30)
    fig.suptitle("Линейное внимание и RoPE: абсолютные значения", x=0.055, y=0.97, ha="left", color=COLORS["softmax"], fontsize=21, weight="bold")
    fields = ("training_ms_per_batch", "generation_mean_ms", "generation_tokens_per_second")
    titles = ("Обучение, мс/батч ↓", "Генерация 64 токенов, мс ↓", "Новых токенов/с ↑")
    for row_index, bundle in enumerate(bundles):
        for ax, field, title in zip(axes[row_index], fields, titles):
            values = [bundle["speed"][key][field] for key in VARIANTS]
            bars = ax.bar(range(4), values, color=[COLORS[k[0]] for k in VARIANTS], edgecolor="white", linewidth=1)
            for bar, key in zip(bars, VARIANTS):
                bar.set_hatch("///" if key[1] else "")
            ax.bar_label(bars, labels=[f"{v:.1f}".replace(".", ",") for v in values], padding=4, fontsize=10)
            ax.set_xticks(range(4), ["softmax", "softmax\n+ RoPE", "linear", "linear\n+ RoPE"], fontsize=9)
            ax.set_title(title, loc="left", fontsize=12, pad=10)
            ax.set_ylim(0, max(values) * 1.26)
            style_axis(ax)
        axes[row_index, 0].set_ylabel(f"Контекст {bundle['config']['seq_length']}", color=COLORS["softmax"], weight="bold", fontsize=13, labelpad=13)
    save_figure(fig, output_dir, "linear_attention_speed_absolute")


def f(value, digits=6):
    return f"{float(value):.{digits}f}"


def table(lines, headings, rows):
    lines.append(" | ".join(headings))
    lines.append(" | ".join("-" * len(str(h)) for h in headings))
    lines.extend(" | ".join(map(str, row)) for row in rows)
    lines.append("")


def export_points(bundles, output_dir):
    points = []
    for key in VARIANTS:
        for row in bundles[0]["val"][key]:
            points.append(dict(figure="linear_attention_comparison", panel="validation_loss", run=bundles[0]["runs"].name, context=512, variant=NAMES[key], epoch=int(row["epoch"]), step=int(row["step"]), value=float(row["loss"]), unit="loss"))
    for bundle in bundles:
        for field, panel in (("training_ms_per_batch", "training_speedup"), ("generation_tokens_per_second", "generation_speedup")):
            for rope in (False, True):
                points.append(dict(figure="linear_attention_comparison", panel=panel, run=bundle["runs"].name, context=bundle["config"]["seq_length"], variant="linear / softmax" + (" + RoPE" if rope else ""), epoch="", step="", value=speedup(bundle, rope, field), unit="ratio"))
        for field, unit in (("training_ms_per_batch", "ms/batch"), ("generation_mean_ms", "ms"), ("generation_tokens_per_second", "tokens/s")):
            for key in VARIANTS:
                points.append(dict(figure="linear_attention_speed_absolute", panel=field, run=bundle["runs"].name, context=bundle["config"]["seq_length"], variant=NAMES[key], epoch="", step="", value=bundle["speed"][key][field], unit=unit))
    with (output_dir / "linear_attention_comparison_points.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    return points


def write_report(bundles, points, output_dir):
    short, long = bundles
    last = {key: short["val"][key][-1] for key in VARIANTS}
    lines = ["ЛИНЕЙНОЕ ВНИМАНИЕ И RoPE", "", "1. ГОТОВЫЕ ПОДПИСИ ДЛЯ ПОСТЕРА", ""]
    lines += ["Подписи в нижней части основного рисунка Linear Attention:", ""]
    for title, body in comparison_captions(bundles):
        lines += [title, body, ""]
    lines.append(
        f"RoPE снижает validation loss у обоих вариантов (512, эпоха 10): "
        f"softmax {f(last[('softmax', False)]['loss'], 3)} → {f(last[('softmax', True)]['loss'], 3)}; "
        f"linear {f(last[('linear', False)]['loss'], 3)} → {f(last[('linear', True)]['loss'], 3)}."
    )
    for bundle in bundles:
        context = bundle["config"]["seq_length"]
        for rope in (False, True):
            train = speedup(bundle, rope, "training_ms_per_batch")
            generation = speedup(bundle, rope, "generation_tokens_per_second")
            tag = "с RoPE" if rope else "без RoPE"
            train_change = 100 * (1 - 1 / train)
            throughput_change = 100 * (generation - 1)
            lines.append(
                f"Контекст {context}, {tag}: linear / softmax по скорости обучения = {train:.3f}×, "
                f"по скорости генерации = {generation:.3f}×. "
                f"Время обучения {'меньше' if train_change >= 0 else 'больше'} на {abs(train_change):.1f}%; "
                f"throughput генерации {'выше' if throughput_change >= 0 else 'ниже'} на {abs(throughput_change):.1f}%."
            )
    lines += [
        "", "На сравнительном графике 1× означает одинаковую скорость linear и softmax. "
        "Выше 1× — linear быстрее, ниже 1× — linear медленнее. "
        "Каждый вариант сравнивается с softmax при том же контексте и том же наличии RoPE.",
        "Скорость обучения = время softmax / время linear. "
        "Скорость генерации = throughput linear / throughput softmax = latency softmax / latency linear.",
        "", "2. ВЫВОД", "",
        "В эксперименте attention_large_50k (контекст 512) RoPE улучшает качество как softmax, "
        "так и linear attention. Softmax сохраняет преимущество по качеству: test PPL "
        f"{f(short['quality'][('softmax', False)]['perplexity'], 2)} против "
        f"{f(short['quality'][('linear', False)]['perplexity'], 2)} без RoPE и "
        f"{f(short['quality'][('softmax', True)]['perplexity'], 2)} против "
        f"{f(short['quality'][('linear', True)]['perplexity'], 2)} с RoPE.",
        "В конфигурации attention_large_context_1k (контекст 8192) linear выигрывает по скорости: "
        f"время обучения на батч меньше на {100 * (1 - 1 / speedup(long, False, 'training_ms_per_batch')):.1f}% "
        f"без RoPE и на {100 * (1 - 1 / speedup(long, True, 'training_ms_per_batch')):.1f}% с RoPE; "
        f"генерация выдаёт на {100 * (speedup(long, False, 'generation_tokens_per_second') - 1):.1f}% "
        f"и {100 * (speedup(long, True, 'generation_tokens_per_second') - 1):.1f}% больше новых токенов/с соответственно. "
        "При контексте 512 наша реализация linear медленнее softmax. "
        "Таким образом, линейная сложность по длине последовательности не гарантирует ускорение на коротком контексте.",
        "На test для attention_large_context_1k RoPE также снижает PPL: "
        f"softmax {f(long['quality'][('softmax', False)]['perplexity'], 2)} → "
        f"{f(long['quality'][('softmax', True)]['perplexity'], 2)}, "
        f"linear {f(long['quality'][('linear', False)]['perplexity'], 2)} → "
        f"{f(long['quality'][('linear', True)]['perplexity'], 2)}. "
        "Без RoPE качество двух attention-вариантов близко; с RoPE softmax показывает меньший PPL. "
        "Это результаты одного запуска с небольшим бюджетом обучения (1000 батчей).",
        "Из этих двух запусков нельзя вывести чистый эффект увеличения контекста: "
        "меняются также ширина модели, batch size и бюджет обучения. "
        "Нормированные столбцы показывают относительный выигрыш внутри каждой конфигурации, "
        "а не устраняют эти различия. PPL между двумя конфигурациями также не является изолированной оценкой влияния контекста.",
        "", "3. КОНФИГУРАЦИИ И МЕТОДИКА", "",
    ]
    for bundle in bundles:
        cfg = bundle["config"]
        budget = bundle["speed"][VARIANTS[0]]["training_batches"]
        lines.append(f"{bundle['runs'].name}: контекст {cfg['seq_length']}, d_model={cfg['d_model']}, "
                     f"слоёв={cfg['num_layers']}, голов={cfg['num_heads']}, batch={cfg['batch_size']}, "
                     f"AMP={cfg['dtype']}, chunk={cfg['linear_chunk_size']}, seed={cfg['seed']}, "
                     f"vocab={cfg['vocab_size']}; {budget} обучающих батчей, 10 эпох.")
    lines += [
        f"GPU: {short['environment'].get('device_name', 'не указан')}; "
        f"PyTorch: {short['environment'].get('torch_version', 'не указан')}.",
        "Linear attention core вычисляется в float32 внутри обучения/бенчмарка с float16 AMP.",
        "Среднее время обучения: время до train/epoch минус записанные интервалы validation/test, "
        "делённое на число обработанных батчей. Включены optimizer, метрики, логирование и расходы между эпохами; "
        "подготовка данных исключена. Это среднее время обработанного батча, включая AMP-skipped updates.",
        "Генерация: отдельный бенчмарк со случайными весами и синтетическими входами; "
        "64 новых токена на последовательность, фиксированное окно контекста; KV/recurrent cache не используется. "
        "Latency — время генерации всех 64 токенов всего batch, throughput — новые токены всего batch в секунду. "
        "Это измерение вычислительной скорости, не качества сгенерированного текста обученных моделей.",
        "Validation: до 50 батчей на эпоху; loss и метрики приведены из строк val/epoch. "
        "Train: полные средние train/epoch, а не последнее окно логирования.",
        "Для reference-графиков training_curves_512 ранее использовалось последнее train-окно эпохи; "
        "train/epoch в этом отчёте может отличаться от этих train-кривых. Validation loss совпадает.",
        "PPL = exp(NLL); NLL агрегируется с весами по числу непустых target-токенов. "
        "Loss и NLL могут слегка различаться из-за численной точности.",
        "Один seed; разброс между независимыми обучениями не оценивался.",
        "", "4. ВСЕ ТОЧКИ ОСНОВНОГО ГРАФИКА", "", "4.1 Validation loss: attention_large_50k", "",
    ]
    table(lines, ["Эпоха", "Батчи", *NAMES.values()], [
        [epoch, short["val"][VARIANTS[0]][epoch - 1]["step"],
         *(f(short["val"][key][epoch - 1]["loss"], 12) for key in VARIANTS)]
        for epoch in range(1, 11)
    ])
    lines += ["4.2 Относительная скорость: все столбцы сравнительного графика", ""]
    table(lines, ["Контекст", "RoPE", "Обучение, ×", "Генерация, ×"], [
        [bundle["config"]["seq_length"], "да" if rope else "нет",
         f(speedup(bundle, rope, "training_ms_per_batch"), 12),
         f(speedup(bundle, rope, "generation_tokens_per_second"), 12)]
        for bundle in bundles for rope in (False, True)
    ])
    lines += ["5. АБСОЛЮТНЫЕ ЗНАЧЕНИЯ СКОРОСТИ: ВСЕ СТОЛБЦЫ ДОПОЛНИТЕЛЬНОГО ГРАФИКА", ""]
    table(lines, ["Контекст", "Вариант", "Батчи", "Обучение, с", "Обучение, мс/батч", "Генерация, мс", "Новых токенов/с", "Std latency, мс"], [
        [bundle["config"]["seq_length"], NAMES[key], row["training_batches"],
         *(f(row[field], 9) for field in ("training_seconds", "training_ms_per_batch", "generation_mean_ms", "generation_tokens_per_second", "generation_stdev_ms"))]
        for bundle in bundles for key, row in bundle["speed"].items()
    ])
    lines += ["Std latency — стандартное отклонение повторов синтетического бенчмарка, не разброс обучений.", "", "6. PPL И ДРУГИЕ МЕТРИКИ ПО ЭПОХАМ", ""]
    for bundle in bundles:
        lines += [f"6. Контекст {bundle['config']['seq_length']}: {bundle['runs'].name}", ""]
        table(lines, ["Эпоха", "Батчи", "Вариант", "Train loss", "Train PPL", "Train accuracy, %", "Val loss", "Val NLL", "Val PPL", "Val accuracy, %"], [
            [epoch, val["step"], NAMES[key], f(train["loss"], 9), f(train["perplexity"], 9),
             f(100 * float(train["token_accuracy"]), 6), f(val["loss"], 9), f(val["nll"], 9),
             f(val["perplexity"], 9), f(100 * float(val["token_accuracy"]), 6)]
            for epoch in range(1, 11) for key in VARIANTS
            for train, val in [(bundle["train"][key][epoch - 1], bundle["val"][key][epoch - 1])]
        ])
    lines += ["7. ЛУЧШИЙ VALIDATION LOSS И СООТВЕТСТВУЮЩИЙ PPL", ""]
    table(lines, ["Контекст", "Вариант", "Эпоха", "Val loss", "Val PPL"], [
        [bundle["config"]["seq_length"], NAMES[key], best["epoch"], f(best["loss"], 9), f(best["perplexity"], 9)]
        for bundle in bundles for key in VARIANTS
        for best in [min(bundle["val"][key], key=lambda row: float(row["loss"]))]
    ])
    lines += ["8. TEST: NLL, PPL И ACCURACY ДЛЯ MODEL_BEST", ""]
    for bundle in bundles:
        meta = bundle["quality_metadata"]
        lines += [f"{bundle['runs'].name}: {bundle['quality_source'].as_posix()}; "
                  f"split={meta['split']}, dtype={meta['dtype']}, "
                  f"limit_stories={meta['dataset_config'].get('limit_stories')}, "
                  f"max_batches={meta.get('max_batches')}.", ""]
        table(lines, ["Вариант", "Эпоха checkpoint", "Test NLL", "Test PPL", "Test accuracy, %", "Target-токены", "Батчи оценки"], [
            [NAMES[key], row["epoch"], f(row["nll"], 12), f(row["perplexity"], 12),
             f(100 * float(row["token_accuracy"]), 9), int(row["evaluated_tokens"]), row["evaluated_batches"]]
            for key, row in bundle["quality"].items()
        ])
    lines += [
        "Test для 512 — весь сохранённый test split; test для 8192 — первые 1000 историй. "
        "Для каждого контекста все четыре варианта оценены на одинаковых target-токенах. "
        "Размер test-выборки между контекстами отличается.",
        "Test для attention_large_context_1k дополнительно рассчитан для всех четырёх model_best "
        "в float32, batch=4, limit_stories=1000, без ограничения числа батчей.",
        "", "9. ИСТОЧНИКИ И ВОСПРОИЗВЕДЕНИЕ", "",
    ]
    for bundle in bundles:
        lines.append(f"Запуски: {bundle['runs'].as_posix()}")
        lines.extend(f"  {bundle['histories'][key]['source']}" for key in VARIANTS)
        lines.append(f"Generation benchmark: {bundle['generation_source'].as_posix()}")
        lines.append(f"Test quality: {bundle['quality_source'].as_posix()}")
    lines += [
        "", "Повторное построение: .venv\\Scripts\\python.exe plot_linear_attention_poster.py",
        f"Численные значения всех {len(points)} точек/столбцов обоих рисунков с полной машинной точностью: "
        "linear_attention_comparison_points.csv. В этом текстовом файле значения округлены до 6–12 знаков после запятой.",
        "Основной рисунок: linear_attention_comparison.svg / .png.",
        "Дополнительный рисунок с абсолютными значениями: linear_attention_speed_absolute.svg / .png.",
    ]
    (output_dir / "linear_attention_comparison_data.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-512", type=Path, default=Path("saved/attention_large_50k"))
    parser.add_argument("--runs-8192", type=Path, default=Path("saved/attention_large_context_1k"))
    parser.add_argument("--generation-512", type=Path, default=Path("outputs/attention_inference_50k.json"))
    parser.add_argument("--generation-8192", type=Path, default=Path("outputs/attention_inference_large_context.json"))
    parser.add_argument("--quality-512", type=Path, default=Path("outputs/attention_quality_large_50k.json"))
    parser.add_argument("--quality-8192", type=Path, default=Path("outputs/attention_quality_large_context_1k_poster.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    bundles = [
        load_bundle(args.runs_512, args.generation_512, args.quality_512, 512),
        load_bundle(args.runs_8192, args.generation_8192, args.quality_8192, 8192),
    ]
    if bundles[0]["quality_metadata"]["tokenizer_sha256"] != bundles[1]["quality_metadata"]["tokenizer_sha256"]:
        raise ValueError("Test evaluations use different tokenizers")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(bundles, args.output_dir)
    plot_absolute(bundles, args.output_dir)
    points = export_points(bundles, args.output_dir)
    write_report(bundles, points, args.output_dir)
    print(f"Saved two SVG/PNG figures, {len(points)} plot points, and the Russian report to {args.output_dir}")


if __name__ == "__main__":
    main()
