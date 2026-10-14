# Linear attention experiments

Запускать команды из корня репозитория, используя Python из установленного
окружения проекта. На Windows вместо `python` можно указать
`.venv/Scripts/python.exe`; на Linux/macOS — `.venv/bin/python`.

## Обучение

```bash
# Linear, sinusoidal positions, TinyStories.
python train_linear.py

# Linear + RoPE, без дополнительных sinusoidal positions.
python train_linear.py model=gpt_linear_rope

# Тот же pipeline без скрипта-обёртки.
python train.py --config-name=train_linear

# GPU и AMP включаются явно; accumulation внутри linear остаётся float32.
python train_linear.py trainer.device=cuda trainer.use_amp=True
```

`train_linear.py` использует текущий интерпретатор и передаёт все аргументы в
общий `train.py`. Он также поддерживает обычные флаги Hydra, например
`--cfg job --resolve` для просмотра итогового конфига без обучения.

По умолчанию: seed 42, 6 слоёв, `d_model=512`, 8 голов, context 512, 10 эпох,
dropout 0.1; AMP выключен. Размер batch, optimizer, schedule, tokenizer и split
наследуются из `train_tiny_stories`. Для ограниченного запуска:

```bash
python train_linear.py --config-name=train_attention_quick model=gpt_linear
python train_linear.py --config-name=train_attention_quick model=gpt_linear_rope
```

Quick: `d_model=256`, 4 слоя, context 256, batch 8, 3 × 200 training batches;
данные ограничены первыми 10 000 train и 1 000 val историями после split.
Validation проходит по всему этому ограниченному val-набору. Quick предназначен
для проверки запуска; его бюджет и метрики нужно отличать от полного эксперимента.
Лимит задаётся через `datasets.train.max_stories` / `datasets.val.max_stories`.
Прежнее имя `limit_stories` поддерживается для существующих checkpoints;
при использовании обоих имён значения должны совпадать. Кеш токенов учитывает лимит.
Для быстрой проверки самого конфига:

```bash
python train_linear.py --cfg job --resolve
```

Первый запуск может скачать TinyStories и обучить BPE. Последующие запуски
загружают vocabulary из `tokenizer.vocab_path`. Для сравнения моделей используйте
один и тот же сохранённый vocabulary и не включайте `tokenizer.override`.
Для воспроизведения существующего запуска укажите его файл vocabulary явно.

## Сравнение четырёх вариантов

```bash
python train.py --multirun --config-name=train_attention_compare model=gpt,gpt_rope,gpt_linear,gpt_linear_rope trainer.seed=42,43,44
```

Все варианты используют одинаковые данные и бюджет. По умолчанию внешнее
логирование отключено. Для одиночного запуска можно выбрать `writer=wandb`.
Результаты сохраняются в `saved/attention_compare/`, quick — в
`saved/attention_quick/`; имя запуска содержит attention, RoPE и seed.
При изменении размера модели или бюджета задайте отдельный `writer.run_name`.

Trainer сохраняет `config.yaml`, `git_commit.txt`, `git_diff.patch` и checkpoints;
checkpoint содержит конфиг модели, optimizer и scheduler. Сохраните также файл
tokenizer, версию исходных данных и версии Python/PyTorch/CUDA. Для исторических
результатов из `EG2107/transformer6` используйте параметры и данные конкретного
checkpoint: новые конфиги являются явными рецептами запуска, а не заменой
потерянных overrides старого эксперимента.

## Оценка качества

```bash
python evaluate_lm.py saved/attention_compare/tiny_stories_linear_ropeFalse_seed42/model_best.pth --split test --device auto --output outputs/linear_quality.json
```

Можно передать несколько checkpoints одной командой. Скрипт загружает модель
из её конфига, проверяет hash vocabulary и согласованность validation-конфигов.
`val` — отложенная часть training split, `test` — официальный validation split
TinyStories. При необходимости первой загрузки добавьте `--allow-download`.
Для проверки используйте `--max-batches 2 --limit-stories 16`.

NLL и accuracy усредняются по непустым target-токенам; perplexity равна
`exp(corpus NLL)`. Скрипт пишет JSON и CSV. Метрики обучения в общем trainer
сохраняют поведение `main`; для сравнения качества используйте эту отдельную
оценку на одинаковом held-out наборе.
Для Multitoken оценивается первая, next-token голова на том же наборе входов;
веса дополнительных горизонтов относятся к training loss и не смешиваются с NLL.

Linear работает и с общим `inference.py`. Для checkpoint полного рецепта:

```bash
python inference.py model=gpt_linear inferencer.from_pretrained=saved/attention_compare/tiny_stories_linear_ropeFalse_seed42/model_best.pth tokenizer.vocab_path=data/tokenizer/bpe/tiny_stories.json
```

Для quick/custom модели также передайте её context, размер и число слоёв.

## Скорость и память

```bash
python benchmark_attention.py --device cuda --dtype float16 --component both --seq-lengths 128 512 1024 --output outputs/attention_benchmark.json
```

Benchmark сравнивает softmax, softmax+RoPE, linear и linear+RoPE на искусственных
входах с одинаковым seed. Отдельно измеряет forward и forward+backward, сохраняет
параметры окружения, throughput и CUDA memory в JSON/CSV. Для короткой CPU-проверки:

```bash
python benchmark_attention.py --device cpu --component both --seq-lengths 7 16 --d-model 16 --num-heads 2 --num-layers 2 --vocab-size 32 --batch-size 2 --warmup 1 --iterations 2
```

`--generation-tokens 16` добавляет greedy generation с фиксированным окном.
KV/recurrent cache не используется; время генерации относится к полному
пересчёту окна. Synthetic benchmark не оценивает качество модели.

## Устройство реализации и перенос следующих методов

- `src/model/transformer_utils/linear_attention.py` содержит ядро без параметров;
  QKV, output projection, RoPE и gates остаются в `CausalSelfAttention`.
- `attention_type` по умолчанию `softmax`, linear включается конфигом. Имена
  параметров checkpoint и выход `GPT.forward()['logits']` сохраняются.
- Feature map — `ELU(x)+1`. При linear+RoPE вращение применяется **после** feature
  map только к числителю; знаменатель использует невращённые положительные
  features. Dropout применяется к V. Это сохраняет вычисления исходного linear
  эксперимента; менять порядок операций нужно как отдельную модификацию.
- `linear_chunk_size` управляет размером локальных causal matrices, а
  `linear_eps` — нижней границей знаменателя. Глобальная матрица `T × T` не
  создаётся. При фиксированном размере chunk память растёт линейно по T.
- Headwise/elementwise gates применяются к результату attention перед output
  projection, как в `main`. Их комбинация с linear технически поддерживается,
  но требует отдельного эксперимента и отдельного имени запуска.
- Новые методы подключайте отдельными конфигами, сохраняя baseline defaults.
  ALiBi bias относится к softmax scores; его совместимость с linear нужно
  определить отдельно. Одновременные `attention_type=linear` и `use_alibi=True`
  отклоняются, чтобы bias не игнорировался. Telescopic и Multitoken используют
  свои головы/выходы и objective независимо от attention kernel.

Объединённые конфиги Telescopic и Multitoken можно использовать с linear:

```bash
python train.py --config-name=train_tlm model=gpt_linear_rope writer.run_name=telescopic_linear_rope
python train.py --config-name=train_mtp ++model.decoder.attention_type=linear writer.run_name=mtp_linear
```

В Multitoken сохраняйте `model=gpt_mtp`, чтобы число голов соответствовало
`n_future_tokens` в датасете. Эти комбинации проверены короткими тестами запуска;
для сравнения качества нужен отдельный полный эксперимент.

Проверки реализации, градиентов, causality, AMP и checkpoint layout:

```bash
python -m unittest test_linear_attention
```
