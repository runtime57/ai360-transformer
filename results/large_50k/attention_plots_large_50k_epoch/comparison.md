# Сравнение вариантов attention

Источники:

- `outputs/attention_quality_large_50k.json`
- `saved/attention_large_50k`

Качество на split `test`:

| Вариант | Seed | NLL | Perplexity | Accuracy, % | Токены |
| --- | ---: | ---: | ---: | ---: | ---: |
| softmax without RoPE | 42 | 1.5175 | 4.56 | 61.26 | 4892508 |
| softmax with RoPE | 42 | 1.4797 | 4.39 | 61.98 | 4892508 |
| linear without RoPE | 42 | 1.5996 | 4.95 | 59.61 | 4892508 |
| linear with RoPE | 42 | 1.5669 | 4.79 | 60.23 | 4892508 |

История обучения:

График: `train-points=last-window`, `perplexity-scale=linear`.
CSV сохраняет всю исходную историю, включая точки, не выбранные для графика.

- softmax without RoPE, seed 42: `saved\attention_large_50k\tiny_stories_softmax_ropeFalse_seed42\metrics.csv`; 512 / 6 слоёв / 8 голов, контекст 512, batch 8, epoch_len 5000, validation до 50 батчей.
- softmax with RoPE, seed 42: `saved\attention_large_50k\tiny_stories_softmax_ropeTrue_seed42\metrics.csv`; 512 / 6 слоёв / 8 голов, контекст 512, batch 8, epoch_len 5000, validation до 50 батчей.
- linear without RoPE, seed 42: `saved\attention_large_50k\tiny_stories_linear_ropeFalse_seed42\metrics.csv`; 512 / 6 слоёв / 8 голов, контекст 512, batch 8, epoch_len 5000, validation до 50 батчей.
- linear with RoPE, seed 42: `saved\attention_large_50k\tiny_stories_linear_ropeTrue_seed42\metrics.csv`; 512 / 6 слоёв / 8 голов, контекст 512, batch 8, epoch_len 5000, validation до 50 батчей.

training_time включает train и validation между началом цикла обучения и последней точкой истории; исключает подготовку данных. Для старых логов это время до печати последнего отчёта эпохи. CSV сохраняет scope и источник каждой точки.

Для этого набора данных сравнивайте одинаковые seed, длину контекста, batch и precision. Линейная сложность по длине не гарантирует меньшее время на коротком контексте. Один seed показывает результат одного запуска, а не разброс между обучениями.
