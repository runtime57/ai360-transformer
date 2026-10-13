# Сгенерированные истории TinyStories

18 пар: baseline и RoPE + elementwise gate. В каждой паре одинаковые prompt, seed и настройки. Все тексты сохранены без исправлений.

- comparison.html — автономное сравнение, выбор истории и скрытие названий моделей.
- stories.md и rope_elementwise_stories.md — все 36 историй с prompt и seed перед каждой.
- selected.md и rope_elementwise_selected.md — подборки по пять историй.
- stories.json и rope_elementwise_stories.json — исходные результаты: тексты, prompts, seeds, длины, EOS и время генерации. У gated сохранены также token_ids.
- prompts.json и prompts.csv — полный список 18 prompts и seeds.
- generation_settings.json — настройки, эффективные конфиги моделей, пути и SHA-256 чекпоинтов, хеши исходника и токенизатора.
- test_drive_frozen.json — использованный конфиг baseline, взятый из результатов генерации; JSON читается OmegaConf.
- tokenizer.json — использованный BPE-токенизатор с 4096 токенами.
- comparison_notes.md — наблюдения по качеству.
- generate.py, generate_more.py, generate_rope_elementwise.py — оригинальные скрипты генерации; запускались из корня репозитория и писали в work/story-samples. Первые два читают src/configs/test_drive.yaml; для повторения нужен сохранённый test_drive_frozen.json и соответствующие чекпоинты.
- render_comparison.py — сборка HTML, JSON-сравнения и Markdown из сохранённых результатов в этой папке. Изменён только путь к данным.
- sources/original_generation — неизменённые оригиналы четырёх скриптов.
- sources/repository — копии кода модели, токенизатора, настройки seed и файлов зависимостей. gpt.py совпадает по хешу с использованным при генерации; остальные файлы зависимостей скопированы при упаковке.
- manifest.json — размеры и SHA-256 файлов архива.

Настройки: beam sampling, beam_width=4, beam_temperature=0.8, top_p=0.9, context_len=512, max_new_tokens=2048. Seeds: 42, 123 и 77. Генерация выполнялась на MPS в float32, оба чекпоинта — epoch 10.

Чекпоинты остаются в saved; их SHA-256 проверены при упаковке. Версии среды в generation_settings.json отражают момент упаковки. Одинаковый seed не гарантирует побитового совпадения на другом устройстве или версии PyTorch.

Пересобрать представления без запуска модели из корня репозитория:

    python3 -B output/story-samples/render_comparison.py

Скрипт обновляет HTML, comparison.json и четыре Markdown-файла, сохраняя исходные stories.json и rope_elementwise_stories.json.
