```
.\.venv\Scripts\python.exe train.py -m -cn train_attention_quick `
  model=gpt,gpt_rope,gpt_linear,gpt_linear_rope `
  model.decoder.d_model=128 model.decoder.num_layers=6 `
  model.decoder.num_heads=8 `
  max_seq_len=8192 dataloader.batch_size=4 `
  trainer.epoch_len=100 trainer.epochs=1 `
  datasets.train.limit_stories=1000000 `
  trainer.save_dir=saved/attention_large_context_100 `
  writer=wandb writer.project_name=ai360-transformer
```

```
.\.venv\Scripts\python.exe evaluate_lm.py `
  saved/attention_large_context_100/tiny_stories_softmax_ropeFalse_seed42/model_best.pth `
  saved/attention_large_context_100/tiny_stories_softmax_ropeTrue_seed42/model_best.pth `
  saved/attention_large_context_100/tiny_stories_linear_ropeFalse_seed42/model_best.pth `
  saved/attention_large_context_100/tiny_stories_linear_ropeTrue_seed42/model_best.pth `
  --device cuda --batch-size 4 `
  --output outputs/attention_large_context_100.json
```

```
.\.venv\Scripts\python.exe plot_attention_results.py `
  --quality outputs/attention_large_context_100.json `
  --runs-dir saved/attention_large_context_100 `
  --train-points last-window --x-axis epoch `
  --output-dir outputs/attention_large_context_100
```