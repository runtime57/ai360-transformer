# J-lens на TinyStories GPT

J-lens (Jacobian lens): `J_l = E[∂h_final,t' / ∂h_l,t]` по `t' ≥ t`, чтение `softmax(W_U · LN_final(J_l h_l))`.
Logit lens — частный случай `J_l = I`.

```bash
# распаковать src/saved/tinystory_model(1).zip в DIR
python fit.py --ckpt DIR/saved/tinystory_gpt/model_best.pth --vocab DIR/data/tokenizer/bpe/tinystory.json --out J.pt   # ~1 ч на CPU
python run_experiments.py --ckpt ... --vocab ... --J J.pt   # -> figures/*.png, figures/results.json
```

`core.py` содержит собственный byte-level BPE (без пакета `tokenizers`), загрузку модели и оценку J.
J оценён на 48 окнах по 64 токена из val.txt (строки 0–1199); эксперименты считаются на отложенных строках 1200+.
