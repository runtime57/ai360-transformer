import hashlib
import json
import time
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils.init_utils import set_random_seed

ROOT = Path.cwd()
OUT = ROOT / "work/story-samples"
baseline = json.loads((OUT / "stories.json").read_text())
checkpoint_path = ROOT / "saved/tiny_stories_rope_elementwise_gate/model_best.pth"
tokenizer_path = ROOT / baseline["config"]["tokenizer_vocab"]
source_hash = hashlib.sha256((ROOT / "src/model/gpt.py").read_bytes()).hexdigest()
assert source_hash == baseline["gpt_source_sha256"]
assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() == baseline["tokenizer_sha256"]
assert hashlib.sha256((ROOT / "data/tokenizer/bpe/tiny_stories.json").read_bytes()).hexdigest() == baseline["tokenizer_sha256"]
assert baseline["device"] == "mps" and torch.backends.mps.is_available()
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
checkpoint_config = checkpoint.get("config")
if checkpoint_config is None:
    checkpoint_config = OmegaConf.load(checkpoint_path.with_name("config.yaml"))
model_config = OmegaConf.merge(
    OmegaConf.to_container(checkpoint_config.model, resolve=True, throw_on_missing=True),
    baseline["settings"],
)
assert model_config.decoder.use_rope and model_config.decoder.use_elementwise_gate
assert model_config.decoder.mlp_hidden_dim == 1792
tokenizer = instantiate(checkpoint_config.tokenizer.instance)
tokenizer.load_vocab(tokenizer_path)
set_random_seed(baseline["effective_seed"])
model = instantiate(model_config)
model._set_tokenizer(tokenizer)
model._set_context(vocab_size=tokenizer.vocab_size, max_seq_len=checkpoint_config.datasets.train.max_seq_len)
model.load_state_dict(checkpoint["state_dict"], strict=True)
model.to(baseline["device"]).eval()
assert all(p.dtype == torch.float32 for p in model.parameters())
assert all(block.attention.use_rope and block.attention.gate.out_features == model.decoder.d_model for block in model.decoder.transformer_blocks)
assert {k: getattr(model, k) for k in baseline["settings"]} == baseline["settings"]
metadata = {
    "model_name": "RoPE + elementwise gate",
    "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    "baseline_checkpoint_sha256": hashlib.sha256((ROOT / baseline["config"]["checkpoint"]).read_bytes()).hexdigest(),
    "checkpoint_epoch": checkpoint.get("epoch"),
    "device": baseline["device"],
    "inference_dtype": "float32",
    "model_config": OmegaConf.to_container(model_config, resolve=True),
    "settings": baseline["settings"],
    "gpt_source_sha256": source_hash,
    "tokenizer_sha256": baseline["tokenizer_sha256"],
    "samples": [],
}
print(json.dumps({k: metadata[k] for k in ("model_name", "checkpoint_epoch", "device", "settings")}), flush=True)
for index, original in enumerate(baseline["samples"]):
    sample_seed = original.get("seed", baseline["effective_seed"])
    set_random_seed(sample_seed)
    prompt_tokens = [tokenizer.token_to_id["<bos>"]] + tokenizer.encode(original["prompt"], add_special_tokens=False)
    decoded_prompt = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
    assert decoded_prompt == original["decoded_prompt"]
    prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=baseline["device"])
    started = time.monotonic()
    with torch.inference_mode():
        ids = model.predict(prompt)["seq"][0].tolist()
    sample = {
        "title": original["title"],
        "prompt": original["prompt"],
        "decoded_prompt": decoded_prompt,
        "seed": sample_seed,
        "text": tokenizer.decode(ids, skip_special_tokens=True),
        "new_tokens": len(ids) - len(prompt_tokens),
        "ended_with_eos": ids[-1] == tokenizer.token_to_id["<eos>"],
        "seconds": round(time.monotonic() - started, 2),
        "token_ids": ids,
        "prompt_token_count": len(prompt_tokens),
    }
    assert sample["text"].startswith(decoded_prompt)
    metadata["samples"].append(sample)
    (OUT / "rope_elementwise_stories.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({"index": index + 1, **{k: sample[k] for k in ("title", "seed", "new_tokens", "ended_with_eos", "seconds")}}), flush=True)
assert len(metadata["samples"]) == len(baseline["samples"]) == 18
print("Completed all 18 matched prompts and seeds.", flush=True)
