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
OUT.mkdir(parents=True, exist_ok=True)
config = OmegaConf.load(ROOT / "src/configs/test_drive.yaml")
checkpoint_path = ROOT / config.checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
checkpoint_config = checkpoint.get("config")
if checkpoint_config is None:
    checkpoint_config = OmegaConf.load(checkpoint_path.with_name("config.yaml"))
seed = config.seed if config.seed is not None else checkpoint_config.trainer.seed
device = config.device
if device == "auto":
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = instantiate(checkpoint_config.tokenizer.instance)
vocab_path = config.get("tokenizer_vocab") or checkpoint_config.tokenizer.vocab_path
tokenizer.load_vocab(ROOT / vocab_path)
model_config = OmegaConf.merge(
    OmegaConf.to_container(checkpoint_config.model, resolve=True, throw_on_missing=True),
    OmegaConf.to_container(config.model_overrides, resolve=True, throw_on_missing=True),
)
set_random_seed(seed)
model = instantiate(model_config)
model._set_tokenizer(tokenizer)
model._set_context(vocab_size=tokenizer.vocab_size, max_seq_len=checkpoint_config.datasets.train.max_seq_len)
model.load_state_dict(checkpoint["state_dict"])
model.to(device).eval()
metadata = json.loads((OUT / "stories.json").read_text())
assert metadata["settings"] == {k: getattr(model, k) for k in metadata["settings"]}
metadata["samples"][0]["title"] = "The shiny rock"
for item in metadata["samples"]:
    item.setdefault("seed", metadata["effective_seed"])
prompts = [
    ("The forbidden red button", "Once upon a time, three students from AI360 went to a Yandex data center. Tom brought a little robot. Lily brought a sandwich. Ben brought a big red button. The robot said, \"Do not press that button!\"", 42),
    ("The forbidden red button, another version", "Once upon a time, three students from AI360 went to a Yandex data center. Tom brought a little robot. Lily brought a sandwich. Ben brought a big red button. The robot said, \"Do not press that button!\"", 123),
    ("The data center sneezes", "One day, the AI360 students were working in a Yandex data center. Suddenly, a big computer sneezed. \"Oh no!\" said Tom. \"Our computer has a cold!\"", 77),
    ("The runaway shadow", "One morning, a little girl named Lily woke up and saw that her shadow was gone. She found it outside, playing with a dog.", 123),
    ("The dragon's birthday cake", "Once upon a time, a little dragon wanted to bake a birthday cake for his best friend. But every time he sneezed, the cake caught fire.", 77),
    ("The shoes that did not want to walk", "Once upon a time, a boy named Ben had a pair of red shoes. One day, the shoes said, \"We do not want to walk anymore!\"", 42),
    ("The tiny king in a teapot", "One day, a little girl found a tiny king living in her teapot. The king said, \"Please do not pour hot water on my house!\"", 123),
    ("The moon's letter", "Once upon a time, the moon sent a letter to a little girl. It said, \"I am tired of being up here. Can I come to your birthday party?\"", 77),
]
for title, text, sample_seed in prompts:
    set_random_seed(sample_seed)
    prompt_tokens = [tokenizer.token_to_id["<bos>"]] + tokenizer.encode(text, add_special_tokens=False)
    prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    started = time.monotonic()
    with torch.inference_mode():
        ids = model.predict(prompt)["seq"][0].tolist()
    sample = {
        "title": title, "prompt": text, "seed": sample_seed,
        "decoded_prompt": tokenizer.decode(prompt_tokens, skip_special_tokens=True),
        "text": tokenizer.decode(ids, skip_special_tokens=True),
        "new_tokens": len(ids) - len(prompt_tokens),
        "ended_with_eos": ids[-1] == tokenizer.token_to_id["<eos>"],
        "seconds": round(time.monotonic() - started, 2),
    }
    metadata["samples"].append(sample)
    (OUT / "stories.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    sections = []
    for item in metadata["samples"]:
        sections.append("## " + item["title"] + "\n\n" + item["text"] + "\n")
    (OUT / "stories.md").write_text("\n".join(sections))
    print(json.dumps({k: sample[k] for k in ("title", "seed", "new_tokens", "ended_with_eos", "seconds")}), flush=True)
print("Saved all " + str(len(metadata["samples"])) + " stories", flush=True)
