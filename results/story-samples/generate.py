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
prompts = [
    ("Kotletka and the adventure", config.text),
    ("The moon in a bucket", "Once upon a time, a little girl named Kotletka found the moon in a bucket of water. She wanted to take it home."),
    ("The dragon who was afraid of fire", "Once upon a time, there was a little dragon who was afraid of fire. His best friend was a tiny mouse."),
    ("A letter from tomorrow", "One day, a boy named Tim found a letter in his pocket. It said, \"Do not open the blue door tomorrow.\""),
    ("The cat and the robot", "Once upon a time, a little robot wanted to learn how to be a cat. He asked a big orange cat to help him."),
    ("The cloud in the kitchen", "One morning, Kotletka woke up and found a small cloud in the kitchen. The cloud was crying."),
    ("The secret of the lost sock", "Once upon a time, a little boy found a tiny door inside his lost sock. He opened it and saw"),
    ("The AI360 students", "Once upon a time, there were three AI360 students named Tom, Lily and Ben. They wanted to teach a little robot how to tell stories."),
    ("The Yandex data center", "Once upon a time, the AI360 students visited a Yandex data center. There were many computers and loud fans. Suddenly, all the lights went out."),
    ("The sleepy Yandex computer", "Once upon a time, in a Yandex data center, a little computer felt very sleepy. Two AI360 students came to help it."),
]
metadata = {
    "config": OmegaConf.to_container(config, resolve=True),
    "effective_seed": seed,
    "device": device,
    "checkpoint_epoch": checkpoint.get("epoch"),
    "model_config": OmegaConf.to_container(model_config, resolve=True),
    "settings": {k: getattr(model, k) for k in ("beam_width", "beam_temperature", "use_prob_sampling", "top_p", "context_len", "max_new_tokens")},
    "gpt_source_sha256": hashlib.sha256((ROOT / "src/model/gpt.py").read_bytes()).hexdigest(),
    "tokenizer_sha256": hashlib.sha256((ROOT / vocab_path).read_bytes()).hexdigest(),
    "samples": [],
}
print(json.dumps({k: metadata[k] for k in ("device", "effective_seed", "settings")}), flush=True)
for title, text in prompts:
    set_random_seed(seed)
    prompt_tokens = [tokenizer.token_to_id["<bos>"]] + tokenizer.encode(text, add_special_tokens=False)
    prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    started = time.monotonic()
    with torch.inference_mode():
        ids = model.predict(prompt)["seq"][0].tolist()
    sample = {
        "title": title,
        "prompt": text,
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
    print(json.dumps({k: sample[k] for k in ("title", "new_tokens", "ended_with_eos", "seconds")}), flush=True)
print("Saved " + str(OUT / "stories.md"), flush=True)
