from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils.init_utils import set_random_seed

ROOT = Path(__file__).resolve().parent


@hydra.main(version_base=None, config_path="src/configs", config_name="test_drive")
@torch.inference_mode()
def main(config):
    checkpoint_path = ROOT / config.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        checkpoint_config = OmegaConf.load(checkpoint_path.with_name("config.yaml"))

    seed = config.seed if config.seed is not None else checkpoint_config.trainer.seed
    set_random_seed(seed)
    device = config.device
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else
            "cpu"
        )

    tokenizer = instantiate(checkpoint_config.tokenizer.instance)
    vocab_path = config.get("tokenizer_vocab") or checkpoint_config.tokenizer.vocab_path
    tokenizer.load_vocab(ROOT / vocab_path)


    model_config = OmegaConf.merge(
        OmegaConf.to_container(checkpoint_config.model, resolve=True, throw_on_missing=True),
        OmegaConf.to_container(config.model_overrides, resolve=True, throw_on_missing=True),
    )

    model = instantiate(model_config)
    model._set_tokenizer(tokenizer)
    model._set_context(**{"vocab_size": tokenizer.vocab_size, "max_seq_len": checkpoint_config.datasets.train.max_seq_len})  # слегка костыль
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    prompt_tokens = [tokenizer.token_to_id["<bos>"]] + tokenizer.encode(config.text, add_special_tokens=False)
    prompt = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    print(f"Generating using {device}...", flush=True)
    predicted = model.predict(prompt)
    token_ids = predicted["seq"][0].tolist()
    print(tokenizer.decode(token_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
