import argparse
import math
from pathlib import Path

import torch
from hydra.utils import instantiate


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, device):
    if not prompt:
        raise ValueError("Prompt must not be empty.")
    unknown = sorted(set(prompt) - set(tokenizer.token_to_id))
    if unknown:
        raise ValueError(f"Prompt contains characters outside the vocabulary: {unknown!r}")

    tokens = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)],
        dtype=torch.long,
        device=device,
    )
    generated = []
    model.eval()
    for _ in range(max_new_tokens):
        context = tokens[:, -model.block_size:]
        logits = model(data_object=context)["logits"][:, -1, :]
        if temperature == 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            next_token = torch.multinomial(logits.softmax(dim=-1), num_samples=1)

        generated.append(next_token.item())
        tokens = torch.cat([context, next_token], dim=1)[:, -model.block_size:]

    return prompt + tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("saved/transformer_bpe/model_best.pth"))
    parser.add_argument("--vocab", type=Path, help="Override the saved vocabulary path")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature; 0 selects the most likely token")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and non-negative")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top-k must be positive")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    tokenizer = instantiate(config.tokenizer.instance)
    vocab_path = args.vocab or Path(config.tokenizer.vocab_path)
    if not vocab_path.is_absolute():
        vocab_path = Path(__file__).resolve().parent / vocab_path
    tokenizer.load_vocab(vocab_path)

    if config.model.vocab_size is None:
        config.model.vocab_size = tokenizer.vocab_size
    if config.model.vocab_size != tokenizer.vocab_size:
        raise ValueError("The vocabulary size does not match the trained model.")
    model = instantiate(config.model)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    print(generate(
        model, tokenizer, args.prompt, args.max_new_tokens,
        args.temperature, args.top_k, device,
    ))


if __name__ == "__main__":
    main()
