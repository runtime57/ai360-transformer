import json
import logging
from collections.abc import Iterator
from pathlib import Path

from hydra.utils import instantiate, to_absolute_path

logger = logging.getLogger(__name__)


def _iter_texts(corpus_path: Path, text_field: str | None) -> Iterator[str]:
    """Yield texts from a plain-text or JSONL corpus."""
    if corpus_path.suffix.lower() == ".jsonl":
        if not text_field:
            raise ValueError("tokenizer.text_field is required for a JSONL corpus.")

        with corpus_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get(text_field)
                if not isinstance(text, str):
                    raise ValueError(
                        f"Expected a string in field '{text_field}' at "
                        f"{corpus_path}:{line_number}."
                    )
                yield text
        return

    with corpus_path.open("r", encoding="utf-8") as file:
        for line in file:
            # line = line.rstrip("\n\r")
            if line:
                yield line


def init_tokenizer(config, allow_build: bool = True, log=None):
    """Instantiate a tokenizer and either load or build its vocabulary.

    Args:
        config (DictConfig | None): Hydra tokenizer config. It contains the
            tokenizer ``instance`` config and lifecycle paths.
        allow_build: If False, a missing vocabulary is an error. Inference
            uses this mode to avoid learning from evaluation data.
        log (Logger | None): Logger used for lifecycle messages.
    Returns:
        BaseTokenizer | None: initialized tokenizer, or None when the config
            is absent or explicitly disabled.
    """
    if config is None or not config.get("enabled", True):
        return None

    active_logger = log or logger
    tokenizer = instantiate(config.instance)
    vocab_path = Path(to_absolute_path(config.vocab_path))

    if vocab_path.is_file():
        tokenizer.load_vocab(vocab_path)
        active_logger.info("Loaded tokenizer vocabulary from %s", vocab_path)
        return tokenizer

    if not allow_build:
        raise FileNotFoundError(
            f"Tokenizer vocabulary was not found at '{vocab_path}'. "
            "Build it during training first."
        )

    corpus_path = Path(to_absolute_path(config.train_corpus_path))
    if not corpus_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer training corpus was not found at '{corpus_path}'."
        )

    active_logger.info("Building tokenizer vocabulary from %s", corpus_path)
    tokenizer.train(_iter_texts(corpus_path, config.get("text_field")))
    tokenizer.save_vocab(vocab_path)
    active_logger.info("Saved tokenizer vocabulary to %s", vocab_path)
    return tokenizer

