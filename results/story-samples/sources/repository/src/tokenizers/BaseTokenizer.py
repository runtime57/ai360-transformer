import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Sequence


class BaseTokenizer(ABC):
    """Base interface for tokenizers used in the project.

    The base class owns the vocabulary serialization format. A concrete
    tokenizer only has to implement training, encoding, decoding, and, if
    needed, serialization of model-specific state.
    """

    STATE_VERSION = 1

    def __init__(self, special_tokens: Sequence[str] | None = None):
        """
        Args:
            special_tokens: Tokens such as ``<pad>``, ``<unk>``, ``<bos>``,
                and ``<eos>``. Their exact meaning is defined by a concrete
                tokenizer.
        """
        self.special_tokens = list(special_tokens or [])
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def vocab_size(self) -> int:
        """Return the actual number of tokens in the loaded vocabulary."""
        return len(self)

    @property
    def is_trained(self) -> bool:
        """Whether a vocabulary has already been built or loaded."""
        return bool(self.token_to_id)

    @abstractmethod
    def train(self, texts: Iterable[str]) -> None:
        """Build the tokenizer vocabulary from an iterable of texts."""

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Convert a string into token ids."""

    @abstractmethod
    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Convert token ids back into a string."""

    def save_vocab(self, path: str | Path) -> None:
        """Save the vocabulary and tokenizer-specific state as JSON."""
        if not self.is_trained:
            raise RuntimeError("Cannot save an empty vocabulary. Train the tokenizer first.")
        self._validate_vocabulary(self.token_to_id)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": self.STATE_VERSION,
            "token_to_id": self.token_to_id,
            "special_tokens": self.special_tokens,
            "model_state": self._get_model_state(),
        }

        temporary_path = path.with_suffix(path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)
        temporary_path.replace(path)

    def load_vocab(self, path: str | Path) -> None:
        """Load a vocabulary and tokenizer-specific state from JSON."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as file:
            state = json.load(file)

        version = state.get("version")
        if version != self.STATE_VERSION:
            raise ValueError(
                f"Unsupported tokenizer state version {version!r}; "
                f"expected {self.STATE_VERSION}."
            )

        self._set_vocabulary(state.get("token_to_id"))
        self.special_tokens = list(state.get("special_tokens", []))
        self._load_model_state(state.get("model_state", {}))

    def _set_vocabulary(self, token_to_id: dict[str, int]) -> None:
        """Set both vocabulary lookup tables after validating the mapping.

        Concrete tokenizers should call this method at the end of ``train``.
        """
        self._validate_vocabulary(token_to_id)
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {
            token_id: token for token, token_id in self.token_to_id.items()
        }

    def _get_model_state(self) -> dict[str, Any]:
        """Return extra serializable state required by a concrete tokenizer.

        Override this hook in a concrete tokenizer when its state is larger
        than the token-to-id mapping.
        """
        return {}

    def _load_model_state(self, state: dict[str, Any]) -> None:
        """Restore state returned by :meth:`_get_model_state`."""

    @staticmethod
    def _validate_vocabulary(token_to_id: Any) -> None:
        if not isinstance(token_to_id, dict) or not token_to_id:
            raise ValueError("Tokenizer state must contain a non-empty 'token_to_id' map.")

        token_ids = list(token_to_id.values())
        if any(not isinstance(token, str) for token in token_to_id):
            raise ValueError("All vocabulary tokens must be strings.")
        if any(not isinstance(token_id, int) or token_id < 0 for token_id in token_ids):
            raise ValueError("All vocabulary ids must be non-negative integers.")
        if len(set(token_ids)) != len(token_ids):
            raise ValueError("Vocabulary ids must be unique.")
