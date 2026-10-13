from src.tokenizers import BaseTokenizer
from typing import Any, Iterable, Sequence

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers, processors


class BpeTokenizer(BaseTokenizer):
    def __init__(self, vocab_size, min_frequency):
        super().__init__(special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])

        assert vocab_size >= 256 + len(self.special_tokens), f"vocab_size must be at least {256 + len(self.special_tokens)}"

        self.target_vocab_size = vocab_size
        self.min_frequency = min_frequency
        self._tokenizer = None


    def train(self, texts: Iterable[str]) -> None:
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=self.target_vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)

        tokenizer.post_processor = processors.TemplateProcessing(
            single="<bos> $A <eos>",
            special_tokens=[
                ("<bos>", tokenizer.token_to_id("<bos>")),
                ("<eos>", tokenizer.token_to_id("<eos>")),
            ],
        )

        self._tokenizer = tokenizer
        self._set_vocabulary(tokenizer.get_vocab())

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def _get_model_state(self) -> dict[str, Any]:
        return {
            "tokenizer_json": self._tokenizer.to_str(),
            "target_vocab_size": self.target_vocab_size,
            "min_frequency": self.min_frequency,
        }

    def _load_model_state(self, state: dict[str, Any]) -> None:
        tokenizer = Tokenizer.from_str(state["tokenizer_json"])

        assert tokenizer.get_vocab() == self.token_to_id, "BPE vocabulary does not match token_to_id."

        self.target_vocab_size = state["target_vocab_size"]
        self.min_frequency = state["min_frequency"]
        self._tokenizer = tokenizer
