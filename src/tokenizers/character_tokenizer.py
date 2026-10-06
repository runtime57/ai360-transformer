from src.tokenizers import BaseTokenizer
from typing import Iterable, Sequence


class CharacterTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__(special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])

    def train(self, texts: Iterable[str]) -> None:
        characters = sorted(list(set(char for text in texts for char in text)))
        tokens = self.special_tokens + characters

        self._set_vocabulary({token: idx for idx, token in enumerate(tokens)})

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        unk_id = self.token_to_id["<unk>"]
        token_ids = [
            self.token_to_id.get(char, unk_id)
            for char in text
        ]

        token_ids = [self.token_to_id.get(char, unk_id) for char in text]

        if add_special_tokens:
            token_ids = [self.token_to_id["<bos>"], *token_ids, self.token_to_id["<bos>"]]

        return token_ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        excluded = set(self.special_tokens) if skip_special_tokens else {}
        tokens = [self.id_to_token[idx] for idx in token_ids]
        tokens_filtered = [token for token in tokens if token not in excluded]

        return "".join(tokens_filtered)
