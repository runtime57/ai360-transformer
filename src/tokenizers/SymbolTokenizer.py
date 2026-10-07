from src.tokenizers import BaseTokenizer
from typing import Iterable, Sequence


class SymbolTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__(special_tokens=['<pad>', '<unk>', '<bos>', '<eos>'])

    def train(self, texts: Iterable[str]) -> None:
        tokens = set()
        for text in texts:
            tokens = tokens.union(set(text))
        tokens = self.special_tokens + sorted(list(tokens))
        self._set_vocabulary({token: id for id, token in enumerate(tokens)})

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        unk = self.token_to_id['<unk>']
        tokens = [self.token_to_id.get(token, unk) for token in text]
        if add_special_tokens:
            bos, eos = self.token_to_id['<bos>'], self.token_to_id['<eos>']
            tokens = [bos] + tokens + [eos]
        return tokens

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        special_tokens_ids = [self.token_to_id[token] for token in self.special_tokens]
        text = [self.id_to_token[id] for id in token_ids
                if id not in special_tokens_ids or not skip_special_tokens]
        return ''.join(text)
