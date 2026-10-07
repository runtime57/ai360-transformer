from collections.abc import Iterable, Sequence
import re
import string

from src.tokenizers.BaseTokenizer import BaseTokenizer


class LetterTokenizer(BaseTokenizer):
    def __init__(self, alph: str | None = None, special_tokens: Sequence[str] | None = None):
        self.alph = alph or string.ascii_letters + string.digits + ",;.!?() "
        super().__init__(special_tokens)

    def train(self, texts: Iterable[str]) -> None:
        # init_tokenizer feeds the corpus line by line with newlines stripped
        tokens = self.special_tokens + ["\n"] + [sym for text in texts for sym in text]
        token_to_id = {token: i for i, token in enumerate(sorted(set(tokens)))}
        self._set_vocabulary(token_to_id)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert self.is_trained
        tokens = [self.token_to_id[sym] for sym in text]
        if add_special_tokens:
            tokens = [self.token_to_id['<BOS>']] + tokens + [self.token_to_id['<EOS>']]
        return tokens

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        assert self.is_trained
        tokens = [self.id_to_token[i] for i in token_ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if not re.match(r'<.*>', t)]
        return ''.join(tokens)
