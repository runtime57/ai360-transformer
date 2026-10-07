from src.tokenizers.BaseTokenizer import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    def train(self, texts):
        chars = {"\n"}
        for text in texts:
            chars.update(text)
        self._set_vocabulary({char: i for i, char in enumerate(sorted(chars))})

    def encode(self, text, add_special_tokens=True):
        return [self.token_to_id[char] for char in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self.id_to_token[token_id] for token_id in token_ids)
