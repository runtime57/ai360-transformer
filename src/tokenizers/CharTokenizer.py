from src.tokenizers.BaseTokenizer import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    def train(self, texts):
        unique_chars = set()
        for text in texts:
            unique_chars.update(text)
        self.token_to_id = {char: idx for idx, char in enumerate(sorted(unique_chars))}
        self.id_to_token = {idx: char for char, idx in self.token_to_id.items()}

    def encode(self, text, add_special_tokens = True):
        return [self.token_to_id[char] for char in text if char in self.token_to_id]

    def decode(self, token_ids, skip_special_tokens = True):
        return ''.join(self.id_to_token[token_id] for token_id in token_ids if token_id in self.id_to_token)
