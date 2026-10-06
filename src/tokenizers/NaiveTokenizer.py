import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Sequence
from BaseTokenizer import BaseTokenizer


class NaiveTokenizer(BaseTokenizer):
    def train(self, texts: Iterable[str]):
        alphabet = " abcdefghijklmnopqrstuvwxyz\n"
        self.id_to_token[0] = '<BOS>'
        self.id_to_token[1] = '<EOS>'
        self.id_to_token[2] = '<PAD>'
        self.id_to_token[3] = '<UNK>'
        self.token_to_id['<BOS>'] = 0
        self.token_to_id['<EOS>'] = 1
        self.token_to_id['<PAD>'] = 2
        self.token_to_id['<UNK>'] = 3
        for i, c in enumerate(alphabet):
            self.token_to_id[str(c)] = i + 4
            self.id_to_token[i + 4] = str(c)

    
    def encode(self, text: str) -> list[int]:
        """Convert a string into token ids."""
        res = [0]
        alphabet = " abcdefghijklmnopqrstuvwxyz"
        for i in range(len(text)):
            if (text[i] in alphabet):
                res.append(self.token_to_id[text[i]])
            else:
                res.append(3)
        res.append(1)
        return res

    def decode(self, token_ids: Sequence[int]) -> str:
        """Convert token ids back into a string."""
        res = ""
        for i in token_ids:
            if (i >= 4):
                res += self.token_to_id[i]
            if (i == 3):
                res += "<UNK>"
        return res