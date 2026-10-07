from typing import override

import torch

from src.datasets.base_dataset import BaseDataset
from src.tokenizers.BaseTokenizer import BaseTokenizer
from src.utils.io_utils import ROOT_PATH

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = ROOT_PATH / "data" / "tinyshakespeare"


def download_tinyshakespeare(train_fraction=0.9):
    """
    Download the corpus and split it contiguously into train.txt / test.txt.

    train.txt is also the tokenizer corpus, and train.py builds the tokenizer
    before the datasets, so run this module once before the first training:
        python3 -m src.datasets.tinyshakespeare
    """
    import requests

    text = requests.get(URL, timeout=30).text
    split = int(len(text) * train_fraction)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "train.txt").write_text(text[:split], encoding="utf-8")
    (DATA_DIR / "test.txt").write_text(text[split:], encoding="utf-8")


class TinyShakespeareDataset(BaseDataset):
    def __init__(self, name='train', block_size=128, stride=1, *args, **kwargs):
        self.path = DATA_DIR / f"{name}.txt"
        self.name = name
        self.block_size = block_size
        self.stride = stride
        if not self.path.exists():
            download_tinyshakespeare()
        self.text = self.path.read_text(encoding='utf-8')
        super().__init__(*args, **kwargs)

    def set_tokenizer(self, tokenizer: BaseTokenizer) -> None:
        super().set_tokenizer(tokenizer)
        self.tokens = torch.tensor(self.tokenizer.encode(self.text, add_special_tokens=True), dtype=torch.long)

    @override
    def __len__(self):
        return (self.tokens.size(0) - self.block_size) // self.stride

    @override
    def __getitem__(self, ind):
        ind *= self.stride
        data = self.tokens[ind:ind + self.block_size + 1].long()
        return {
            "data_object": data[:-1],
            "labels": data[1:]
        }


if __name__ == "__main__":
    download_tinyshakespeare()
