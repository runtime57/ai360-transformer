import urllib.request

import torch
from torch.utils.data import Dataset

from src.utils.io_utils import ROOT_PATH

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = ROOT_PATH / "data" / "tinyshakespeare"


def download_shakespeare():
    if (DATA_DIR / "train.txt").exists() and (DATA_DIR / "val.txt").exists():
        return

    print("Downloading Tiny Shakespeare...")
    with urllib.request.urlopen(URL) as response:
        text = response.read().decode()

    split = int(0.9 * len(text))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "train.txt").write_text(text[:split])
    (DATA_DIR / "val.txt").write_text(text[split:])


class ShakespeareDataset(Dataset):
    def __init__(self, name, block_size, stride):
        self.text = (DATA_DIR / f"{name}.txt").read_text()
        self.block_size = block_size
        self.stride = stride

    def set_tokenizer(self, tokenizer):
        self.tokens = torch.tensor(tokenizer.encode(self.text))

    def __len__(self):
        return (len(self.tokens) - self.block_size - 1) // self.stride + 1

    def __getitem__(self, ind):
        start = ind * self.stride
        chunk = self.tokens[start : start + self.block_size + 1]
        return {"data_object": chunk[:-1], "labels": chunk[1:]}
