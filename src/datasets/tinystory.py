import numpy as np
import torch
from tqdm.auto import tqdm

from datasets import load_dataset

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class TinyStoryDataset(BaseDataset):

    def __init__(
        self, name="train", *args, **kwargs
    ):
        """
        Args:
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / "tinystory" / name / "index.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.
        """
        index = []
        data_path = ROOT_PATH / "data" / "tinystory" / name
        data_path.mkdir(exist_ok=True, parents=True)
        ds = load_dataset("roneneldan/TinyStories")
        ds['train']
        if name == "train":
            ds = load_dataset("roneneldan/TinyStories", split="train")
            ds=ds.shuffle(seed=42)
            n=len(ds)
            n*=0.99
            n = int(n)
            ds = ds.select(range(n))  
        elif name == "val":
            ds = load_dataset("roneneldan/TinyStories", split="validation")
        else:
            ds = load_dataset("roneneldan/TinyStories", split="train")
            ds=ds.shuffle(seed=42)
            n=len(ds)
            n*=0.99
            n = int(n)
            ds = ds.select(range(n, len(ds)))  
        processed_ds = self._tokeinse(ds)
        for i, row in enumerate(processed_ds):
            path = data_path / f"{i}.pt"
            torch.save(torch.tensor(row), path)
            index.append({"path": str(path)})
        write_json(index, str(data_path / "index.json"))
        return index

    def _tokeinse(self, ds):
        vocab_path = ROOT_PATH / "data" / "tinystory" / "vocab.json"
        if vocab_path.exists():
            ch_to_int = read_json(str(vocab_path))
        else:
            chars = set()
            for batch in tqdm(ds.iter(batch_size=10_000), desc="vocab"):
                chars.update("".join(batch["text"]))
            ch_to_int = {c: i for i, c in enumerate(sorted(chars))}
            write_json(ch_to_int, str(vocab_path))
        for batch in tqdm(ds.iter(batch_size=10_000), desc="tokenize"):
            for text in batch["text"]:
                yield [ch_to_int.get(c, 0) for c in text]   # yield вместо res.append: не держим всё в памяти
            
