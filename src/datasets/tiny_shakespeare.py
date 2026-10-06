import torch
from datasets import load_dataset

from src.datasets.base_dataset import BaseDataset


class TinyShakespeareDataset(BaseDataset):
    def __init__(
        self,
        block_size=256,
        split="train",
        tokenizer=None,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
        cache_dir=None
    ):
        self.block_size = block_size
        self.split = split
        self._limit = limit
        self._shuffle_index = shuffle_index
        dataset = load_dataset(
            "karpathy/tiny_shakespeare",
            revision="refs/pr/2",
            split=split,
            cache_dir=cache_dir,
        )
        self.text = "".join(dataset["text"])
        self._tokens = None
        super().__init__([], instance_transforms=instance_transforms)
        if tokenizer is not None:
            self.set_tokenizer(tokenizer)

    def set_tokenizer(self, tokenizer):
        tokens = torch.tensor(tokenizer.encode(self.text, add_special_tokens=False), dtype=torch.long)
        index = [
            {"start": start}
            for start in range(0, len(tokens) - self.block_size, self.block_size)
        ]
        self._index = self._shuffle_and_limit_index(index, self._limit, self._shuffle_index)
        self._tokens = tokens
        super().set_tokenizer(tokenizer)

    def __getitem__(self, ind):
        start = self._index[ind]["start"]
        end = start + self.block_size
        return self.preprocess_data(
            {
                "data_object": self._tokens[start:end].clone(),
                "labels": self._tokens[start + 1 : end + 1].clone(),
            }
        )

