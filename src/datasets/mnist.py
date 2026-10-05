from pathlib import Path

import torch
from torchvision.datasets import MNIST

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class MNISTDataset(BaseDataset):
    def __init__(
        self,
        split="train",
        root=None,
        download=True,
        val_size=5000,
        split_seed=42,
        **kwargs,
    ):
        root = ROOT_PATH / "data" / "mnist"
        dataset = MNIST(root=str(root), train=(split != "test"), download=download)
        self.data = dataset.data

        if split == "test":
            indices = range(len(dataset))
        else:
            generator = torch.Generator().manual_seed(split_seed)
            permutation = torch.randperm(len(dataset), generator=generator).tolist()
            indices = (permutation[val_size:] if split == "train" else permutation[:val_size])

        index = [{"path": i, "label": int(dataset.targets[i])} for i in indices]
        super().__init__(index, **kwargs)

    def load_object(self, path):
        return self.data[path].unsqueeze(0).to(torch.float32).div(255)
