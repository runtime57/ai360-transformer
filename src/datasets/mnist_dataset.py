import torch
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_tensor
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class MnistDataset(BaseDataset):
    def __init__(
        self,
        name="train",          # train / val / test
        val_size=5000,
        seed=42,
        download=True,
        *args,
        **kwargs,
    ):
        data_path = ROOT_PATH / "data" / "mnist" / name
        index_path = data_path / "index.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(
                name=name,
                val_size=val_size,
                seed=seed,
                download=download,
            )

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name, val_size, seed, download):
        mnist_root = ROOT_PATH / "data" / "mnist"

        if name == "test":
            dataset = MNIST(root=str(mnist_root), train=False, download=download)
            indices = list(range(len(dataset)))
        else:
            dataset = MNIST(root=str(mnist_root), train=True, download=download)
            generator = torch.Generator().manual_seed(seed)
            permutation = torch.randperm(len(dataset), generator=generator).tolist()

            if name == "train":
                indices = permutation[val_size:]
            elif name == "val":
                indices = permutation[:val_size]
            else:
                raise ValueError(f"Unknown name: {name}")

        data_path = mnist_root / name
        data_path.mkdir(parents=True, exist_ok=True)
        index = []
        width = max(5, len(str(len(indices))))
        for i, idx in enumerate(tqdm(indices, desc=f"Preparing MNIST {name}")):
            image, label = dataset[idx]
            image_tensor = to_tensor(image)
            image_path = data_path / f"{i:0{width}d}.pt"
            torch.save(image_tensor, image_path)
            index.append({
                "path": str(image_path),
                "label": int(label),
            })

        write_json(index, str(data_path / "index.json"))
        return index