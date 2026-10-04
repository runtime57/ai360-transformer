import shutil

import numpy as np
import safetensors
import safetensors.torch
import torchvision
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json

class MnistDatasset(BaseDataset):
    def __init__(self, name="test", *args, **kwargs):
        index_path = ROOT_PATH / "data" / "minst" / name / "index.json"

        if index_path.exists():
            index = read_json(index_path)
        else:
            index = self._create_index(name)
        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        index = []
        data_path = ROOT_PATH / "data" / "minst" / name
        data_path.mkdir(parents=True, exist_ok = True)
        transform = torchvision.transforms.ToTensor()
        mnist_data = torchvision.datasets.MNIST(
            str(data_path), train=(name == "train"), download=True, transform=transform
        )
        for i in tqdm(range(len(mnist_data))):
            # create dataset
            img, label = mnist_data[i]

            save_dict = {"tensor": img}
            save_path = data_path / f"{i:06}.safetensors"
            safetensors.torch.save_file(save_dict, save_path)

            # parse dataset metadata and append it to index
            index.append({"path": str(save_path), "label": label})

        shutil.rmtree(data_path / "MNIST")  # remove

        # write index to disk
        write_json(index, str(data_path / "index.json"))

        return index
