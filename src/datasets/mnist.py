from pathlib import Path
import shutil

from torchvision.datasets import MNIST
from tqdm.auto import tqdm

import safetensors.torch
from torchvision.transforms.functional import pil_to_tensor
from src.utils.io_utils import ROOT_PATH, read_json, write_json

from .base_dataset import BaseDataset

class MnistDataset(BaseDataset):
    def __init__(self, name='train', *args, **kwargs):
        index_path = ROOT_PATH / 'data' / 'mnist' / name / 'index.json'
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._download(name)
        super().__init__(index, *args, **kwargs)
    
    def _download(self, name):
        index = []
        path: Path = ROOT_PATH / 'data' / 'mnist' / name
        path.mkdir(exist_ok=True, parents=True)

        mnist = MNIST(str(path), train=(name == 'train'), download=True)

        for i in tqdm(range(len(mnist))):
            img, label = mnist[i]
            img = pil_to_tensor(img).float() / 255.0  # PIL -> (1, 28, 28) in [0, 1]
            filename = path / f'{i:06d}.safetensors'
            safetensors.torch.save_file({'tensor': img}, filename)
            index.append({'path': str(filename), 'label': label})

        shutil.rmtree(path / 'MNIST')
        write_json(index, str(path / 'index.json'))
        return index

    def load_object(self, path):
        return safetensors.torch.load_file(path)['tensor']
