import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class TransformerDataset(BaseDataset):
    def __init__(
        self, input_length, n_classes, dataset_length, name="train", *args, **kwargs
    ):
        index_path = ROOT_PATH / "data" / "transformer" / name / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(input_length, n_classes, dataset_length, name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, input_length, n_classes, dataset_length, name):
        index = []
        data_path = ROOT_PATH / "data" / "transformer" / name
        data_path.mkdir(exist_ok=True, parents=True)

        dataset = load_dataset("https://huggingface.co/datasets/karpathy/tiny_shakespeare", split=name)

        number_of_zeros = int(np.log10(dataset_length)) + 1
        for i in tqdm(range(dataset_length)):
            # create dataset
            transformer_path = data_path / f"{i:0{number_of_zeros}d}.pt"
            transformer_data = torch.randn(input_length)
            transformer_label = torch.randint(n_classes, size=(1,)).item()
            torch.save(transformer_data, transformer_path)

            index.append({"path": str(transformer_path), "label": transformer_label})

        # write index to disk
        write_json(index, str(data_path / "index.json"))

        return index
