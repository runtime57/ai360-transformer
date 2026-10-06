import urllib.request

import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class TinyTextDataset(BaseDataset):

    def __init__(
        self, name="train", *args, **kwargs
    ):
        """
        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / "tinytext" / name / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
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
        data_path = ROOT_PATH / "data" / "tinytext" / name
        data_path.mkdir(exist_ok=True, parents=True)
        write_json(index, str(data_path / "index.json"))
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = urllib.request.urlopen(url).read().decode()
        chars = set(text)
        ch_to_int = {c: i for i , c in enumerate(chars)}
        for c in text:
            index.append({"token":ch_to_int[c], "value":c})
        return index
