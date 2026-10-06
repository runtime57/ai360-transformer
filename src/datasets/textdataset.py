import numpy as np
import torch
from tqdm.auto import tqdm
import urllib

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class TextDataset(BaseDataset):

    def __init__(
        self, tokenizer, block_size, name="train", *args, **kwargs
    ):
        """
        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        """
        self.set_tokenizer(tokenizer=tokenizer)
        self.block_size = block_size
        index_path = ROOT_PATH / "data" / name / "index.json"

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

        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        index = []
        data_path = ROOT_PATH / "data" / "text" / name
        data_path.mkdir(exist_ok=True, parents=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = urllib.request.urlopen(url).read().decode()
        seq = self.tokenizer.encode(text)

        dataset_length = len(seq)
        # to get pretty object names
        number_of_zeros = int(np.log10(dataset_length)) + 1

        window = [0 for i in range(self.block_size)]
        for i in tqdm(range(dataset_length)):
            img, label = window, text[i]
            obj = torch.from_numpy(np.array(img))
            obj_path = data_path / f"{i:0{number_of_zeros}d}.pt"
            torch.save(obj, obj_path)
            index.append({"path": str(obj_path), "label": int(label)})
            window.pop(0)
            window.append(text[i])


        # write index to disk
        write_json(index, str(data_path / "index.json"))

        return index
