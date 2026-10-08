import numpy as np
import torch
from tqdm.auto import tqdm
from pathlib import Path
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
        ROOT_PATH = Path(__file__).resolve().parent
        data_dir = ROOT_PATH / "data"
        data_dir.mkdir(exist_ok=True)
        train_path = data_dir / "train_input.txt"
        test_path = data_dir / "test_input.txt"

        data_dir2 = ROOT_PATH / "data" / name
        data_dir2.mkdir(exist_ok=True)

        url = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
               "master/data/tinyshakespeare/input.txt")
        input_path = data_dir / "tinyshakespeare_input.txt"
        if not input_path.exists():
            print("Downloading Tiny Shakespeare...")
            urllib.request.urlretrieve(url, input_path)
            print(f"Saved to: {input_path}")
        else: print("Dataset already downloaded.")

        with open(input_path, "r", encoding="utf-8") as f:
            textf = f.read()
        print(f"Total characters: {len(textf):,}")

        train_ratio = 0.9
        split_idx = int(len(textf) * train_ratio)
        text = {}
        text["train"] = textf[:split_idx]
        text["test"] = textf[split_idx:]

        with open(train_path, "w", encoding="utf-8") as f:
            f.write(text["train"])
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(text["test"])


        seq = self.tokenizer.encode(text[name])

        dataset_length = len(seq)
        # to get pretty object names
        number_of_zeros = int(np.log10(dataset_length)) + 1

        window = [0 for _ in range(self.block_size)]
        for i in tqdm(range(dataset_length)):
            obj = torch.tensor(window, dtype=torch.int32)
            window.pop(0)
            window.append(seq[i])
            label = window

            obj_path = data_dir / name / f"{i:0{number_of_zeros}d}.pt"
            torch.save(obj, obj_path)
            index.append({"path": str(obj_path), "label": label})


        # write index to disk
        write_json(index, str(data_dir / name / "index.json"))

        return index
