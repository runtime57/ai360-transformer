import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import load_dataset

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class TextDataset(BaseDataset):

    def __init__(
        self, tokenizer, block_size, name="train", *args, **kwargs
    ):
        self.set_tokenizer(tokenizer=tokenizer)
        self.block_size = block_size
        index_path = ROOT_PATH / "small shakespeare" / name / "index.json"
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)
        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        index = []
        data_path = ROOT_PATH / "small shakespeare" / name
        data_path.mkdir(exist_ok=True, parents=True)

        ds = load_dataset("karpathy/tiny_shakespeare", trust_remote_code=True)
        correct_keys = {
            "train" : "train",
            "test" : "test",
            "val" : "validation"
        }
        # call name to brake the code cause why not
        text = ds[correct_keys.get(name, name)]["text"][0]
        seq = self.tokenizer.encode(text, add_special_tokens=False)

        dataset_length = len(seq)
        number_of_zeros = int(np.log10(dataset_length)) + 1

        tokens_window = [0] * self.block_size
        labels_window = [0] * self.block_size
        for i in tqdm(range(dataset_length)):
            labels_window.pop(0)
            labels_window.append(seq[i])
            obj = torch.tensor(tokens_window)
            obj_path = data_path / f"{i:0{number_of_zeros}d}.pt"
            torch.save(obj, obj_path)
            index.append({"path": str(obj_path), "label": labels_window})
            tokens_window.pop(0)
            tokens_window.append(seq[i])

        write_json(index, str(data_path / "index.json"))
        return index