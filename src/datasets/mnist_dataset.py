import numpy as np
import torch
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class MnistDataset(BaseDataset):
    """
    mnist of a nested dataset class to show basic structure.

    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(
        self, input_length, name="train", *args, **kwargs
    ):
        """
        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / "mnist" / name / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(input_length, name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.
        """
        index = []
        data_path = ROOT_PATH / "data" / "mnist" / name
        is_train = (name == "train")
        mnist_data = torch.load(str(data_path), train=is_train, download=True)
        data_path.mkdir(exist_ok=True, parents=True)
        dataset_length = len(mnist_data)

        # to get pretty object names
        number_of_zeros = int(np.log10(dataset_length)) + 1

        for i in tqdm(range(dataset_length)):
            image, label = mnist_data[i]
            image_path = data_path / f"{i:0{number_of_zeros}d}.pt"
            torch.save(mnist_data, image_path)
            # parse dataset metadata and append it to index
            index.append({"path": str(image_path), "label": label})

        # write index to disk
        write_json(index, str(data_path / "index.json"))

        return index
