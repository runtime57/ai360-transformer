from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.tokenizers import BaseTokenizer


class BaseDataset(ABC, Dataset):
    def __init__(self, instance_transforms: dict | None = None, tokenizer: 'BaseTokenizer | None' = None):
        self.instance_transforms = instance_transforms
        if tokenizer is not None:
            self.set_tokenizer(tokenizer)

    @abstractmethod
    def __len__() -> int:
        ...

    @abstractmethod
    def __getitem__(self, ind: int) -> dict:
        ...

    def set_tokenizer(self, tokenizer: 'BaseTokenizer') -> None:
        self.tokenizer = tokenizer

    def preprocess_data(self, instance_data: dict) -> dict:
        if self.instance_transforms is not None:
            for name, transform in self.instance_transforms.items():
                instance_data[name] = transform(instance_data[name])
        return instance_data

    instance_transforms = None

        
