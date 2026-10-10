import torch
from tqdm.auto import tqdm

from datasets import load_dataset

from src.utils.io_utils import ROOT_PATH, write_txt


class TinyStoryDataset:
    HF_NAME = "roneneldan/TinyStories"
    HF_SPLITS = {"train": "train", "val": "validation"}

    def __init__(self, part, max_seq_len, limit=None, tokenizer=None, override=False, processed_dir_path=None, transforms=None):
        assert part in self.HF_SPLITS

        self.part = part
        self.max_seq_len = max_seq_len
        self.transforms = transforms

        ds = load_dataset(self.HF_NAME, split=self.HF_SPLITS[part])
        if limit is not None:
            ds = ds.shuffle(seed=42).select(range(min(limit, len(ds))))
        self.texts = ds["text"]

        if processed_dir_path is None:
            processed_dir_path = ROOT_PATH / "data" / "tinystory" / "processed"

        # plain-text corpus is used to train the tokenizer (see tokenizer config)
        self.processed_file_path = processed_dir_path / f"{part}.txt"
        if override or not self.processed_file_path.exists():
            processed_dir_path.mkdir(exist_ok=True, parents=True)
            write_txt("\n".join(self.texts), self.processed_file_path)

        self.tokenizer = None
        self.tokens = None
        if tokenizer is not None:
            self._set_tokenizer(tokenizer)

    def __len__(self):
        return self.tokens.shape[0]

    def _tokenize(self):
        # every story is wrapped in <bos> ... <eos>, then stories are packed together
        ids = []
        for text in tqdm(self.texts, desc=f"tokenize {self.part}"):
            ids.extend(self.tokenizer.encode(text))
        return torch.as_tensor(ids, dtype=torch.long)

    def _split_by_maxlen(self, tokens, max_seq_len):
        pad_len = (-tokens.shape[0]) % max_seq_len

        if pad_len > 0:
            pad_id = self.tokenizer.token_to_id["<pad>"]
            padding = torch.full((pad_len,), pad_id, dtype=tokens.dtype, device=tokens.device)
            tokens = torch.cat([tokens, padding])

        return tokens.reshape(-1, max_seq_len)

    def _set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokens = self._split_by_maxlen(self._tokenize(), self.max_seq_len + 1)

    def _get_context(self):
        return {
            "part": self.part,
            "vocab_size": len(self.tokenizer.id_to_token),
            "ignore_class_id": self.tokenizer.token_to_id['<pad>']
        }

    def preprocess_data(self, instance_data):
        if self.transforms is not None:
            for transform in self.transforms:
                instance_data.update(transform(**instance_data))
        return instance_data

    def __getitem__(self, idx):
        data = self.tokens[idx]

        seq, target = data[:-1], data[1:]

        instance_data = {"seq": seq, "target": target}
        instance_data = self.preprocess_data(instance_data)

        return instance_data
