from pathlib import Path
import torch

from tqdm import tqdm
from datasets import load_dataset, load_from_disk

from src.utils.io_utils import ROOT_PATH


class TinyStoriesDataset:
    def __init__(
        self,
        part,
        max_seq_len,
        val_ratio,
        seed=42,
        allow_download=True,
        override=False,
        raw_data_path=None,
        tokenizer=None,
        tokenization_workers=4,
        transforms=None,
    ):
        assert part in ['train', 'val', 'test']

        data_dir = ROOT_PATH / "data" / "tiny_story"

        raw_data_path = Path(raw_data_path) if raw_data_path is not None else data_dir / "raw"

        self.part = part
        self.max_seq_len = max_seq_len
        self.transforms = transforms
        self.seed = seed

        if override or not raw_data_path.exists():
            assert allow_download

            raw_data_path.parent.mkdir(parents=True, exist_ok=True)
            dataset = load_dataset("roneneldan/TinyStories")
            dataset.save_to_disk(str(raw_data_path))

        self.tokenization_workers = tokenization_workers
        self.stories = self._prepare(part, val_ratio, raw_data_path)

        if tokenizer is not None:
            self._set_tokenizer(tokenizer)

    def __len__(self):
        return self.tokens.shape[0]

    def _get_train_text(self):
        return (item["text"] for item in self.stories)

    def _prepare(self, part, val_ratio, raw_data_path):
        dataset = load_from_disk(str(raw_data_path))

        if part == 'test':
            return dataset["validation"]

        train = dataset["train"].shuffle(seed=self.seed)
        split_pos = int(len(train) * (1 - val_ratio))

        if self.part == "train":
            return train.select(range(split_pos))
        return train.select(range(split_pos, len(train)))

    def _split_by_maxlen(self, tokens, max_seq_len):
        pad_len = (-tokens.shape[0]) % max_seq_len

        if pad_len > 0:
            padding = torch.full((pad_len,), self.pad_id, dtype=tokens.dtype, device=tokens.device)
            tokens = torch.cat([tokens, padding])

        return tokens.reshape(-1, max_seq_len)

    @staticmethod
    def tokenize_stories(batch, tokenizer):
        input_ids = []

        for text in batch["text"]:
            tokens = list(tokenizer.encode(text))
            input_ids.append([*tokens])

        return {"input_ids": input_ids}

    def _set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

        tokenized = self.stories.map(
            self.tokenize_stories,
            batched=True,
            batch_size=1000,
            num_proc=min(self.tokenization_workers, len(self.stories)),
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=self.stories.column_names,
            desc=f"Tokenizing {self.part}",
        )

        token_ids = []

        for start in tqdm(range(0, len(tokenized), 1000), desc=f"Collecting {self.part}"):
            batch = tokenized[start:start + 1000]
            for story_tokens in batch["input_ids"]:
                if self.part == "train":
                    token_ids.extend(story_tokens)
                else:
                    for offset in range(0, len(story_tokens) - 1, self.max_seq_len):
                        data = torch.as_tensor(story_tokens[offset:offset + self.max_seq_len + 1], dtype=torch.long)
                        token_ids.append(self._split_by_maxlen(data, self.max_seq_len + 1))

        if self.part == "train":
            tokens = torch.as_tensor(token_ids, dtype=torch.long)
            self.tokens = self._split_by_maxlen(tokens, self.max_seq_len + 1)
        else:
            self.tokens = torch.cat(token_ids, dim=0)

    def _get_context(self):
        return {
            "part": self.part,
            "vocab_size": len(self.tokenizer.id_to_token),
            "ignore_class_id": self.pad_id,
            "max_seq_len": self.max_seq_len
        }

    @property
    def pad_id(self):
        return self.tokenizer.token_to_id["<pad>"]

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
