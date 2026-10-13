import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import torch

from datasets import load_dataset, load_from_disk

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

logger = logging.getLogger(__name__)


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
        cache_dir=None,
        n_future_tokens=None,
        tokenizer=None,
        tokenization_workers=4,
        transforms=None,
    ):
        assert part in ['train', 'val', 'test']

        data_dir = ROOT_PATH / "data" / "tiny_story"

        self.raw_data_path = Path(raw_data_path) if raw_data_path is not None else data_dir / "raw"
        # tokenized story streams; may point to a read-only dir (e.g. /kaggle/input/...)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else data_dir / "tokens"

        self.part = part
        self.max_seq_len = max_seq_len
        self.val_ratio = val_ratio
        self.transforms = transforms
        self.seed = seed
        self.n_future_tokens = n_future_tokens
        self.allow_download = allow_download
        self.override = override
        self.tokenization_workers = min(tokenization_workers, os.cpu_count() or 1)
        self._stories = None

        if tokenizer is not None:
            self._set_tokenizer(tokenizer)

    def __len__(self):
        return self.tokens.shape[0]

    @property
    def stories(self):
        # loaded lazily: not needed at all when the tokenized cache exists
        if self._stories is None:
            if self.override or not self.raw_data_path.exists():
                assert self.allow_download

                self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
                dataset = load_dataset("roneneldan/TinyStories")
                dataset.save_to_disk(str(self.raw_data_path))

            self._stories = self._prepare(self.part, self.val_ratio, self.raw_data_path)
        return self._stories

    def _get_train_text(self):
        # yield batches (lists of str): row-by-row access on a shuffled dataset is slow
        # and batches let the Rust BPE trainer pre-tokenize in parallel
        return (batch["text"] for batch in self.stories.iter(batch_size=10_000))

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

    def _cache_paths(self, tokenizer):
        # the stream depends only on the tokenizer and the split, not on max_seq_len / n_future_tokens
        state = {
            "token_to_id": tokenizer.token_to_id,
            "model_state": tokenizer._get_model_state(),
            "seed": self.seed,
            "val_ratio": self.val_ratio,
        }
        key = hashlib.sha1(json.dumps(state, sort_keys=True).encode()).hexdigest()[:12]
        prefix = self.cache_dir / f"{self.part}_{key}"
        return Path(f"{prefix}_tokens.npy"), Path(f"{prefix}_offsets.npy")

    def _tokenize_to_stream(self, tokenizer):
        tokenized = self.stories.map(
            self.tokenize_stories,
            batched=True,
            batch_size=1000,
            num_proc=min(self.tokenization_workers, len(self.stories)),
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=self.stories.column_names,
            desc=f"Tokenizing {self.part}",
        )

        dtype = np.uint16 if len(tokenizer.token_to_id) <= np.iinfo(np.uint16).max + 1 else np.int32

        # read the arrow column chunk by chunk instead of looping over python lists
        flat, lengths = [], []
        for chunk in tokenized.data.column("input_ids").chunks:
            flat.append(chunk.flatten().to_numpy(zero_copy_only=False).astype(dtype))
            lengths.append(pc.list_value_length(chunk).to_numpy(zero_copy_only=False))

        flat = np.concatenate(flat)
        offsets = np.concatenate([[0], np.cumsum(np.concatenate(lengths))]).astype(np.int64)
        return flat, offsets

    def _load_or_build_stream(self, tokenizer):
        tokens_path, offsets_path = self._cache_paths(tokenizer)

        if tokens_path.exists() and offsets_path.exists():
            logger.info("Loading tokenized %s stream from %s", self.part, tokens_path)
            return np.load(tokens_path), np.load(offsets_path)

        logger.info("No tokenized cache for %s at %s, tokenizing", self.part, tokens_path)
        flat, offsets = self._tokenize_to_stream(tokenizer)

        # write to temporary files first so an interrupted run never leaves a broken cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for path, array in ((tokens_path, flat), (offsets_path, offsets)):
            tmp_path = path.with_suffix(".tmp.npy")
            np.save(tmp_path, array)
            tmp_path.replace(path)
        logger.info("Saved tokenized %s stream to %s", self.part, tokens_path)

        return flat, offsets

    def _set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

        flat, offsets = self._load_or_build_stream(tokenizer)

        # None: plain next-token targets [L]; int: targets of all future offsets [N, L]
        len_chunk = self.max_seq_len + (self.n_future_tokens or 1)

        if self.part == "train":
            # int32 keeps RAM at half of int64; __getitem__ casts to long
            tokens = torch.from_numpy(flat.astype(np.int32))
            self.tokens = self._split_by_maxlen(tokens, len_chunk)
        else:
            chunks = []
            for start, end in zip(offsets[:-1], offsets[1:]):
                story_tokens = flat[start:end]
                for offset in range(0, len(story_tokens) - 1, self.max_seq_len):
                    data = story_tokens[offset:offset + len_chunk]
                    row = np.full(len_chunk, self.pad_id, dtype=np.int32)
                    row[:len(data)] = data
                    chunks.append(row)
            self.tokens = torch.from_numpy(np.stack(chunks))

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
        data = self.tokens[idx].long()

        seq, target = data[:self.max_seq_len], data[1:]
        if self.n_future_tokens is not None:
            target = target.unfold(0, self.max_seq_len, 1)

        instance_data = {"seq": seq, "target": target}
        instance_data = self.preprocess_data(instance_data)

        return instance_data
