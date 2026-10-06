import torch

from src.utils.io_utils import ROOT_PATH, read_txt, write_txt


class ShakespeareDataset:
    def __init__(self, part, max_seq_len, val_ratio, tokenizer, override=False, raw_data_path=None, processed_dir_path=None, transforms=None):
        assert part in ['train', 'val']

        self.max_seq_len = max_seq_len
        self.transforms = transforms

        if raw_data_path is None:
            raw_data_path = ROOT_PATH / "data" / "tiny_shakespeare" / "raw" / "input.txt"
        if processed_dir_path is None:
            processed_dir_path = ROOT_PATH / "data" / "tiny_shakespeare" / "processed"

        self.processed_file_path = processed_dir_path / f"{part}.txt"
        if override or not self.processed_file_path.exists():
            self._preprocess(val_ratio, raw_data_path, processed_dir_path)

        text = read_txt(str(self.processed_file_path))

        self.tokenizer = tokenizer
        tokens = torch.as_tensor(self.tokenizer.encode(text), dtype=torch.long)
        self.tokens = self._split_by_maxlen(tokens, max_seq_len + 1)

    def __len__(self):
        return self.tokens.shape[0]

    def _preprocess(self, val_ratio, raw_data_path, processed_dir_path):
        processed_dir_path.mkdir(exist_ok=True, parents=True)

        text = read_txt(str(raw_data_path))
        split_pos = int(len(text) * (1 - val_ratio))

        train_text = text[:split_pos]
        val_text = text[split_pos:]

        write_txt(train_text, processed_dir_path / "train.txt")
        write_txt(val_text, processed_dir_path / "val.txt")

    def _split_by_maxlen(self, tokens, max_seq_len):
        pad_len = (-tokens.shape[0]) % max_seq_len

        if pad_len > 0:
            pad_id = self.tokenizer.token_to_id["<pad>"]
            padding = torch.full((pad_len,), pad_id, dtype=tokens.dtype, device=tokens.device)
            tokens = torch.cat([tokens, padding])

        return tokens.reshape(-1, max_seq_len)

    def _set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.tokenizer = tokenizer
        tokens = torch.as_tensor(self.tokenizer.encode(read_txt(self.processed_file_path)), dtype=torch.long)
        self.tokens = self._split_by_maxlen(tokens, self.max_seq_len + 1)


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
