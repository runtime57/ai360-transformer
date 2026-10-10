import torch
import requests

from src.utils.io_utils import ROOT_PATH, read_txt, write_txt


class ShakespeareDataset:
    def __init__(self, part, max_seq_len, val_ratio, allow_download=True, tokenizer=None, override=False, raw_data_path=None, processed_dir_path=None, transforms=None):
        assert part in ['train', 'val']

        self.part = part
        self.max_seq_len = max_seq_len
        self.transforms = transforms

        if raw_data_path is None:
            raw_data_path = ROOT_PATH / "data" / "tiny_shakespeare" / "raw" / "input.txt"
        if processed_dir_path is None:
            processed_dir_path = ROOT_PATH / "data" / "tiny_shakespeare" / "processed"

        self.processed_file_path = processed_dir_path / f"{part}.txt"
        if override or not self.processed_file_path.exists():
            if not raw_data_path.exists():
                assert allow_download, "tiny_shakespeare raw text file was not found"
                self.download_dataset(raw_data_path)
            self._preprocess(val_ratio, raw_data_path, processed_dir_path)

        self.tokenizer = tokenizer
        if self.tokenizer is not None:
            tokens = torch.as_tensor(self.tokenizer.encode(read_txt(self.processed_file_path)), dtype=torch.long)
            self.tokens = self._split_by_maxlen(tokens, max_seq_len + 1)

    def __len__(self):
        return self.tokens.shape[0]

    @staticmethod
    def download_dataset(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        part_path = path.with_name(path.name + ".part")

        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            with open(part_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)

        part_path.replace(path)

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
        tokens = torch.as_tensor(self.tokenizer.encode(read_txt(self.processed_file_path)), dtype=torch.long)
        self.tokens = self._split_by_maxlen(tokens, self.max_seq_len + 1)

    def _get_context(self):
        return {
            "part": self.part,
            "vocab_size": len(self.tokenizer.id_to_token),
            "ignore_class_id": self.tokenizer.token_to_id['<pad>'],
            "max_seq_len": self.max_seq_len
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
