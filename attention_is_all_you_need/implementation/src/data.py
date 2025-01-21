import csv
from pathlib import Path

from tokenizers.implementations import CharBPETokenizer
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader

DATA_DIR_PATH = Path(__file__).parent / ".." / "data"


def create_dataloader(
    path: str, max_length: int, batch_size: int, limit: int | None = None
) -> DataLoader[tuple[str, str]]:
    dataset = Wmt14Dataset(path, max_length, limit=limit)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    return dataloader


VOCAB_SIZE = 37000
MAX_LENGTH = 256


class Wmt14Dataset(Dataset):
    def __init__(self, path: str, max_length: int, limit: int | None = None):
        self.path = path
        self.max_length = max_length
        self.data = self._init_data(limit)
        self.tokenizer = CharBPETokenizer(
            str(DATA_DIR_PATH / "vocab.json"),
            str(DATA_DIR_PATH / "merges.txt"),
        )
        self.tokenizer.enable_padding(pad_token="<pad>", length=MAX_LENGTH)
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        source = item[0]
        target = item[1]

        # Encode source text
        encoder_tokens = self.tokenizer.encode(source)
        encoder_input = encoder_tokens.ids

        # Encode target text
        target_tokens = self.tokenizer.encode(target)
        decoder_input = target_tokens.ids

        return (
            torch.tensor(encoder_input).type(torch.int32),
            torch.tensor(decoder_input).type(torch.int32),
        )

    def _init_data(self, limit: int | None) -> list[tuple[str, str]]:
        data = []
        counter = 0
        with open(self.path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader)
            for row in tqdm.tqdm(reader):
                if len(row) != 2:
                    continue
                data.append((row[0], row[1]))
                counter += 1
                if limit and counter >= limit:
                    break
        return data


def _create_bpe_vocab():
    bpe_path = DATA_DIR_PATH / "vocab.json"
    bpe_dataset_path = DATA_DIR_PATH / "bpe_dataset.txt"
    if bpe_path.is_file():
        print("Vocab file exists, skipping...")
        return
    if not bpe_dataset_path.is_file():
        with open(DATA_DIR_PATH / "bpe_dataset.txt", "w+") as writer:
            with open(DATA_DIR_PATH / "wmt14_translate_de-en_train.csv", "r") as f:
                reader = csv.reader(f, delimiter=",")
                next(reader)
                for row in tqdm.tqdm(reader):
                    for col in row:
                        writer.write(col + "\n")

    tokenizer = CharBPETokenizer()
    tokenizer.train(
        files=[str(bpe_dataset_path)],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["<unk>", "<w>", "</w>", "<pad>"],
    )
    tokenizer.enable_padding(pad_token="<pad>", length=MAX_LENGTH)
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    tokenizer.save_model(str(DATA_DIR_PATH))


if __name__ == "__main__":
    _create_bpe_vocab()
