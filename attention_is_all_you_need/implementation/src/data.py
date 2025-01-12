import csv
from pathlib import Path
from typing import Callable

from tokenizers.implementations import CharBPETokenizer
import tqdm
from torch.utils.data import Dataset, DataLoader

DATA_DIR_PATH = Path(__file__).parent / ".." / "data"


def create_dataloader(path: str) -> DataLoader[tuple[str, str]]:
    tokenizer = CharBPETokenizer(
        str(DATA_DIR_PATH / "vocab.json"), str(DATA_DIR_PATH / "merges.txt")
    )
    dataset = Wmt14Dataset(path, lambda x: tokenizer.encode(x).ids)
    dataloader = DataLoader(dataset, 32, True, pin_memory=True)
    return dataloader


class Wmt14Dataset(Dataset):
    def __init__(self, path: str, target_transform: Callable[[str], list[int]]):
        self.path = path
        self.target_transform = target_transform
        self.data = self._init_data()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        source = item[0]
        target = item[1]
        if self.target_transform:
            target = self.target_transform(target)
            source = self.target_transform(source)
        return source, target

    def _init_data(self) -> list[tuple[str, str]]:
        data = []
        with open(self.path, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader)
            for row in tqdm.tqdm(reader):
                if len(row) != 2:
                    continue
                data.append((row[0], row[1]))
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

    vocab_size = 37000
    tokenizer = CharBPETokenizer()
    tokenizer.train(
        files=[str(bpe_dataset_path)],
        vocab_size=vocab_size,  # Number of tokens in the vocabulary
        min_frequency=2,  # Minimum frequency for a token to be included
    )
    tokenizer.save_model(str(DATA_DIR_PATH))


if __name__ == "__main__":
    # TODO
    """
    We need data preprocessing, including:
    1. Adding <s> and </s>
    2. Pad and truncate to max length
    3. Add <s> and </s> to special characters
    4. <unk> is already set for unknowns, so just need to add it to list!
    """
    _create_bpe_vocab()
