import csv
from json import decoder
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader

DATA_DIR_PATH = Path(__file__).parent / ".." / "data"


def create_dataloader(
    path: str,
    max_length: int,
    batch_size: int,
    limit: int | None = None,
) -> DataLoader[tuple[str, str]]:
    dataset = Wmt14Dataset(path, max_length, limit=limit)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    return dataloader


VOCAB_SIZE = 37000
MAX_LENGTH = 50


def get_tokenizer() -> Tokenizer:
    bpe_path = DATA_DIR_PATH / "bpe_tokenizer.json"
    tokenizer = Tokenizer.from_file(str(bpe_path))
    return tokenizer


class Wmt14Dataset(Dataset):
    def __init__(
        self,
        path: str,
        max_length: int,
        limit: int | None,
    ):
        self.path = path
        self.max_length = max_length
        self.data = self._init_data(limit)
        self.tokenizer = get_tokenizer()
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        source = item[0]
        target = item[1]
        source_tokenizer = self.tokenizer.encode(source)
        original_post_processor = self.tokenizer.post_processor
        self.tokenizer.post_processor = None  # type: ignore
        target_tokenizer = self.tokenizer.encode(target)
        self.tokenizer.post_processor = original_post_processor  # type: ignore
        cls_token_id = self.tokenizer.token_to_id("[CLS]")
        sep_token_id = self.tokenizer.token_to_id("[SEP]")
        decoder_input = [cls_token_id] + target_tokenizer.ids[:-1]
        decoder_output = target_tokenizer.ids + [sep_token_id]
        if len(decoder_input) > self.max_length:
            decoder_input = decoder_input[: self.max_length]
            decoder_output = decoder_output[: self.max_length]
        else:
            decoder_input += [self.pad_token_id] * (
                self.max_length - len(decoder_input)
            )
            decoder_output += [self.pad_token_id] * (
                self.max_length - len(decoder_output)
            )

        return (
            torch.tensor(source_tokenizer.ids).type(torch.long),
            torch.tensor(decoder_input).type(torch.long),
            torch.tensor(decoder_output).type(torch.long),
            " ".join(target_tokenizer.tokens),
        )

    def _init_data(self, limit: int | None) -> list[tuple[str, str]]:
        data: list[tuple[str, str]] = []
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


def create_bpe_vocab():
    bpe_path = DATA_DIR_PATH / "bpe_tokenizer.json"
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
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()  # type: ignore

    # Train the tokenizer
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,  # type: ignore
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],  # type: ignore
    )

    files = [str(bpe_dataset_path)]
    tokenizer.train(files, trainer)
    tokenizer.enable_padding(
        pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=MAX_LENGTH
    )
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",  # Single sentence format
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",  # Paired sentence format
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )  # type: ignore
    tokenizer.save(str(bpe_path))


if __name__ == "__main__":
    create_bpe_vocab()
