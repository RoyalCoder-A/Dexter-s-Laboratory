from typing import Literal
import torch
from datasets import load_dataset

from attention_is_all_you_need.src.utils.tokenizer import get_tokenizer


def get_dataloader(
    split: Literal["train", "valid", "test"], batch_size: int
) -> torch.utils.data.DataLoader:
    ds = Wmt14Dataset(split)
    dl = torch.utils.data.DataLoader(
        dataset=ds, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    return dl


class Wmt14Dataset(torch.utils.data.Dataset):
    def __init__(self, split: Literal["train", "valid", "test"]) -> None:
        super().__init__()
        self.tokenizer = get_tokenizer()
        self.dataset = load_dataset("wmt14", "de-en", split=split).to_list()  # type: ignore

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.dataset[index]["translation"]
        src: str = item["de"]
        tgt: str = item["en"]
        dec_src = "[CLS] " + " ".join(tgt.split()[:-1])
        src_ids = self.tokenizer.encode(src).ids
        tgt_ids = self.tokenizer.encode(tgt).ids
        dec_src_ids = self.tokenizer.encode(dec_src).ids
        return (
            torch.tensor(src_ids).long(),
            torch.tensor(dec_src_ids).long(),
            torch.tensor(tgt_ids).long(),
        )
