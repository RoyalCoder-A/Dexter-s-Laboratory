from typing import Literal
from tokenizers import Tokenizer
import torch
from datasets import load_dataset


def get_dataloader(
    split: Literal["train", "validation", "test"],
    batch_size: int,
    tokenizer: Tokenizer,
    limit: int | None = None,
) -> torch.utils.data.DataLoader:
    ds = Wmt14Dataset(split, tokenizer, limit=limit)
    dl = torch.utils.data.DataLoader(
        dataset=ds, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    return dl


class Wmt14Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: Literal["train", "validation", "test"],
        tokenizer: Tokenizer,
        limit: int | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = load_dataset("wmt14", "de-en", split=split).to_list()  # type: ignore
        if limit:
            self.dataset = self.dataset[:limit]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.dataset[index]["translation"]
        src: str = item["de"]
        tgt: str = item["en"]
        dec_src = "[CLS] " + " ".join(tgt.split()[:-1])
        dec_tgt = " ".join(tgt.split()) + " [SEP]"
        src_ids = self.tokenizer.encode(src).ids
        dec_tgt_ids = self.tokenizer.encode(dec_tgt).ids
        dec_src_ids = self.tokenizer.encode(dec_src).ids
        return (
            torch.tensor(src_ids).long(),
            torch.tensor(dec_src_ids).long(),
            torch.tensor(dec_tgt_ids).long(),
        )
