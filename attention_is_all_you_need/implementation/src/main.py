from pathlib import Path

import torch

from attention_is_all_you_need.implementation.src.data import create_dataloader

if __name__ == "__main__":
    print(torch.mps.is_available())
    dataloader = create_dataloader(
        str(Path(__file__).parent / ".." / "data" / "wmt14_translate_de-en_train.csv")
    )
    for data in dataloader:
        encoder_input, decoder_input = data
        print(encoder_input.shape, decoder_input.shape)
        exit()
