from pathlib import Path

import torch
import torchinfo

from attention_is_all_you_need.implementation.src.data import create_dataloader
from attention_is_all_you_need.implementation.src.layers.pre_layer import PreLayer

if __name__ == "__main__":
    print(torch.mps.is_available())
    if torch.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)
    dataloader = create_dataloader(
        str(Path(__file__).parent / ".." / "data" / "wmt14_translate_de-en_train.csv")
    )
    layer = PreLayer(37000, 512, 256, 0.1, device)
    print(
        torchinfo.summary(
            layer, input_size=(32, 256), dtypes=[torch.int32], device=device
        )
    )
    for data in dataloader:
        encoder_input, decoder_input = data
        print(encoder_input.shape, decoder_input.shape)
        print(decoder_input[0])
        print(layer(encoder_input.to(device)))
        exit()
