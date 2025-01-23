from pathlib import Path

import torch
import torchinfo

from attention_is_all_you_need.implementation.src.data import (
    MAX_LENGTH,
    VOCAB_SIZE,
    create_dataloader,
)
from attention_is_all_you_need.implementation.src.layers.pre_layer import PreLayer
from attention_is_all_you_need.implementation.src.transformer_model import (
    TransformerModel,
)

if __name__ == "__main__":
    if torch.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)
    dataloader = create_dataloader(
        str(Path(__file__).parent / ".." / "data" / "wmt14_translate_de-en_train.csv"),
        MAX_LENGTH,
        32,
        100,
    )
    transformer = TransformerModel(VOCAB_SIZE, MAX_LENGTH, 6, 512, 2048, 8, device)
    for data in dataloader:
        encoder_input, decoder_input, decoder_output = data
        print(
            "Data shapes: ",
            encoder_input.shape,
            decoder_input.shape,
            decoder_output.shape,
        )
        torchinfo.summary(
            transformer,
            device=device,
            input_data=(encoder_input, decoder_input),
        )
        exit()
