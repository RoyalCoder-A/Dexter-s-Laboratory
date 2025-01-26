import argparse
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from attention_is_all_you_need.implementation.src import trainer
from attention_is_all_you_need.implementation.src.data import (
    MAX_LENGTH,
    VOCAB_SIZE,
    create_dataloader,
)
from attention_is_all_you_need.implementation.src.layers.pre_layer import PreLayer
from attention_is_all_you_need.implementation.src.trainer import Trainer
from attention_is_all_you_need.implementation.src.transformer_model import (
    TransformerModel,
)

if __name__ == "__main__":

    BATCH_SIZE = 32

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/cuda/mps)")
    args = parser.parse_args()
    device = args.device
    print(device)
    train_dataloader, tokenizer = create_dataloader(
        str(Path(__file__).parent / ".." / "data" / "wmt14_translate_de-en_train.csv"),
        MAX_LENGTH,
        BATCH_SIZE,
        limit=1000,
    )
    test_dataloader, _ = create_dataloader(
        str(
            Path(__file__).parent
            / ".."
            / "data"
            / "wmt14_translate_de-en_validation.csv"
        ),
        MAX_LENGTH,
        BATCH_SIZE,
        tokenizer=tokenizer,
    )
    transformer = TransformerModel(VOCAB_SIZE, MAX_LENGTH, 6, 512, 2048, 8, device)
    # transformer = torch.compile(transformer)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = torch.optim.Adam(
        params=transformer.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=1
    )
    summary_writer = SummaryWriter(
        str(Path(__file__).parent / ".." / "runs" / "first_run")
    )
    trainer = Trainer(
        model=transformer,
        optimizer=optimizer,
        batch_size=BATCH_SIZE,
        d_model=512,
        train_data_loader=train_dataloader,
        test_data_loader=test_dataloader,
        loss_fn=loss_fn,
        device=device,
        tokenizer=tokenizer,
        warmup_steps=4000,
        num_epochs=3125,
        summary_writer=summary_writer,
    )
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    trainer.train()
