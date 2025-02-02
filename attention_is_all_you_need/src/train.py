from pathlib import Path
from typing import Literal
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from attention_is_all_you_need.src.utils.dataset import get_dataloader
from attention_is_all_you_need.src.utils.tokenizer import (
    DATA_DIR_PATH,
    VOCAB_SIZE,
    get_tokenizer,
    train_bpe_tokenizer,
)
from attention_is_all_you_need.src.utils.trainer import Trainer
from attention_is_all_you_need.src.utils.transformer_model import TransformerModel


def train(
    batch_size: int,
    epochs: int,
    device: Literal["cpu", "cuda", "mps"],
    data_path: Path = DATA_DIR_PATH,
):
    train_bpe_tokenizer(data_path)
    tokenizer = get_tokenizer(data_path)
    transformer_model = TransformerModel(VOCAB_SIZE, 512, 6, 2048, 8, 0.1)
    train_dl = get_dataloader("train", batch_size, tokenizer)
    val_dl = get_dataloader("validation", batch_size, tokenizer)
    summary_writer = SummaryWriter(
        data_path / "runs" / f"{device}__{batch_size}__{epochs}"
    )
    checkpoint_path = data_path / "checkpoints" / f"{device}__{batch_size}__{epochs}.pt"
    if checkpoint_path.exists():
        transformer_model.load_state_dict(
            torch.load(checkpoint_path, weights_only=False)
        )
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
    transformer_model = transformer_model.to(device)
    transformer_model.compile()
    trainer = Trainer(
        transformer_model,
        batch_size,
        epochs,
        4000,
        512,
        tokenizer,
        device,
        train_dl,
        val_dl,
        summary_writer,
        checkpoint_path,
    )
    trainer.train()
