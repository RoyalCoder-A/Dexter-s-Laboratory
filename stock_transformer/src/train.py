from pathlib import Path
from typing import Literal
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from stock_transformer.src.utils.dataset import FEATURES_COUNT, get_dataloader
from stock_transformer.src.utils.trainer import Trainer
from stock_transformer.src.utils.transformer_model import TransformerModel


def train(
    *,
    batch_size: int,
    epochs: int,
    device: Literal["cpu", "cuda", "mps"],
    train_ds_path: Path,
    test_ds_path: Path,
    data_path: Path,
    train_name: str = "",
):
    transformer_model = TransformerModel(FEATURES_COUNT, 512, 6, 2048, 8, 0.1)
    train_dl, test_dl = get_dataloader(train_ds_path, test_ds_path, batch_size)
    summary_writer_path = data_path / "runs"
    checkpoint_path = data_path / "checkpoints"
    if train_name:
        summary_writer_path = summary_writer_path / train_name
        checkpoint_path = checkpoint_path / train_name
    else:
        summary_writer_path = summary_writer_path / f"{device}__{batch_size}__{epochs}"
    checkpoint_path = checkpoint_path / f"{device}__{batch_size}__{epochs}.pt"
    summary_writer = SummaryWriter(summary_writer_path)
    if checkpoint_path.exists():
        transformer_model.load_state_dict(
            torch.load(checkpoint_path, weights_only=False)
        )
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
    transformer_model = transformer_model.to(device)
    if device == "cuda":
        transformer_model.compile()
    trainer = Trainer(
        transformer_model,
        batch_size,
        epochs,
        4000,
        512,
        device,
        train_dl,
        test_dl,
        summary_writer,
        checkpoint_path,
    )
    trainer.train()
