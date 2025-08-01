from pathlib import Path
from typing import Literal

import torch
import tqdm
from torchmetrics.regression import MeanAbsoluteError
from torch.utils.tensorboard.writer import SummaryWriter

from stock_transformer.src.utils.transformer_model import TransformerModel


class Trainer:
    def __init__(
        self,
        transformer_model: TransformerModel,
        batch_size: int,
        epochs: int,
        warmup_steps: int,
        d_model: int,
        device: Literal["cpu", "cuda", "mps"],
        train_dl: torch.utils.data.DataLoader,
        valid_dl: torch.utils.data.DataLoader,
        summary_writer: SummaryWriter,
        checkpoint_path: Path,
    ) -> None:
        self.transformer_model = transformer_model
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.loss_fn = torch.nn.HuberLoss().to(device)
        self.validation_metric = MeanAbsoluteError().to(device)
        self.optimizer = torch.optim.Adam(
            self.transformer_model.parameters(), betas=(0.9, 0.98), eps=1e-9
        )
        self.summary_writer = summary_writer
        self.current_step = 0
        self.checkpoint_path = checkpoint_path

    def train(self):
        self.current_step = 0
        best_loss = float("inf")
        for i in range(self.epochs):
            print(f"Epoch {i+1} from {self.epochs}")
            train_loss = self._train_step()
            test_loss, test_val = self._test_step()
            self.summary_writer.add_scalar("train_loss", train_loss, i)
            self.summary_writer.add_scalar("test_loss", test_loss, i)
            self.summary_writer.add_scalar("test_val", test_val, i)
            print(f"Train loss: {train_loss:.4f}")
            print(f"Test loss: {test_loss:.4f}")
            print(f"Test val: {test_val:.4f}")
            if test_loss < best_loss:
                best_loss = test_loss
                self._save_checkpoint()
            print("=" * 80)

    def _train_step(self) -> float:
        self.transformer_model.train()
        losses: list[float] = []
        pbar = tqdm.tqdm(self.train_dl)
        for encoder_input, decoder_input, tgt in pbar:
            self.optimizer.zero_grad()
            pred_logits = self.transformer_model(
                encoder_input.to(self.device),
                decoder_input.to(self.device),
            )
            pred_logits = pred_logits.reshape(-1, pred_logits.size(-1))
            tgt = tgt.to(self.device).reshape(-1, tgt.size(-1))
            loss: torch.Tensor = self.loss_fn(pred_logits, tgt)
            loss_number = loss.detach().cpu().item()
            losses.append(loss_number)
            loss.backward()
            self.optimizer.step()
            lr = self._update_lr()
            self.summary_writer.add_scalar("learning_rate", lr, self.current_step)
            pbar.set_description(f"loss: {loss_number:.4f} lr: {lr:.6f}")
        return sum(losses) / len(losses)

    def _test_step(self) -> tuple[float, float]:
        self.transformer_model.eval()
        with torch.inference_mode():
            losses: list[float] = []
            val_scores: list[float] = []
            pbar = tqdm.tqdm(self.valid_dl)
            for encoder_input, decoder_input, tgt in pbar:
                pred_logits = self.transformer_model(
                    encoder_input.to(self.device),
                    decoder_input.to(self.device),
                )
                pred_logits = pred_logits.reshape(-1, pred_logits.size(-1))
                tgt = tgt.to(self.device).reshape(-1, tgt.size(-1))
                loss: torch.Tensor = self.loss_fn(
                    pred_logits,
                    tgt,
                )
                loss_number = loss.detach().cpu().item()
                losses.append(loss_number)
                validation = self.validation_metric(pred_logits, tgt)
                pbar.set_description(f"loss: {loss_number:.4f} val: {validation:.4f}")
                val_scores.append(validation)
            return sum(losses) / len(losses), sum(val_scores) / len(val_scores)

    def _update_lr(self) -> float:
        self.current_step += 1
        lr = self.d_model**-0.5 * min(
            self.current_step**-0.5, self.current_step * self.warmup_steps**-1.5
        )
        for param in self.optimizer.param_groups:
            param["lr"] = lr
        return lr

    def _save_checkpoint(self) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.transformer_model.state_dict(),
            self.checkpoint_path,
        )
