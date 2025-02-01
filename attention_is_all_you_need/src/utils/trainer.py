from pathlib import Path
from typing import Literal

import torch
import tqdm
from attention_is_all_you_need.src.utils.tokenizer import get_tokenizer
from attention_is_all_you_need.src.utils.transformer_model import TransformerModel
from torchmetrics.text import BLEUScore
from torch.utils.tensorboard.writer import SummaryWriter


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
        self.loss_fn = torch.nn.CrossEntropyLoss().to(device)
        self.validation_metric = BLEUScore().to(device)
        self.optimizer = torch.optim.Adam(
            self.transformer_model.parameters(), betas=(0.9, 0.98), eps=1e-9
        )
        self.summary_writer = summary_writer
        self.tokenizer = get_tokenizer()
        self.current_step = 0
        self.checkpoint_path = checkpoint_path

    def train(self):
        self.current_step = 0
        best_loss = float("inf")
        for i in range(self.epochs):
            print(f"Epoch {i+1} from {self.epochs}")
            train_loss = self._train_step()
            test_loss, test_bleu = self._test_step()
            self.summary_writer.add_scalar("train_loss", train_loss, i)
            self.summary_writer.add_scalar("test_loss", test_loss, i)
            self.summary_writer.add_scalar("test_bleu", test_bleu, i)
            print(f"Train loss: {train_loss:.4f}")
            print(f"Test loss: {test_loss:.4f}")
            print(f"Test BLEU: {test_bleu:.4f}")
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(
                    self.transformer_model.state_dict(),
                    self.checkpoint_path,
                )
            print("=" * 80)

    def _train_step(self) -> float:
        self.transformer_model.train()
        losses: list[float] = []
        pbar = tqdm.tqdm(self.train_dl)
        for encoder_input, decoder_input, tgt in pbar:
            pred_logits = self.transformer_model(
                encoder_input.to(self.device),
                decoder_input.to(self.device),
            )
            loss: torch.Tensor = self.loss_fn(pred_logits, tgt.to(self.device))
            loss_number = loss.detach().cpu().item()
            losses.append(loss_number)
            self.optimizer.zero_grad()
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
            bleu_scores: list[float] = []
            pbar = tqdm.tqdm(self.valid_dl)
            for encoder_input, decoder_input, tgt in pbar:
                pred_logits = self.transformer_model(
                    encoder_input.to(self.device),
                    decoder_input.to(self.device),
                )
                loss: torch.Tensor = self.loss_fn(pred_logits, tgt.to(self.device))
                loss_number = loss.detach().cpu().item()
                losses.append(loss_number)
                bleu = self._calculate_bleu(pred_logits, tgt)
                pbar.set_description(f"loss: {loss_number:.4f} bleu: {bleu:.4f}")
                bleu_scores.append(bleu)
            return sum(losses) / len(losses), sum(bleu_scores) / len(bleu_scores)

    def _calculate_bleu(
        self, pred_logits: torch.Tensor, tgt_ids: torch.Tensor
    ) -> float:
        pred_ids = torch.argmax(pred_logits, dim=-1).detach().cpu().numpy()
        pred_strs = self.tokenizer.decode_batch(pred_ids)
        tgt_strs = self.tokenizer.decode_batch(tgt_ids.detach().cpu().numpy())
        bleu_score = self.validation_metric(
            [pred_str for pred_str in pred_strs], [[tgt_str] for tgt_str in tgt_strs]
        )
        return bleu_score

    def _update_lr(self) -> float:
        self.current_step += 1
        lr = self.d_model**-0.5 * min(
            self.current_step**-0.05, self.current_step * self.warmup_steps**-1.5
        )
        for param in self.optimizer.param_groups:
            param["lr"] = lr
        return lr
