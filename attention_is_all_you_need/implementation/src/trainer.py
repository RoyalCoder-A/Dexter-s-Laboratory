import math
import torch
from torch.utils.tensorboard.writer import SummaryWriter
import tqdm
from torchmetrics.text import BLEUScore
from tokenizers.implementations import CharBPETokenizer

from attention_is_all_you_need.implementation.src.data import Wmt14Dataset


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        train_data_loader: torch.utils.data.DataLoader,
        test_data_loader: torch.utils.data.DataLoader,
        summary_writer: SummaryWriter,
        d_model: int,
        warmup_steps: int,
        batch_size: int,
        num_epochs: int,
        tokenizer: CharBPETokenizer,
        checkpoint_path: str,
        device: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.summary_writer = summary_writer
        self.num_epochs = num_epochs
        self.device = device
        self.gradient_clip_val = min(5.0, math.sqrt(batch_size / 256))
        self.best_loss = float("inf")
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.tokenizer = tokenizer
        self.bleu_fn = BLEUScore().to(device)
        self.checkpoint_path = checkpoint_path

    def train(self):
        for i in range(self.num_epochs):
            print(f"Epoch {i+1} from {self.num_epochs}")
            train_loss = self._train_step()
            test_loss, bleu_score = self._test_step()
            print(f"Train loss: {train_loss:.4f}")
            print(f"Test loss: {test_loss:.4f}")
            print(f"Test bleu score: {bleu_score:.4f}")
            if test_loss < self.best_loss:
                print("Saving")
                self.best_loss = test_loss
                torch.save(self.model.state_dict(), self.checkpoint_path)
            print("=" * 20)
            self.summary_writer.add_scalar("train_loss", train_loss, i)
            self.summary_writer.add_scalar("test_loss", test_loss, i)
            self.summary_writer.add_scalar("bleu_score", bleu_score, i)

    def _train_step(self):
        losses = []
        self.model.train()
        pbar = tqdm.tqdm(self.train_data_loader, desc="Training")
        for encoder_x, decoder_x, y, _ in pbar:
            lr = self._update_learning_rate()
            self.summary_writer.add_scalar("learning_rate", lr, self.current_step)
            encoder_x, decoder_x, y = (
                encoder_x.to(self.device),
                decoder_x.to(self.device),
                y.to(self.device).long(),
            )
            pred_logits = self.model(encoder_x, decoder_x)
            loss: torch.Tensor = self.loss_fn(
                pred_logits.view(
                    -1, pred_logits.size(-1)
                ),  # reshape to [batch_size * seq_len, vocab_size]
                y.view(-1),  # reshape to [batch_size * seq_len]
            )
            loss_number = loss.detach().cpu().item()
            pbar.set_description(f"Training loss: {loss_number:.4f}")
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_val
            )
            self.optimizer.step()
            losses.append(loss_number)
        return torch.tensor(losses).mean().item()

    def _test_step(self):
        losses = []
        bleu_scores = []
        self.model.eval()
        with torch.inference_mode():
            for encoder_x, decoder_x, y, tgt in tqdm.tqdm(
                self.test_data_loader, desc="Testing"
            ):
                encoder_x, decoder_x, y = (
                    encoder_x.to(self.device),
                    decoder_x.to(self.device),
                    y.to(self.device),
                )
                pred_logits = self.model(encoder_x, decoder_x)
                pred_probs = torch.nn.functional.softmax(pred_logits, dim=-1)
                preds = torch.argmax(pred_probs, dim=-1)
                bleu_score = self._calculate_bleu(tgt, preds)
                bleu_scores.append(bleu_score)
                loss: torch.Tensor = self.loss_fn(
                    pred_logits.view(
                        -1, pred_logits.size(-1)
                    ),  # reshape to [batch_size * seq_len, vocab_size]
                    y.view(-1).long(),  # reshape to [batch_size * seq_len]
                )
                losses.append(loss.detach().cpu().item())
        return (
            torch.tensor(losses).mean().item(),
            torch.tensor(bleu_scores).mean().item(),
        )

    def _calculate_bleu(self, tgt: tuple, pred: torch.Tensor) -> float:
        pred_tokens = [
            self.tokenizer.decode(list(pred[i].detach().cpu().numpy()))
            for i in range(pred.size(0))
        ]
        return self.bleu_fn(pred_tokens, [[x] for x in tgt])

    def _update_learning_rate(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step += 1
        return lr

    def get_lr(self) -> float:
        step = max(1, self.current_step)
        return (self.d_model**-0.5) * min(step**-0.5, step * self.warmup_steps**-1.5)
