import argparse
import os
from pathlib import Path

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from tokenizers.implementations import CharBPETokenizer
import tqdm
from torchmetrics.text import BLEUScore

from attention_is_all_you_need.implementation.src import trainer
from attention_is_all_you_need.implementation.src.data import (
    MAX_LENGTH,
    VOCAB_SIZE,
    create_dataloader,
)
from attention_is_all_you_need.implementation.src.trainer import Trainer
from attention_is_all_you_need.implementation.src.transformer_model import (
    TransformerModel,
)


BASE_DIR = Path(__file__).parent.parent
BATCH_SIZE = 512


def train(
    device: str, train_dataloader: DataLoader, tokenizer: CharBPETokenizer
) -> None:
    test_dataloader, _ = create_dataloader(
        str(BASE_DIR / "data" / "wmt14_translate_de-en_validation.csv"),
        MAX_LENGTH,
        BATCH_SIZE,
        tokenizer=tokenizer,
    )
    transformer = TransformerModel(VOCAB_SIZE, MAX_LENGTH, 6, 512, 2048, 8, device)
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
        num_epochs=12,
        summary_writer=summary_writer,
    )
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    trainer.train()


def test(device: str, tokenizer: CharBPETokenizer, model_path: str) -> None:
    model_full_path = Path(os.getcwd()) / model_path
    test_dataloader, _ = create_dataloader(
        str(BASE_DIR / "data" / "wmt14_translate_de-en_validation.csv"),
        MAX_LENGTH,
        BATCH_SIZE,
        tokenizer=tokenizer,
    )
    transformer = TransformerModel(VOCAB_SIZE, MAX_LENGTH, 6, 512, 2048, 8, device)
    transformer.load_state_dict(torch.load(model_full_path, weights_only=False))
    bleu_scores = []
    bleu_fn = BLEUScore().to(device)
    transformer.eval()
    with torch.inference_mode():
        for encoder_x, decoder_x, y, tgt in tqdm.tqdm(test_dataloader, desc="Testing"):
            encoder_x, decoder_x, y = (
                encoder_x.to(device),
                decoder_x.to(device),
                y.to(device),
            )
            pred_logits = transformer(encoder_x, decoder_x)
            pred_probs = torch.nn.functional.softmax(pred_logits, dim=-1)
            preds = torch.argmax(pred_probs, dim=-1)
            pred_tokens = [
                tokenizer.decode(list(preds[i].detach().cpu().numpy()))
                for i in range(preds.size(0))
            ]
            bleu_scores.append(bleu_fn(pred_tokens, [[x] for x in tgt]))
            for target, prediction in zip(tgt, pred_tokens):
                print(f"Target: {target}")
                print(f"Prediction: {prediction}")
                print("=" * 50)
    print(f"BLEU Score: {sum(bleu_scores) / len(bleu_scores)}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/cuda/mps)")
    parser.add_argument("--test-model-path", default="", help="Path to test model")
    args = parser.parse_args()
    device = args.device
    print(device)
    train_dataloader, tokenizer = create_dataloader(
        str(Path(__file__).parent / ".." / "data" / "wmt14_translate_de-en_train.csv"),
        MAX_LENGTH,
        BATCH_SIZE,
        limit=10000,
    )
    if not args.test_model_path:
        train(device, train_dataloader, tokenizer)
    else:
        test(device, tokenizer, args.test_model_path)
