import argparse
import os
from pathlib import Path

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
import tqdm
from torchmetrics.text import BLEUScore

from attention_is_all_you_need.implementation.src import trainer
from attention_is_all_you_need.implementation.src.data import (
    MAX_LENGTH,
    VOCAB_SIZE,
    create_bpe_vocab,
    create_dataloader,
    get_tokenizer,
)
from attention_is_all_you_need.implementation.src.trainer import Trainer
from attention_is_all_you_need.implementation.src.transformer_model import (
    TransformerModel,
)


BASE_DIR = Path(__file__).parent.parent
BATCH_SIZE = 512


def test(model_path: str, device: str) -> None:
    tokenizer = get_tokenizer()
    transformer = TransformerModel(VOCAB_SIZE, MAX_LENGTH, 6, 512, 2048, 8, device)
    transformer.load_state_dict(torch.load(model_path, map_location=device))
    transformer.to(device)
    transformer.eval()

    with torch.inference_mode():
        while True:
            try:
                sentence = input("Enter sentence to test (or 'q' to quit): ")
                if sentence.lower() == "q":
                    break

                # Encode and pad input sentence
                encoder_tokens = tokenizer.encode(sentence).ids
                print("\nEncoder input:", tokenizer.decode(encoder_tokens))

                encoder_input_tokens = (
                    torch.tensor(
                        _pad_sequence(encoder_tokens, MAX_LENGTH, pad_token_id=0),
                        dtype=torch.long,
                    )
                    .to(device)
                    .unsqueeze(0)
                )

                # Start with CLS token and explicitly decode it to verify
                decoder_tokens = torch.tensor(
                    [[2] + [0] * (MAX_LENGTH - 1)], dtype=torch.long
                ).to(device)
                generated = []
                current_length = 1  # Start after CLS token

                print(
                    f"Initial decoder input:",
                    tokenizer.decode(decoder_tokens[0].tolist()),
                )

                for i in range(MAX_LENGTH - 1):
                    print(f"\nStep {i+1}")
                    print(
                        "Current decoder:",
                        tokenizer.decode(decoder_tokens[0, :current_length].tolist()),
                    )

                    pred_logits = transformer(encoder_input_tokens, decoder_tokens)
                    next_token_logits = pred_logits[0, current_length - 1, :]

                    # Apply softmax with temperature
                    temperature = 0.7
                    next_token_logits = next_token_logits / temperature
                    probs = torch.softmax(next_token_logits, dim=-1)

                    # Zero out probabilities for special tokens except SEP
                    probs[0] = 0  # PAD
                    probs[1] = 0  # UNK
                    probs[2] = 0  # CLS
                    probs[4] = 0  # MASK

                    # Renormalize probabilities
                    probs = probs / probs.sum()

                    # Get top 5 predictions
                    top_probs, top_indices = torch.topk(probs, 5)
                    print("\nTop 5 predictions:")
                    for prob, idx in zip(
                        top_probs.cpu().numpy(), top_indices.cpu().numpy()
                    ):
                        token = tokenizer.decode([idx])
                        print(f"Token: {token} (ID: {idx}), Probability: {prob:.4f}")

                    # Sample from the distribution
                    pred_token_id = torch.multinomial(probs, num_samples=1).item()
                    generated.append(pred_token_id)

                    if pred_token_id == 3:  # SEP token
                        print("Generated SEP token, stopping.")
                        break

                    # Update decoder input tensor directly
                    decoder_tokens[0, current_length] = pred_token_id
                    current_length += 1

                output = tokenizer.decode(generated)

                print(f"\nInput: {sentence}")
                print(f"Prediction: {output}")
                print(f"Generated token IDs: {generated}")
                print("=" * 50)

            except Exception as e:
                print(f"Error occurred: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback

                traceback.print_exc()
                continue


def _pad_sequence(seq: list[int], max_length: int, pad_token_id: int) -> list[int]:
    """Pad a sequence to max_length with pad_token_id."""
    return seq + [pad_token_id] * (max_length - len(seq))


def train(device: str) -> None:
    train_dataloader = create_dataloader(
        str(Path(__file__).parent / ".." / "data" / "wmt14_translate_de-en_train.csv"),
        MAX_LENGTH,
        BATCH_SIZE,
    )
    test_dataloader = create_dataloader(
        str(BASE_DIR / "data" / "wmt14_translate_de-en_validation.csv"),
        MAX_LENGTH,
        BATCH_SIZE,
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
        tokenizer=get_tokenizer(),
        warmup_steps=4000,
        num_epochs=10,
        summary_writer=summary_writer,
        checkpoint_path=str(BASE_DIR / "best_model.pth"),
    )
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    trainer.train()


def eval(device: str, model_path: str) -> None:
    test_dataloader, _ = create_dataloader(
        str(BASE_DIR / "data" / "wmt14_translate_de-en_test.csv"),
        MAX_LENGTH,
        BATCH_SIZE,
    )
    transformer = TransformerModel(VOCAB_SIZE, MAX_LENGTH, 6, 512, 2048, 8, device)
    transformer.load_state_dict(
        torch.load(model_path, weights_only=False, map_location=device)
    )
    bleu_scores = []
    bleu_fn = BLEUScore().to(device)
    tokenizer = get_tokenizer()
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
    tokenizer = get_tokenizer()
    transformer = TransformerModel(VOCAB_SIZE, MAX_LENGTH, 6, 512, 2048, 8, device)
    transformer.load_state_dict(torch.load(model_path, map_location=device))
    transformer.to(device)
    transformer.eval()

    with torch.inference_mode():
        while True:
            try:
                sentence = input("Enter sentence to test (or 'q' to quit): ")
                if sentence.lower() == "q":
                    break

                # Encode input sentence
                encoder_input_tokens = (
                    torch.tensor(tokenizer.encode(sentence).ids, dtype=torch.long)
                    .to(device)
                    .unsqueeze(0)
                )

                # Start with just the CLS token
                decoder_tokens = [tokenizer.token_to_id("[CLS]")] + [
                    tokenizer.token_to_id("[PAD]")
                ] * 49
                decoder_input_tokens = torch.tensor(
                    [decoder_tokens], dtype=torch.long
                ).to(device)

                generated_tokens = []
                max_length = 100

                for _ in range(max_length):
                    padded_decoder_tokens = _pad_sequence(
                        decoder_tokens, MAX_LENGTH, tokenizer.token_to_id("[PAD]")
                    )
                    decoder_input_tokens = torch.tensor(
                        [padded_decoder_tokens], dtype=torch.long
                    ).to(device)

                    # Get model predictions
                    pred_logits = transformer(
                        encoder_input_tokens, decoder_input_tokens
                    )

                    # Get predictions for the last valid position
                    last_token_logits = pred_logits[:, len(decoder_tokens) - 1, :]
                    pred_token = torch.argmax(last_token_logits, dim=-1)

                    # Convert to scalar
                    pred_token_id = pred_token.item()
                    generated_tokens.append(pred_token_id)

                    # Stop if we predict SEP token
                    if pred_token_id == tokenizer.token_to_id("[SEP]"):
                        break

                    # Add the predicted token to decoder input
                    decoder_tokens.append(pred_token_id)

                # Decode the generated tokens
                output = tokenizer.decode(generated_tokens)

                print(f"Input: {sentence}")
                print(f"Prediction: {output}")
                print("=" * 50)

            except Exception as e:
                print(f"Error occurred: {str(e)}")
                continue


def run(device: str, mode: str) -> None:
    print(device)
    if mode == "train":
        train(device)
    elif mode == "eval":
        eval(device, str(BASE_DIR / "best_model.pth"))
    else:
        test(str(BASE_DIR / "best_model.pth"), device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/cuda/mps)")
    parser.add_argument(
        "--mode",
        default="train",
        help="Mode of application (train/eval/test)",
        choices=["train", "eval", "test"],
    )
    args = parser.parse_args()
    device = args.device
    create_bpe_vocab()
    run(device, args.mode)
