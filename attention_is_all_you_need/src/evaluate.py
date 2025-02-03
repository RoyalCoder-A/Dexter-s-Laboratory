from pathlib import Path
import random
from typing import Literal

import torch
from attention_is_all_you_need.src.utils.tokenizer import (
    DATA_DIR_PATH,
    MAX_LENGTH,
    VOCAB_SIZE,
    get_tokenizer,
)
from attention_is_all_you_need.src.utils.transformer_model import TransformerModel
from datasets import load_dataset


def evaluate(
    device: Literal["cpu", "cuda", "mps"], data_path: Path = DATA_DIR_PATH
) -> None:
    tokenizer = get_tokenizer(data_path)
    transformer_model = TransformerModel(VOCAB_SIZE, 512, 6, 2048, 8, 0.1)
    transformer_model.load_state_dict(
        torch.load(
            data_path / "checkpoints" / "cuda__512__12.pt",
            weights_only=False,
            map_location=device,
        )
    )
    transformer_model = transformer_model.to(device)
    if device != "mps":
        transformer_model.compile()
    transformer_model.eval()
    dataset = load_dataset("wmt14", "de-en", split="train")
    with torch.inference_mode():
        while True:
            test_sentence = tokenizer.encode("Mew Mew!").ids
            input_seq = input("Enter sentence: (or q to exit, r for random)")
            if input_seq == "q":
                break
            if input_seq == "r":
                item = random.choice(next(dataset.iter(batch_size=100))["translation"])  # type: ignore
                print(f"Random sentence: {item['en']}")
                input_seq = item["de"]
            print(f"Input: {input_seq}")
            encoder_ids = (
                torch.tensor([tokenizer.encode(input_seq).ids]).to(device).long()
            )
            decoder_ids = (
                [tokenizer.token_to_id("[CLS]")]
                + test_sentence[:7]
                + [tokenizer.token_to_id("[PAD]")] * (MAX_LENGTH - 8)
            )
            final_result_ids = []
            for i in range(MAX_LENGTH - 1):
                result_logits = transformer_model(
                    encoder_ids,
                    torch.tensor(decoder_ids).to(device).unsqueeze(0).long(),
                )
                result_ids = torch.argmax(result_logits, dim=-1).squeeze(0)
                current_id = result_ids[i]
                print(f"Current token: {tokenizer.id_to_token(current_id)}")
                decoder_ids[i + 1] = current_id
                final_result_ids.append(current_id)
            print(f"Result: {tokenizer.decode(final_result_ids)}")
