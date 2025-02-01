from attention_is_all_you_need.src.utils.dataset import get_dataloader
from attention_is_all_you_need.src.utils.tokenizer import train_bpe_tokenizer


if __name__ == "__main__":
    dl = get_dataloader("train", batch_size=32)
    for data in dl:
        print(data)
        break
