import torch

from attention_is_all_you_need.implementation.src.data import get_dataset

if __name__ == "__main__":
    print(torch.mps.is_available())
    print(get_dataset())
