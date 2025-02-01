import torch
from attention_is_all_you_need.src import train


if __name__ == "__main__":
    train.train(2, 1, "cpu")
