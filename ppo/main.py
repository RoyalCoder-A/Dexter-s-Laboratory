from pathlib import Path
from ppo.src import train


if __name__ == "__main__":
    train.train(
        8,
        128,
        32 * 8,
        3,
        0.99,
        0.95,
        0.1,
        0.01,
        1,
        10000,
        2.5 * 10e-4,
        Path("./data"),
        "cpu",
        1000,
        Path("./logs"),
    )
