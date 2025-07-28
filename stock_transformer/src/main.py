from pathlib import Path
from stock_transformer.src import train


if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent / "data"
    train.train(
        32,
        20,
        "mps",
        11,
        str(parent_dir / "train.csv"),
        str(parent_dir / "test.csv"),
        10,
        parent_dir,
        "train_4",
    )
