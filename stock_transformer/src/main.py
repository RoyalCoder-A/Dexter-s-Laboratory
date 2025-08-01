from pathlib import Path
from stock_transformer.src import train


if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent / "data"
    train.train(
        batch_size=2,
        epochs=20,
        device="cpu",
        train_ds_path=parent_dir / "train.pkl",
        test_ds_path=parent_dir / "test.pkl",
        data_path=parent_dir,
        train_name="train_4",
    )
