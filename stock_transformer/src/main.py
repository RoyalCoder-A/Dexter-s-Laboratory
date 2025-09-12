from pathlib import Path
from stock_transformer.src import train
from stock_transformer.src.utils.dataset import _Dataset


if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent / "data"
    train.train(
        batch_size=512,
        epochs=20,
        device="mps",
        train_ds_path=parent_dir / "train_3_target.pkl",
        test_ds_path=parent_dir / "test_3_target.pkl",
        data_path=parent_dir,
        train_name="train_4",
    )
