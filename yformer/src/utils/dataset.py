from typing import Literal
from datasets import load_dataset
import numpy
import torch
from sklearn.preprocessing import StandardScaler


def create_dataloader(
    input_window: int,
    pred_window: int,
    split: Literal["train", "validation", "test"],
    batch_size: int,
):
    ds = ETTHourDataset(input_window, pred_window, split)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, pin_memory=True
    )


class ETTHourDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_window: int,
        pred_window: int,
        split: Literal["train", "validation", "test"],
    ):
        ds = load_dataset(
            "ett", "h1", multivariate=True, split=split, trust_remote_code=True
        )
        self.input_window = input_window
        self.pred_window = pred_window
        self.ds = numpy.array(ds.to_list()[0]["target"])  # type: ignore
        self.scaler = StandardScaler()
        if split == "train":
            self.ds = self.scaler.fit_transform(self.ds.T).T
        else:
            self.ds = self.scaler.transform(self.ds.T).T
        self.ds = torch.tensor(self.ds)
        self.windows = self._create_training_windows()

    def __len__(self):
        return len(self.windows)

    def _create_training_windows(self):
        windows = []
        for i in range(self.ds.shape[1] - self.input_window - self.pred_window + 1):
            x = self.ds[:, i : (i + self.input_window)]
            y = self.ds[
                :, (i + self.input_window) : (i + self.input_window + self.pred_window)
            ]
            windows.append((x, y))
        return windows

    def __getitem__(self, idx):
        x, y = self.windows[idx]
        return {
            "input": x.float(),
            "target": y.float(),
        }
