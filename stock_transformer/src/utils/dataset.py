import datetime
from typing import Any, Hashable, Literal, TypedDict
import pandas as pd
import torch


class StockDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ds: pd.DataFrame,
        window_period: int,
        normalize_params: "_NORMALIZE_PARAMS_TYPE | None" = None,
    ) -> None:
        super().__init__()
        df, self.normalize_params = self._normalize_data(ds, normalize_params)
        self.ds = self._setup_df(df, window_period)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

    @staticmethod
    def _setup_df(df: pd.DataFrame, window_period: int):
        """
        return shape: list[(windows_period, 11)]
        """
        if "open_date" in df.columns:
            df["open_date"] = pd.to_datetime(df["open_date"])
        if "open_date" in df.columns:
            df = df.sort_values("open_date").reset_index(drop=True)
        ema_periods = [5, 8, 9, 12, 34, 50]
        for period in ema_periods:
            df[f"ema_{period}"] = df["c"].ewm(span=period, adjust=False).mean()
        df.dropna(inplace=True)
        feature_cols = [
            "o",
            "h",
            "l",
            "c",
            "v",
            "ema_5",
            "ema_8",
            "ema_9",
            "ema_12",
            "ema_34",
            "ema_50",
        ]
        features = df[feature_cols].values
        windows = []
        for i in range(len(features) - window_period + 1):
            window = features[i : i + window_period]
            windows.append(torch.tensor(window, dtype=torch.float64))

        return windows

    @staticmethod
    def _normalize_data(
        df: pd.DataFrame,
        normalize_params: "_NORMALIZE_PARAMS_TYPE | None" = None,
    ):
        if not normalize_params:
            normalize_params = {}
            for col in df.columns:
                normalize_params[col] = {"mean": df[col].mean(), "std": df[col].std()}
        for col in df.columns:
            df[col] = (df[col] - normalize_params[col]["mean"]) / normalize_params[col][
                "std"
            ]
        return df, normalize_params


_NORMALIZE_PARAMS_TYPE = dict[str, dict[Literal["mean", "std"], float]]
