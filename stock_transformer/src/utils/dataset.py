from dataclasses import dataclass
import datetime
from pathlib import Path
import pickle
from typing import Literal, cast
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from stock_transformer.src.utils.candles import fetch_15m_ohlcv_binance

WINDOW_PERIOD = 10
FEATURES_COUNT = 11


def get_dataloader(
    train_ds_path: Path, test_ds_path: Path, batch_size: int
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_dl = torch.utils.data.DataLoader(
        StockDataset(train_ds_path),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_dl = torch.utils.data.DataLoader(
        StockDataset(test_ds_path),
        batch_size=batch_size,
    )
    return train_dl, test_dl


class StockDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path: Path) -> None:
        super().__init__()
        self.ds_path = ds_path
        self.ds = load_ds(ds_path)

    def __len__(self):
        return len(self.ds.ds)

    def __getitem__(self, idx: int):
        item = self.ds.ds[idx]
        encoder_input = item[0]  # (windows_size, feature_size)
        decoder_input = torch.concat(
            [torch.ones((1, 1)) * -1, item[1][:-1]]
        )  # (window_size, 1)
        tgt = item[1]  # (window_size, 1)
        return encoder_input, decoder_input, tgt


_NORMALIZE_PARAMS_TYPE = dict[str, dict[Literal["mean", "std"], float]]


def _normalize_data(
    df: pd.DataFrame,
    normalize_params: "_NORMALIZE_PARAMS_TYPE | None" = None,
):
    if not normalize_params:
        normalize_params = {}
        for col in df.columns:
            if col in ("open_date", "symbol"):
                continue
            normalize_params[col] = {
                "mean": df[col].replace([np.inf, -np.inf], np.nan).mean(),
                "std": df[col].replace([np.inf, -np.inf], np.nan).std(),
            }
    for col in df.columns:
        if col == "open_date":
            continue
        df[col] = (df[col] - normalize_params[col]["mean"]) / normalize_params[col][
            "std"
        ]
    return df, normalize_params


def _prepare_dataset(
    df: pd.DataFrame, normalized_params: _NORMALIZE_PARAMS_TYPE | None = None
):
    df, normalized_params = _normalize_data(df, normalized_params)
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
    target_cols = ["target"]
    result: list[tuple[torch.Tensor, torch.Tensor]] = []
    pbar = tqdm(df.groupby("symbol"))
    for symbol, group in pbar:
        pbar.set_description(f"Processing {symbol}")
        features = group[feature_cols].values
        targets = group[target_cols].values
        for i in range(len(features) - WINDOW_PERIOD + 1):
            feature_window = features[i : i + WINDOW_PERIOD]
            target_window = targets[i : i + WINDOW_PERIOD]
            result.append(
                (
                    torch.tensor(feature_window).float(),
                    torch.tensor(target_window).float(),
                )
            )
    return result, normalized_params


@dataclass
class _Dataset:
    ds: list[tuple[torch.Tensor, torch.Tensor]]
    normalized_params: _NORMALIZE_PARAMS_TYPE


def load_ds(path: Path) -> _Dataset:
    with open(path, "ab") as file:
        obj = pickle.load(file)
    return cast(_Dataset, obj)


if __name__ == "__main__":
    crypto_symbols = [
        "BTCUSDT",  # Bitcoin
        "ETHUSDT",  # Ethereum
        "BNBUSDT",  # Binance Coin
        "SOLUSDT",  # Solana
        "ADAUSDT",  # Cardano
        "XRPUSDT",  # Ripple
        "DOGEUSDT",  # Dogecoin
        "LINKUSDT",  # Chainlink
        "AVAXUSDT",  # Avalanche
        "MATICUSDT",  # Polygon
    ]
    train_start_date = datetime.datetime.now(
        datetime.timezone.utc
    ) - datetime.timedelta(days=3 * 365)
    train_end_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=90
    )

    test_start_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=90
    )
    test_end_date = datetime.datetime.now(datetime.timezone.utc)

    train_result = pd.DataFrame()
    test_result = pd.DataFrame()
    for symbol in tqdm(crypto_symbols):
        train_df = fetch_15m_ohlcv_binance(symbol, train_start_date, train_end_date)
        test_df = fetch_15m_ohlcv_binance(symbol, test_start_date, test_end_date)
        train_result = pd.concat([train_result, train_df])
        test_result = pd.concat([test_result, test_df])

    train_ds, normalized_params = _prepare_dataset(train_df)
    test_ds, _ = _prepare_dataset(test_result, normalized_params)

    train_obj = _Dataset(train_ds, normalized_params)
    test_obj = _Dataset(test_ds, normalized_params)

    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "train.pkl", "ab") as f:
        pickle.dump(train_obj, f)

    with open(data_dir / "test.pkl", "ab") as f:
        pickle.dump(test_obj, f)
