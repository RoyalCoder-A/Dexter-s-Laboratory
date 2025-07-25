import datetime
from pathlib import Path
import time
from typing import Literal
import pandas as pd
import torch
from binance.client import Client


def get_dataloader(
    train_ds_path: str, test_ds_path: str, window_period: int, batch_size: int
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_df = pd.read_csv(train_ds_path)
    test_df = pd.read_csv(test_ds_path)
    train_ds = StockDataset(train_df, window_period)
    test_ds = StockDataset(test_df, window_period, train_ds.normalize_params)
    return (
        torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, pin_memory=True
        ),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size),
    )


class StockDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ds: pd.DataFrame,
        window_period: int,
        normalize_params: "_NORMALIZE_PARAMS_TYPE | None" = None,
    ) -> None:
        super().__init__()
        df, self.normalize_params = self._normalize_data(ds, normalize_params)
        self.window_period = window_period
        self.ds: list[torch.Tensor] = self._setup_df(df, window_period * 2)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        item = self.ds[idx]
        src = item[: self.window_period]  # (window_period, 11)
        tgt = item[self.window_period :]  # (window_period, 11)
        dec_src = torch.concat(
            ((torch.ones((1, 11)).float() * -1), tgt[:-1, :])
        )  # (window_period, 11)
        dec_tgt = tgt  # (window_period, 11)
        return (
            src,
            dec_src,
            dec_tgt,
        )

    @staticmethod
    def _setup_df(df: pd.DataFrame, window_period: int):
        """
        return shape: list[(windows_period * 2, 11)]
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
        windows: list[torch.Tensor] = []
        for i in range(len(features) - window_period + 1):
            window = features[i : i + window_period]
            windows.append(torch.tensor(window).float())

        return windows

    @staticmethod
    def _normalize_data(
        df: pd.DataFrame,
        normalize_params: "_NORMALIZE_PARAMS_TYPE | None" = None,
    ):
        if not normalize_params:
            normalize_params = {}
            for col in df.columns:
                if col == "open_date":
                    continue
                normalize_params[col] = {"mean": df[col].mean(), "std": df[col].std()}
        for col in df.columns:
            if col == "open_date":
                continue
            df[col] = (df[col] - normalize_params[col]["mean"]) / normalize_params[col][
                "std"
            ]
        return df, normalize_params


_NORMALIZE_PARAMS_TYPE = dict[str, dict[Literal["mean", "std"], float]]


def fetch_eth_10m_ohlcv_binance(start_dt, end_dt, api_key=None, api_secret=None):
    """
    Fetches ETH/USDT 10-minute OHLCV data from Binance between start_dt and end_dt.
    Returns a DataFrame with columns: open_date, o, h, l, c, v.
    """
    client = Client(api_key, api_secret)
    limit = 1000  # max candles per request

    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    all_klines = []

    while start_ts < end_ts:
        klines = client.get_klines(
            symbol="ETHUSDT",
            interval=Client.KLINE_INTERVAL_15MINUTE,
            startTime=start_ts,
            endTime=end_ts,
            limit=limit,
        )
        if not klines:
            break
        all_klines += klines
        # Advance to just after the last fetched open_time
        start_ts = klines[-1][0] + 1
        time.sleep(0.2)  # to respect rate limits

    # Define column names for the full kline data
    cols = [
        "open_time",
        "o",
        "h",
        "l",
        "c",
        "v",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(all_klines, columns=cols)

    # Convert and filter DataFrame
    df["open_date"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df[["open_date", "o", "h", "l", "c", "v"]].astype(
        {"o": float, "h": float, "l": float, "c": float, "v": float}
    )
    df.set_index("open_date", inplace=True)
    return df


if __name__ == "__main__":
    end_dt = datetime.datetime.now(datetime.timezone.utc)
    start_dt = end_dt - datetime.timedelta(days=365)

    print(
        f"Fetching ETH/USDT 10m data from {start_dt.isoformat()} to {end_dt.isoformat()}..."
    )
    df = fetch_eth_10m_ohlcv_binance(start_dt, end_dt)
    parent_dir = Path(__file__).parent.parent.parent / "data"
    parent_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(parent_dir / "train.csv")

    end_dt = start_dt
    start_dt = end_dt - datetime.timedelta(days=30)
    print(
        f"Fetching ETH/USDT 10m data from {start_dt.isoformat()} to {end_dt.isoformat()}..."
    )
    df = fetch_eth_10m_ohlcv_binance(start_dt, end_dt)
    df.to_csv(parent_dir / "test.csv")
