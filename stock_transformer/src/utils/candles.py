import datetime
from binance.client import Client
import pandas as pd
import time

def fetch_15m_ohlcv_binance(
    symbol: str,
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
    api_key=None,
    api_secret=None,
):
    client = Client(api_key, api_secret)
    limit = 1000  # max candles per request

    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    all_klines = []

    while start_ts < end_ts:
        klines = client.get_klines(
            symbol=symbol,
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
    ema_periods = [5, 8, 9, 12, 34, 50]
    for period in ema_periods:
        df[f"ema_{period}"] = df["c"].ewm(span=period, adjust=False).mean()
    df.dropna(inplace=True)
    df.sort_values(by="open_date", ascending=True, inplace=True)
    df["target"] = df["c"].pct_change().shift(-1)
    df.dropna(inplace=True)
    df["symbol"] = symbol
    df.set_index(["open_date", "symbol"], drop=True, inplace=True)
    return df