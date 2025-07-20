import datetime
from typing import Any, Hashable, TypedDict
import pandas as pd
import torch

class StockDataset(torch.utils.data.Dataset):
    def __init__(self, ds: list[dict[Hashable, Any]], window_period: int) -> None:
        super().__init__()
        self.ds = self._setup_df(ds, window_period)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        return self.ds[idx]
    
    @staticmethod
    def _setup_df(data: list[dict[Hashable, Any]], window_period: int):
        """
        return shape: list[(windows_period, 11)]
        """
        df = pd.DataFrame(data, columns=["open_date", "o", "h", "l", "c", "v"])
        if 'open_date' in df.columns:
            df['open_date'] = pd.to_datetime(df['open_date'])
        if 'open_date' in df.columns:
            df = df.sort_values('open_date').reset_index(drop=True)
        ema_periods = [5, 8, 9, 12, 34, 50]
        for period in ema_periods:
            df[f'ema_{period}'] = df['c'].ewm(span=period, adjust=False).mean()
        df.dropna(inplace=True) 
        feature_cols = ['o', 'h', 'l', 'c', 'v', 'ema_5', 'ema_8', 'ema_9', 'ema_12', 'ema_34', 'ema_50']
        features = df[feature_cols].values
        windows = []
        for i in range(len(features) - window_period + 1):
            window = features[i:i + window_period]
            windows.append(torch.tensor(window, dtype=torch.float64))
        
        return windows
    

class _RecordType(TypedDict):
    open_date: datetime.datetime
    o: float
    h: float
    l: float
    c: float
    v: float
    ema_5: float
    ema_8: float
    ema_9: float
    ema_12: float
    ema_34: float
    ema_50: float
