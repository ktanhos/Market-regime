from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .data import RAW_DIR, save_parquet, standardize_ohlcv


class VNStockAdapter:
    """Single access point for VNStock market data."""

    def __init__(self, source: str = "VCI"):
        self.source = source

    def _market(self):
        try:
            from vnstock import Market
            return Market(source=self.source)
        except Exception:
            return None

    def fetch_index(self, symbol: str, start: str = "2010-01-01", end: Optional[str] = None) -> pd.DataFrame:
        market = self._market()
        if market is not None:
            try:
                return standardize_ohlcv(market.index(symbol).ohlcv(start=start, end=end))
            except Exception:
                pass
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol, source=self.source)
        return standardize_ohlcv(stock.quote.history(start=start, end=end, interval="1D"))

    def fetch_equity(self, symbol: str, start: str = "2010-01-01", end: Optional[str] = None) -> pd.DataFrame:
        market = self._market()
        if market is not None:
            try:
                return standardize_ohlcv(market.equity(symbol).ohlcv(start=start, end=end))
            except Exception:
                pass
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol, source=self.source)
        return standardize_ohlcv(stock.quote.history(start=start, end=end, interval="1D"))


def update_market_dataset(symbol: str, dataset_name: Optional[str] = None, asset_type: str = "index", start: str = "2010-01-01", end: Optional[str] = None, source: str = "VCI") -> str:
    adapter = VNStockAdapter(source=source)
    if asset_type == "index":
        df = adapter.fetch_index(symbol, start=start, end=end)
    elif asset_type == "equity":
        df = adapter.fetch_equity(symbol, start=start, end=end)
    else:
        raise ValueError("asset_type must be index or equity")
    name = dataset_name or symbol.lower()
    return str(save_parquet(df, name, data_dir=RAW_DIR))


def incremental_update(symbol: str, dataset_name: str, asset_type: str = "index", source: str = "VCI", overlap_days: int = 15) -> str:
    data_path = RAW_DIR / f"{dataset_name}.parquet"
    if not data_path.exists():
        return update_market_dataset(symbol, dataset_name, asset_type, source=source)

    old = standardize_ohlcv(pd.read_parquet(data_path))
    last_date = pd.to_datetime(old["date"]).max()
    start = (last_date - timedelta(days=overlap_days)).strftime("%Y-%m-%d")
    adapter = VNStockAdapter(source=source)
    if asset_type == "index":
        new = adapter.fetch_index(symbol, start=start)
    elif asset_type == "equity":
        new = adapter.fetch_equity(symbol, start=start)
    else:
        raise ValueError("asset_type must be index or equity")

    combined = pd.concat([old, new], ignore_index=True)
    combined = standardize_ohlcv(combined)
    save_parquet(combined, dataset_name, data_dir=RAW_DIR)
    return str(data_path)
