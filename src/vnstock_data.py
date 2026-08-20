from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .data import RAW_DIR, save_parquet, standardize_ohlcv


class VNStockAdapter:
    """Single access point for VNStock market data.

    The adapter tries the current Market interface first and preserves the
    underlying errors so deployment problems are visible instead of silently
    falling through to an incompatible legacy call.
    """

    def __init__(self, source: str = "VCI"):
        self.source = source

    def fetch_index(self, symbol: str, start: str = "2010-01-01", end: Optional[str] = None) -> pd.DataFrame:
        errors = []

        try:
            from vnstock import Market
            market = Market(source=self.source)
            df = market.index(symbol).ohlcv(start=start, end=end)
            return standardize_ohlcv(df)
        except Exception as exc:
            errors.append(f"Market interface: {type(exc).__name__}: {exc}")

        try:
            from vnstock import Vnstock
            stock = Vnstock().stock(symbol=symbol, source=self.source)
            df = stock.quote.history(start=start, end=end, interval="1D")
            return standardize_ohlcv(df)
        except Exception as exc:
            errors.append(f"Legacy interface: {type(exc).__name__}: {exc}")

        detail = " | ".join(errors)
        raise ConnectionError(
            f"Không thể lấy dữ liệu {symbol} từ VNStock. {detail}"
        )

    def fetch_equity(self, symbol: str, start: str = "2010-01-01", end: Optional[str] = None) -> pd.DataFrame:
        errors = []

        try:
            from vnstock import Market
            market = Market(source=self.source)
            df = market.equity(symbol).ohlcv(start=start, end=end)
            return standardize_ohlcv(df)
        except Exception as exc:
            errors.append(f"Market interface: {type(exc).__name__}: {exc}")

        try:
            from vnstock import Vnstock
            stock = Vnstock().stock(symbol=symbol, source=self.source)
            df = stock.quote.history(start=start, end=end, interval="1D")
            return standardize_ohlcv(df)
        except Exception as exc:
            errors.append(f"Legacy interface: {type(exc).__name__}: {exc}")

        detail = " | ".join(errors)
        raise ConnectionError(
            f"Không thể lấy dữ liệu {symbol} từ VNStock. {detail}"
        )


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
