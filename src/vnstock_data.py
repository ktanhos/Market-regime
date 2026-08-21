from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .data import RAW_DIR, save_parquet, standardize_ohlcv


class VNStockAdapter:
    def __init__(self, source: str = "VCI"):
        self.source = source

    def _quote(self, symbol: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
        from vnstock.api.quote import Quote
        q = Quote(symbol=symbol, source=self.source)
        errors = []
        for method in ["history", "ohlcv"]:
            try:
                fn = getattr(q, method)
                try:
                    df = fn(start=start, end=end, interval="1D")
                except TypeError:
                    df = fn(start=start, end=end)
                return standardize_ohlcv(df)
            except Exception as exc:
                errors.append(f"{method}: {type(exc).__name__}: {exc}")
        raise ConnectionError(" | ".join(errors))

    def fetch_equity(self, symbol: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
        try:
            return self._quote(symbol, start, end)
        except Exception as exc:
            raise ConnectionError(f"Không thể lấy dữ liệu {symbol} từ VNStock API mới: {exc}") from exc

    def fetch_index(self, symbol: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
        try:
            return self._quote(symbol, start, end)
        except Exception as exc:
            raise ConnectionError(f"Không thể lấy dữ liệu {symbol} từ VNStock API mới: {exc}") from exc


def _history_start(asset_type: str) -> str:
    if asset_type == "index":
        return (date.today() - timedelta(days=365 * 8)).isoformat()
    return (date.today() - timedelta(days=365 * 1)).isoformat()


def update_market_dataset(symbol: str, dataset_name: Optional[str] = None, asset_type: str = "index", start: Optional[str] = None, end: Optional[str] = None, source: str = "VCI") -> str:
    adapter = VNStockAdapter(source=source)
    start = start or _history_start(asset_type)
    end = end or date.today().isoformat()
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
    return update_market_dataset(symbol, dataset_name, asset_type, start=start, source=source)
