from __future__ import annotations

from datetime import date, timedelta
from typing import Optional
import pandas as pd

from .data import RAW_DIR, save_parquet, standardize_ohlcv


class VNStockAdapter:
    def __init__(self, source: str = "VCI"):
        self.source = source

    def _quote(self, symbol: str):
        from vnstock.api.quote import Quote
        return Quote(symbol=symbol, source=self.source)

    def fetch_equity(self, symbol: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
        q = self._quote(symbol)
        errors = []
        for name in ["history", "ohlcv"]:
            try:
                fn = getattr(q, name)
                df = fn(start=start, end=end, interval="1D")
                return standardize_ohlcv(df)
            except TypeError:
                try:
                    df = fn(start=start, end=end)
                    return standardize_ohlcv(df)
                except Exception as exc:
                    errors.append(f"{name}: {type(exc).__name__}: {exc}")
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
        raise ConnectionError(f"Không thể lấy dữ liệu {symbol}. {' | '.join(errors)}")

    def fetch_index(self, symbol: str, start: str, end: Optional[str] = None) -> pd.DataFrame:
        return self.fetch_equity(symbol, start=start, end=end)


def _default_start(asset_type: str) -> str:
    today = date.today()
    if asset_type == "index":
        return (today - timedelta(days=365 * 8)).isoformat()
    return (today - timedelta(days=430)).isoformat()


def update_market_dataset(symbol: str, dataset_name: Optional[str] = None, asset_type: str = "index", start: Optional[str] = None, end: Optional[str] = None, source: str = "VCI") -> str:
    adapter = VNStockAdapter(source=source)
    start = start or _default_start(asset_type)
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
