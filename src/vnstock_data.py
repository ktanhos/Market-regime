from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .data import standardize_ohlcv, save_parquet


class VNStockAdapter:
    """Single access point for vnstock market data.

    All vnstock-specific calls are isolated here so the rest of the project
    remains unchanged if the library API changes.
    """

    def __init__(self, source: str = "VCI"):
        self.source = source

    def _market(self):
        try:
            from vnstock import Market
            return Market(source=self.source)
        except (ImportError, TypeError):
            # Compatibility fallback for older vnstock installations.
            return None

    def fetch_index(
        self,
        symbol: str,
        start: str = "2010-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        market = self._market()
        if market is not None:
            try:
                df = market.index(symbol).ohlcv(start=start, end=end)
                return standardize_ohlcv(df)
            except Exception:
                pass

        # Backward-compatible fallback.
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol, source=self.source)
        df = stock.quote.history(start=start, end=end, interval="1D")
        return standardize_ohlcv(df)

    def fetch_equity(
        self,
        symbol: str,
        start: str = "2010-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        market = self._market()
        if market is not None:
            try:
                df = market.equity(symbol).ohlcv(start=start, end=end)
                return standardize_ohlcv(df)
            except Exception:
                pass

        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol, source=self.source)
        df = stock.quote.history(start=start, end=end, interval="1D")
        return standardize_ohlcv(df)


def update_market_dataset(
    symbol: str,
    dataset_name: Optional[str] = None,
    asset_type: str = "index",
    start: str = "2010-01-01",
    end: Optional[str] = None,
    source: str = "VCI",
) -> str:
    adapter = VNStockAdapter(source=source)
    if asset_type == "index":
        df = adapter.fetch_index(symbol, start=start, end=end)
    elif asset_type == "equity":
        df = adapter.fetch_equity(symbol, start=start, end=end)
    else:
        raise ValueError("asset_type must be index or equity")

    name = dataset_name or symbol.lower()
    path = save_parquet(df, name)
    return str(path)


def incremental_update(
    symbol: str,
    dataset_name: str,
    asset_type: str = "index",
    source: str = "VCI",
    overlap_days: int = 15,
) -> str:
    """Update an existing dataset with an overlapping window."""
    data_path = Path("data/raw") / f"{dataset_name}.parquet"

    if not data_path.exists():
        return update_market_dataset(
            symbol=symbol,
            dataset_name=dataset_name,
            asset_type=asset_type,
            source=source,
        )

    old = pd.read_parquet(data_path)
    old = standardize_ohlcv(old)
    last_date = pd.to_datetime(old["time"]).max()
    start = (last_date - timedelta(days=overlap_days)).strftime("%Y-%m-%d")

    adapter = VNStockAdapter(source=source)
    if asset_type == "index":
        new = adapter.fetch_index(symbol, start=start)
    elif asset_type == "equity":
        new = adapter.fetch_equity(symbol, start=start)
    else:
        raise ValueError("asset_type must be index or equity")

    combined = pd.concat([old, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["time"], keep="last")
    combined = combined.sort_values("time").reset_index(drop=True)
    save_parquet(combined, dataset_name)
    return str(data_path)
