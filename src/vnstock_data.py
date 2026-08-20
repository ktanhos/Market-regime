from typing import Optional

import pandas as pd

from .data import standardize_ohlcv, save_parquet


def fetch_ohlcv(
    symbol: str,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    source: str = "VCI",
) -> pd.DataFrame:
    """Fetch daily OHLCV using the current vnstock equity API.

    This adapter intentionally isolates vnstock-specific code from the rest
    of the project so API changes do not affect indicator logic.
    """
    try:
        from vnstock import Vnstock
    except ImportError as exc:
        raise ImportError("vnstock is not installed") from exc

    stock = Vnstock().stock(symbol=symbol, source=source)
    df = stock.quote.history(start=start, end=end, interval="1D")
    return standardize_ohlcv(df)


def update_market_dataset(
    symbol: str,
    name: Optional[str] = None,
    start: str = "2010-01-01",
    end: Optional[str] = None,
    source: str = "VCI",
) -> str:
    """Fetch and store one market dataset as Parquet."""
    df = fetch_ohlcv(symbol=symbol, start=start, end=end, source=source)
    dataset_name = name or symbol.lower()
    path = save_parquet(df, dataset_name)
    return str(path)
