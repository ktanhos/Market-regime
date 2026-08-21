from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import time
import pandas as pd

from src.vnstock_data import incremental_update

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def _last_date(path: Path):
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    for col in ["time", "date", "datetime"]:
        if col in df.columns:
            return pd.to_datetime(df[col]).max().date()
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.max().date()
    return None


def _start_for(path: Path, fallback_years: int = 12) -> str:
    last = _last_date(path)
    if last is None:
        return (date.today() - timedelta(days=365 * fallback_years)).isoformat()
    return (last - timedelta(days=10)).isoformat()


def refresh_market(symbols: list[str], sleep_seconds: float = 1.1):
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    jobs = [("VNINDEX", "vnindex", "index")] + [(s, s, "stock") for s in symbols]
    results = []
    for symbol, dataset, asset_type in jobs:
        path = RAW / f"{dataset.lower()}.parquet"
        start = _start_for(path)
        try:
            incremental_update(symbol=symbol, dataset_name=dataset, asset_type=asset_type, start=start, end=date.today().isoformat())
            results.append((symbol, True, start, ""))
        except Exception as exc:
            results.append((symbol, False, start, str(exc)))
        time.sleep(sleep_seconds)
    return pd.DataFrame(results, columns=["symbol", "success", "start", "error"])
