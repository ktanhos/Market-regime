from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path("data")


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common VNStock OHLCV column names and sort by date."""
    out = df.copy()
    rename = {}
    for column in out.columns:
        key = str(column).strip().lower()
        mapping = {
            "time": "date",
            "datetime": "date",
            "tradingdate": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        if key in mapping:
            rename[column] = mapping[key]
    out = out.rename(columns=rename)
    if "date" not in out.columns:
        raise ValueError("Data must contain a date column")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    return out.reset_index(drop=True)


def save_parquet(df: pd.DataFrame, name: str, data_dir: Optional[Path] = None) -> Path:
    """Persist normalized data locally as Parquet."""
    target_dir = data_dir or DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_parquet(name: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load a local Parquet dataset."""
    target_dir = data_dir or DATA_DIR
    path = target_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_parquet(path)
