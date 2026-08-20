from pathlib import Path
from typing import Optional

import pandas as pd

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    mapping = {
        "time": "date",
        "datetime": "date",
        "tradingdate": "date",
        "date": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    for column in out.columns:
        key = str(column).strip().lower()
        if key in mapping:
            rename[column] = mapping[key]
    out = out.rename(columns=rename)
    if "date" not in out.columns:
        raise ValueError("Data must contain a date column")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out = out.sort_values("date").drop_duplicates("date", keep="last")
    return out.reset_index(drop=True)


def save_parquet(df: pd.DataFrame, name: str, data_dir: Optional[Path] = None) -> Path:
    target_dir = data_dir or RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_parquet(name: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    target_dir = data_dir or RAW_DIR
    path = target_dir / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_parquet(path)
