from __future__ import annotations

from pathlib import Path
import time
import pandas as pd

from src.vnstock_data import incremental_update

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"


def refresh_market(symbols: list[str], sleep_seconds: float = 1.1, progress_callback=None):
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    jobs = [("VNINDEX", "vnindex", "index")] + [
        (symbol, symbol.lower(), "equity") for symbol in symbols
    ]

    results = []
    total = len(jobs)

    for i, (symbol, dataset, asset_type) in enumerate(jobs, start=1):
        if progress_callback:
            progress_callback(i, total, symbol)

        try:
            path = incremental_update(
                symbol=symbol,
                dataset_name=dataset,
                asset_type=asset_type,
            )
            results.append((symbol, True, path, ""))
        except Exception as exc:
            results.append((symbol, False, "", f"{type(exc).__name__}: {exc}"))

        if i < total:
            time.sleep(sleep_seconds)

    return pd.DataFrame(
        results,
        columns=["symbol", "success", "path", "error"],
    )
