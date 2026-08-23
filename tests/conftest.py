"""Dữ liệu giả lập dùng riêng cho kiểm thử.

Đây là fixture để kiểm thử logic, KHÔNG phải dữ liệu thị trường và không bao giờ
được ghi vào data/. Mọi bài test đều trỏ kho dữ liệu sang thư mục tạm.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, storage  # noqa: E402


def synthetic_ohlcv(periods: int = 400, seed: int = 0, drift: float = 0.0003, vol: float = 0.012, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=periods)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, vol, periods)))
    span = 1 + np.abs(rng.normal(0, 0.006, periods))
    return pd.DataFrame(
        {
            "time": dates,
            "open": close / span,
            "high": close * span,
            "low": close / span,
            "close": close,
            "volume": rng.integers(1_000_000, 9_000_000, periods),
        }
    )


@pytest.fixture()
def temp_store(tmp_path, monkeypatch):
    """Chuyển toàn bộ đường dẫn kho dữ liệu sang thư mục tạm."""
    raw = tmp_path / "raw"
    stocks = raw / "stocks"
    processed = tmp_path / "processed"
    reference = tmp_path / "reference"
    for path in (raw, stocks, processed, reference):
        path.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "STOCK_DIR", stocks)
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "REFERENCE_DIR", reference)
    monkeypatch.setattr(config, "UNIVERSE_FILE", reference / "vn30_universe.json")
    monkeypatch.setattr(config, "UPDATE_LOG_FILE", processed / "update_log.json")
    monkeypatch.setattr(config, "VN30_SNAPSHOT_FILE", processed / "vn30_snapshot.json")
    monkeypatch.setattr(config, "VNINDEX_FEATURES_FILE", processed / "vnindex_features.parquet")
    monkeypatch.setattr(config, "REQUEST_DELAY_SECONDS", 0.0)
    storage.ensure_dirs()
    return tmp_path


@pytest.fixture()
def sleepless(monkeypatch):
    calls: list[float] = []

    def fake_sleep(seconds: float = 0.0) -> None:
        calls.append(float(seconds))

    return fake_sleep, calls
