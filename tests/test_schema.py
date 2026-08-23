"""Chuẩn hóa, kiểm tra và hợp nhất dữ liệu."""

from __future__ import annotations

import pandas as pd
import pytest

from src.schema import (
    CANONICAL_COLUMNS,
    DataQualityError,
    inspect_frame,
    merge_history,
    standardize_ohlcv,
    to_close_panel,
    validate_frame,
)
from tests.conftest import synthetic_ohlcv


def test_standardize_maps_time_to_date():
    """Lỗi kiến trúc cũ: lớp ghi dùng 'date', lớp đọc dùng 'time'."""
    out = standardize_ohlcv(synthetic_ohlcv(30))
    assert list(out.columns) == list(CANONICAL_COLUMNS)
    assert str(out["date"].dtype).startswith("datetime64")


def test_standardize_accepts_date_column_too():
    frame = synthetic_ohlcv(30).rename(columns={"time": "date"})
    assert "date" in standardize_ohlcv(frame).columns


def test_standardize_rejects_empty_and_missing_date():
    with pytest.raises(DataQualityError):
        standardize_ohlcv(pd.DataFrame())
    with pytest.raises(DataQualityError):
        standardize_ohlcv(pd.DataFrame({"close": [1.0, 2.0]}))
    with pytest.raises(DataQualityError):
        standardize_ohlcv(None)


def test_duplicate_dates_are_collapsed_keeping_latest():
    frame = synthetic_ohlcv(10)
    duplicated = pd.concat([frame, frame.tail(3).assign(close=999.0)], ignore_index=True)
    out = standardize_ohlcv(duplicated)
    assert out["date"].duplicated().sum() == 0
    assert out["close"].iloc[-1] == 999.0


def test_merge_history_dedupes_and_prefers_new():
    old = standardize_ohlcv(synthetic_ohlcv(100, seed=1))
    new = old.tail(20).copy()
    new["close"] = new["close"] * 1.5
    merged = merge_history(old, new)
    assert len(merged) == len(old)
    assert merged["date"].duplicated().sum() == 0
    assert merged["close"].iloc[-1] == pytest.approx(new["close"].iloc[-1])


def test_merge_history_with_no_previous_data():
    new = synthetic_ohlcv(40)
    assert len(merge_history(None, new)) == 40


def test_validate_rejects_broken_prices():
    frame = standardize_ohlcv(synthetic_ohlcv(30))
    broken = frame.copy()
    broken.loc[broken.index[0], "high"] = broken.loc[broken.index[0], "low"] - 1
    with pytest.raises(DataQualityError):
        validate_frame(broken)

    negative = frame.copy()
    negative.loc[negative.index[0], "close"] = -5.0
    with pytest.raises(DataQualityError):
        validate_frame(negative)


def test_inspect_reports_range():
    report = inspect_frame(standardize_ohlcv(synthetic_ohlcv(50)))
    assert report.rows == 50
    assert report.duplicate_dates == 0
    assert report.first_date < report.last_date


def test_close_panel_does_not_invent_values():
    frames = {
        "AAA": standardize_ohlcv(synthetic_ohlcv(50, seed=2)),
        "BBB": standardize_ohlcv(synthetic_ohlcv(20, seed=3)),
    }
    panel = to_close_panel(frames)
    assert panel.shape[1] == 2
    # BBB thiếu lịch sử phải hiện là NaN chứ không được kéo dài giá.
    assert panel["BBB"].isna().sum() > 0
