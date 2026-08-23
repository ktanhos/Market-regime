"""Lớp gọi API: phân loại lỗi, backoff, rate limit, thiếu mã, không retry vô hạn."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.schema import DataQualityError, standardize_ohlcv
from src.vnstock_data import (
    EMPTY,
    PERMANENT,
    RATE_LIMITED,
    TRANSIENT,
    FetchError,
    classify_error,
    default_start,
    fetch_history,
    friendly_message,
    sources_for,
)
from tests.conftest import synthetic_ohlcv


def test_source_order_reflects_real_provider_support():
    """KBS không hỗ trợ chỉ số VN30 nên VN30 chỉ được gọi qua VCI."""
    assert sources_for("VN30", "index") == ("VCI",)
    assert sources_for("VNINDEX", "index")[0] == "VCI"
    assert "KBS" in sources_for("FPT", "stock")


def test_error_classification():
    assert classify_error(Exception("HTTP 429 Too Many Requests")) == RATE_LIMITED
    assert classify_error(Exception("Rate limit exceeded")) == RATE_LIMITED
    assert classify_error(TimeoutError("connection timed out")) == TRANSIENT
    assert classify_error(ValueError("Mã chỉ số 'VN30' không được hỗ trợ bởi KBS")) == PERMANENT
    assert classify_error(DataQualityError("Nguồn trả về bảng rỗng")) == EMPTY
    assert "giới hạn" in friendly_message(RATE_LIMITED).lower()


def test_fetch_succeeds_on_first_call():
    calls = []

    def caller(symbol, source, start, end):
        calls.append((symbol, source))
        return standardize_ohlcv(synthetic_ohlcv(60))

    result = fetch_history("FPT", start="2024-01-01", end="2024-06-01", caller=caller, sleep=lambda s: None)
    assert result.attempts == 1
    assert len(calls) == 1
    assert result.source == config.PRIMARY_SOURCE


def test_transient_error_retries_then_falls_back_to_second_source():
    seen = []

    def caller(symbol, source, start, end):
        seen.append(source)
        if source == "VCI":
            raise ConnectionError("connection reset by peer")
        return standardize_ohlcv(synthetic_ohlcv(30))

    delays = []
    result = fetch_history("FPT", start="2024-01-01", caller=caller, sleep=delays.append)
    assert result.source == "KBS"
    assert seen.count("VCI") == config.MAX_ATTEMPTS_PER_SOURCE
    assert delays and all(d <= config.BACKOFF_MAX_SECONDS for d in delays)


def test_permanent_error_does_not_retry_same_source():
    seen = []

    def caller(symbol, source, start, end):
        seen.append(source)
        raise ValueError("Mã chỉ số không được hỗ trợ")

    with pytest.raises(FetchError) as info:
        fetch_history("VNX", start="2024-01-01", caller=caller, sleep=lambda s: None)
    assert seen == ["VCI", "KBS"]  # mỗi nguồn đúng một lần, không lặp
    assert info.value.kind == TRANSIENT or info.value.kind == EMPTY


def test_rate_limit_aborts_immediately():
    seen = []

    def caller(symbol, source, start, end):
        seen.append(source)
        raise RuntimeError("429 Too Many Requests")

    with pytest.raises(FetchError) as info:
        fetch_history("FPT", start="2024-01-01", caller=caller, sleep=lambda s: None)
    assert info.value.kind == RATE_LIMITED
    assert len(seen) == 1  # dừng ngay, không nhân thêm lượt gọi


def test_empty_response_is_an_error_not_silent_success():
    def caller(symbol, source, start, end):
        raise DataQualityError("Nguồn trả về bảng rỗng")

    with pytest.raises(FetchError):
        fetch_history("FPT", start="2024-01-01", caller=caller, sleep=lambda s: None)


def test_default_start_uses_long_history_for_index_and_short_for_stock():
    assert default_start("index") == config.INDEX_HISTORY_START
    assert default_start("stock") > "2000-01-01"
    incremental = default_start("index", pd.Timestamp("2026-08-20"))
    assert incremental < "2026-08-20"  # có chồng lấn để bắt dữ liệu điều chỉnh muộn


def test_connectivity_check_reports_failures_without_raising(monkeypatch):
    import src.vnstock_data as module

    def failing(symbol, start, end=None, asset_type="stock", **kwargs):
        raise FetchError("không kết nối được", TRANSIENT, symbol=symbol, source="VCI")

    monkeypatch.setattr(module, "fetch_history", failing)
    rows = module.connectivity_check(sleep=lambda s: None)
    assert [r["symbol"] for r in rows] == ["VNINDEX", "FPT"]
    assert all(r["ok"] is False and r["message"] for r in rows)
