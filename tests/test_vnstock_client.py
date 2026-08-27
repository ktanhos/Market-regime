"""Lớp gọi API: phân loại lỗi, backoff, rate limit, thiếu mã, không retry vô hạn."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src import config
from src.schema import DataQualityError, standardize_ohlcv
from src.vnstock_data import (
    EMPTY,
    RateLimiter,
    SourceHealth,
    fetch_equity,
    fetch_index,
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


def test_rate_limit_waits_and_retries_before_giving_up():
    """Nguồn cho 20 lượt/phút và bảo chờ 45 giây; chờ rồi thử lại là đúng việc."""
    seen = []

    def caller(symbol, source, start, end):
        seen.append(source)
        raise RuntimeError("429 Too Many Requests")

    waits: list[float] = []
    with pytest.raises(FetchError) as info:
        fetch_history("FPT", start="2024-01-01", caller=caller, sleep=waits.append)

    assert info.value.kind == RATE_LIMITED
    # Một lượt đầu cộng đúng MAX_RATE_LIMIT_RETRIES lần chờ rồi thử lại.
    assert len(seen) == config.MAX_RATE_LIMIT_RETRIES + 1
    assert waits.count(config.RATE_LIMIT_COOLDOWN_SECONDS) == config.MAX_RATE_LIMIT_RETRIES


def test_rate_limit_that_clears_lets_the_fetch_succeed():
    attempts = {"n": 0}

    def caller(symbol, source, start, end):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("Rate limit exceeded")
        return standardize_ohlcv(synthetic_ohlcv(30))

    result = fetch_history("FPT", start="2024-01-01", caller=caller, sleep=lambda s: None)
    assert result.rows == 30
    assert attempts["n"] == 2


# --- Điều tiết theo phút ------------------------------------------------------

def test_rate_limiter_paces_calls_within_the_window():
    clock = {"t": 0.0}
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        clock["t"] += seconds

    limiter = RateLimiter(max_calls=3, window=60.0, sleep=sleep, clock=lambda: clock["t"])
    for _ in range(3):
        assert limiter.acquire() == 0.0        # ba lượt đầu đi ngay
    assert limiter.acquire() > 0               # lượt thứ tư phải chờ
    assert slept and slept[0] == pytest.approx(60.05, abs=0.1)


def test_rate_limiter_releases_slots_as_the_window_slides():
    clock = {"t": 0.0}
    limiter = RateLimiter(max_calls=2, window=60.0, sleep=lambda s: None, clock=lambda: clock["t"])
    limiter.acquire()
    limiter.acquire()
    clock["t"] = 61.0                          # cửa sổ đã trôi qua
    assert limiter.acquire() == 0.0


def test_rate_limiter_budget_follows_the_api_key():
    """Gói Khách 20 lượt/phút, có API key thì cao hơn; ta đặt dưới hạn mức thật."""
    assert config.requests_per_minute(False) < 20
    assert config.requests_per_minute(True) > config.requests_per_minute(False)
    assert config.request_spacing_seconds(False) > config.request_spacing_seconds(True)


def test_a_full_bootstrap_fits_inside_the_guest_budget():
    """33 lượt gọi cho một lượt khởi tạo phải nằm trong hạn mức gói Khách."""
    calls = 1 + 2 + 30                          # danh sách + hai chỉ số + 30 cổ phiếu
    spacing = config.request_spacing_seconds(False)
    per_minute = 60.0 / spacing
    assert per_minute <= 20                     # không vượt hạn mức quan sát được
    assert calls * spacing < 15 * 60            # vẫn hoàn tất trong thời gian hợp lý


def test_fetch_history_goes_through_the_limiter():
    clock = {"t": 0.0}
    limiter = RateLimiter(max_calls=100, window=60.0, sleep=lambda s: None, clock=lambda: clock["t"])

    def caller(symbol, source, start, end):
        return standardize_ohlcv(synthetic_ohlcv(30))

    fetch_history("FPT", start="2024-01-01", caller=caller, sleep=lambda s: None, limiter=limiter)
    assert len(limiter._calls) == 1


def test_empty_response_is_an_error_not_silent_success():
    def caller(symbol, source, start, end):
        raise DataQualityError("Nguồn trả về bảng rỗng")

    with pytest.raises(FetchError):
        fetch_history("FPT", start="2024-01-01", caller=caller, sleep=lambda s: None)


def test_default_start_uses_long_history_for_index_and_short_for_stock():
    """Chỉ số cần nền dài, cổ phiếu chỉ cần đủ MA200 của chính nó."""
    index_start = pd.Timestamp(default_start("index"))
    stock_start = pd.Timestamp(default_start("stock"))
    today = pd.Timestamp(date.today())

    assert (today - index_start).days >= 365 * config.INDEX_HISTORY_YEARS - 2
    assert (today - stock_start).days == config.STOCK_HISTORY_CALENDAR_DAYS
    assert stock_start > index_start


def test_incremental_start_overlaps_the_last_stored_session():
    last = pd.Timestamp("2026-08-20")
    start = pd.Timestamp(default_start("index", last))
    assert (last - start).days == config.INCREMENTAL_OVERLAP_DAYS
    assert 10 <= config.INCREMENTAL_OVERLAP_DAYS <= 15


def test_connectivity_check_reports_failures_without_raising(monkeypatch):
    import src.vnstock_data as module

    def failing(symbol, start, end=None, asset_type="stock", **kwargs):
        raise FetchError("không kết nối được", TRANSIENT, symbol=symbol, source="VCI")

    def failing_members(*args, **kwargs):
        raise FetchError("không lấy được danh sách", TRANSIENT, symbol="VN30", source="VCI")

    monkeypatch.setattr(module, "fetch_history", failing)
    monkeypatch.setattr(module, "fetch_index_members", failing_members)
    probes = module.connectivity_check(sleep=lambda s: None)
    assert [p.name for p in probes] == ["VNINDEX", "FPT", "VN30 Universe"]
    assert all(p.ok is False and p.error for p in probes)
    assert all(p.as_dict()["Kết quả"] == "FAILED" for p in probes)


def test_connectivity_check_reports_schema_and_range_on_success(monkeypatch):
    import src.vnstock_data as module

    def ok(symbol, start, end=None, asset_type="stock", **kwargs):
        return module.FetchResult(
            symbol=symbol,
            frame=standardize_ohlcv(synthetic_ohlcv(30)),
            source="VCI",
            attempts=1,
        )

    monkeypatch.setattr(module, "fetch_history", ok)
    monkeypatch.setattr(module, "fetch_index_members", lambda *a, **k: [f"S{i:02d}" for i in range(30)])
    probes = module.connectivity_check(sleep=lambda s: None)

    assert all(p.ok for p in probes)
    price = probes[0].as_dict()
    assert price["Kết quả"] == "SUCCESS"
    assert price["Số dòng"] == 30
    assert price["Ngày đầu"] and price["Ngày cuối"]
    assert price["Schema"] == "date, open, high, low, close, volume"
    assert probes[2].rows == 30


def test_named_helpers_pick_the_right_asset_type():
    seen = []

    def caller(symbol, source, start, end):
        seen.append((symbol, source))
        return standardize_ohlcv(synthetic_ohlcv(30))

    module_index = fetch_index("VN30", start="2024-01-01", caller=caller, sleep=lambda s: None)
    assert module_index.source == "VCI"
    fetch_equity("FPT", start="2024-01-01", caller=caller, sleep=lambda s: None)
    assert [s for _, s in seen] == ["VCI", "VCI"]


def test_fetch_index_members_rejects_an_implausibly_short_list(monkeypatch):
    import src.vnstock_data as module

    class FakeListing:
        def __init__(self, source):
            self.source = source

        def symbols_by_group(self, group):
            return pd.Series(["ACB", "FPT"])

    monkeypatch.setattr(module, "_listing_class", lambda: FakeListing)
    with pytest.raises(FetchError) as info:
        module.fetch_index_members("VN30")
    assert info.value.kind == module.EMPTY


# --- Source health: tránh lặp lại thử một nguồn đang lỗi cho từng mã --------

def test_vci_failing_then_kbs_succeeding_marks_vci_degraded_for_the_session():
    """Kịch bản 1: VCI lỗi, KBS thành công -> VCI bị đánh dấu tạm không khả dụng."""
    health = SourceHealth()

    def caller(symbol, source, start, end):
        if source == "VCI":
            raise ConnectionError("connection reset by peer")
        return standardize_ohlcv(synthetic_ohlcv(30))

    result = fetch_history(
        "FPT", start="2024-01-01", caller=caller, sleep=lambda s: None, source_health=health,
    )
    assert result.source == "KBS"
    assert health.is_degraded("VCI")
    assert not health.is_degraded("KBS")

    # Mã tiếp theo trong cùng phiên: không còn thử VCI nữa, đi thẳng KBS.
    seen = []

    def caller2(symbol, source, start, end):
        seen.append(source)
        return standardize_ohlcv(synthetic_ohlcv(30))

    result2 = fetch_history(
        "VNM", start="2024-01-01", caller=caller2, sleep=lambda s: None, source_health=health,
    )
    assert seen == ["KBS"]           # VCI không được thử lại
    assert result2.source == "KBS"


def test_vci_succeeding_keeps_the_existing_fallback_order():
    """Kịch bản 2: VCI thành công -> giữ nguyên logic ưu tiên hiện tại."""
    health = SourceHealth()
    seen = []

    def caller(symbol, source, start, end):
        seen.append(source)
        return standardize_ohlcv(synthetic_ohlcv(30))

    result = fetch_history(
        "FPT", start="2024-01-01", caller=caller, sleep=lambda s: None, source_health=health,
    )
    assert result.source == "VCI"
    assert seen == ["VCI"]
    assert not health.is_degraded("VCI")

    # Mã tiếp theo vẫn thử VCI trước như bình thường.
    seen2 = []

    def caller2(symbol, source, start, end):
        seen2.append(source)
        return standardize_ohlcv(synthetic_ohlcv(30))

    fetch_history("VNM", start="2024-01-01", caller=caller2, sleep=lambda s: None, source_health=health)
    assert seen2 == ["VCI"]


def test_a_single_failed_attempt_that_recovers_on_retry_does_not_degrade_the_source():
    """Chưa đủ bằng chứng (lỗi một lần rồi tự hồi phục ở lần thử lại) -> không đánh dấu."""
    health = SourceHealth()
    attempts = {"n": 0}

    def caller(symbol, source, start, end):
        attempts["n"] += 1
        if source == "VCI" and attempts["n"] == 1:
            raise ConnectionError("connection reset by peer")
        return standardize_ohlcv(synthetic_ohlcv(30))

    result = fetch_history(
        "FPT", start="2024-01-01", caller=caller, sleep=lambda s: None, source_health=health,
    )
    assert result.source == "VCI"    # tự hồi phục ở lần thử thứ hai của chính VCI
    assert not health.is_degraded("VCI")


def test_source_health_never_excludes_the_only_available_source():
    """VN30 chỉ có VCI: dù bị đánh dấu lỗi vẫn phải thử, không còn lựa chọn khác."""
    health = SourceHealth()
    health.record_source_exhausted("VCI", TRANSIENT)
    assert health.is_degraded("VCI")
    assert health.order_sources(("VCI",)) == ("VCI",)


def test_source_health_state_does_not_persist_across_sessions():
    """Kịch bản 5: một phiên lỗi không vô hiệu hóa vĩnh viễn; phiên sau kiểm tra lại."""
    session1 = SourceHealth()

    def failing(symbol, source, start, end):
        if source == "VCI":
            raise ConnectionError("connection reset by peer")
        return standardize_ohlcv(synthetic_ohlcv(30))

    fetch_history("FPT", start="2024-01-01", caller=failing, sleep=lambda s: None, source_health=session1)
    assert session1.is_degraded("VCI")

    # Phiên cập nhật mới = đối tượng SourceHealth mới, không kế thừa trạng thái cũ.
    session2 = SourceHealth()
    seen = []

    def ok(symbol, source, start, end):
        seen.append(source)
        return standardize_ohlcv(synthetic_ohlcv(30))

    fetch_history("FPT", start="2024-01-01", caller=ok, sleep=lambda s: None, source_health=session2)
    assert seen == ["VCI"]           # phiên mới vẫn thử VCI trước như bình thường
    assert not session2.is_degraded("VCI")


def test_fetch_index_members_normalises_and_sorts(monkeypatch):
    import src.vnstock_data as module

    class FakeListing:
        def __init__(self, source):
            self.source = source

        def symbols_by_group(self, group):
            return pd.Series([" fpt ", "ACB", "acb", "VNM"] + [f"S{i:02d}" for i in range(25)])

    monkeypatch.setattr(module, "_listing_class", lambda: FakeListing)
    symbols = module.fetch_index_members("VN30")
    assert symbols == sorted(symbols)
    assert symbols.count("ACB") == 1
    assert "FPT" in symbols
