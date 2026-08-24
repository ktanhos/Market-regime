"""Pipeline cập nhật đầu-cuối, không chạm mạng."""

from __future__ import annotations

import pandas as pd
import pytest

from src import config, quality, storage
from src import universe as universe_module
from src.schema import standardize_ohlcv
from src.features import build_snapshot
from src.updater import rebuild_features, run_update
from src.vnstock_data import RATE_LIMITED, TRANSIENT, FetchError, FetchResult
from tests.conftest import synthetic_ohlcv

UNIVERSE = [f"S{i:02d}" for i in range(30)]


def make_fetcher(fail: dict | None = None, periods: int = 320):
    """Bộ giả lập API. ``fail`` ánh xạ mã -> FetchError sẽ ném ra."""
    fail = fail or {}
    calls: list[tuple[str, str, str]] = []

    def fetcher(symbol, start, end=None, asset_type="stock", sleep=None, **kwargs):
        calls.append((symbol, start, end))
        if symbol in fail:
            raise fail[symbol]
        seed = abs(hash(symbol)) % 500
        frame = standardize_ohlcv(synthetic_ohlcv(periods, seed=seed))
        return FetchResult(symbol=symbol, frame=frame, source="VCI", attempts=1)

    fetcher.calls = calls
    return fetcher


def universe_fetcher():
    return list(UNIVERSE)


# --- Kịch bản thành công -----------------------------------------------------

def test_full_update_writes_every_dataset(temp_store):
    fetcher = make_fetcher()
    report = run_update(fetcher=fetcher, universe_fetcher=universe_fetcher, sleep=lambda s: None)

    assert report.total_count == 32          # VNINDEX + VN30 + 30 cổ phiếu
    assert report.success_count == 32
    assert report.failures == []
    assert storage.index_path(config.VNINDEX_DATASET).exists()
    assert storage.index_path(config.VN30_INDEX_DATASET).exists()
    assert len(storage.available_stock_symbols()) == 30
    assert config.UPDATE_LOG_FILE.exists()
    assert config.VN30_SNAPSHOT_FILE.exists()


def test_api_is_called_sequentially_one_call_per_symbol(temp_store):
    fetcher = make_fetcher()
    run_update(fetcher=fetcher, universe_fetcher=universe_fetcher, sleep=lambda s: None)
    symbols = [c[0] for c in fetcher.calls]
    assert len(symbols) == 32
    assert len(symbols) == len(set(symbols))   # không gọi lặp một mã


def test_second_update_is_incremental_not_a_full_refetch(temp_store):
    run_update(fetcher=make_fetcher(), universe_fetcher=universe_fetcher, sleep=lambda s: None)
    second = make_fetcher()
    run_update(fetcher=second, universe_fetcher=universe_fetcher, sleep=lambda s: None)

    starts = {symbol: pd.Timestamp(start) for symbol, start, _ in second.calls}
    # Lần hai phải nối tiếp từ ngày cuối cùng đã lưu, không tải lại toàn bộ lịch sử.
    for name, path in (
        ("VNINDEX", storage.index_path(config.VNINDEX_DATASET)),
        ("S00", storage.stock_path("S00")),
    ):
        last = storage.last_stored_date(path)
        assert (last - starts[name]).days == config.INCREMENTAL_OVERLAP_DAYS
    assert starts["VNINDEX"] > pd.Timestamp(config.index_history_start())


def test_merge_does_not_duplicate_dates_on_repeated_updates(temp_store):
    run_update(fetcher=make_fetcher(), universe_fetcher=universe_fetcher, sleep=lambda s: None)
    before = storage.load_index(config.VNINDEX_DATASET)
    run_update(fetcher=make_fetcher(), universe_fetcher=universe_fetcher, sleep=lambda s: None)
    after = storage.load_index(config.VNINDEX_DATASET)

    assert after["date"].duplicated().sum() == 0
    assert len(after) >= len(before)


def test_update_log_records_rows_before_and_after_merge(temp_store):
    run_update(fetcher=make_fetcher(), universe_fetcher=universe_fetcher, sleep=lambda s: None)
    log = storage.read_json(config.UPDATE_LOG_FILE)
    entry = next(d for d in log["datasets"] if d["name"] == "VNINDEX")
    for key in ("rows_before_merge", "rows_after_merge", "rows_added", "source",
                "requested_start", "requested_end", "first_date", "last_date"):
        assert key in entry


# --- Kịch bản lỗi ------------------------------------------------------------

def test_one_missing_symbol_does_not_stop_the_rest(temp_store):
    fetcher = make_fetcher(fail={"S07": FetchError("nguồn không có mã này", TRANSIENT, symbol="S07", source="VCI")})
    report = run_update(fetcher=fetcher, universe_fetcher=universe_fetcher, sleep=lambda s: None)

    assert report.success_count == 31
    assert len(report.failures) == 1
    assert report.failures[0]["symbol"] == "S07"
    assert report.failures[0]["message"]           # luôn có nguyên nhân
    assert "S07" not in storage.available_stock_symbols()


def test_rate_limit_stops_the_run_and_explains_why(temp_store):
    fetcher = make_fetcher(
        fail={"S03": FetchError("API đang giới hạn số lượt truy cập.", RATE_LIMITED, symbol="S03", source="VCI")}
    )
    report = run_update(fetcher=fetcher, universe_fetcher=universe_fetcher, sleep=lambda s: None)

    assert report.rate_limited is True
    assert "giới hạn" in report.aborted_reason.lower()
    assert report.success_count < report.total_count
    # Đã dừng sớm thay vì tiếp tục gọi hết 30 mã.
    assert len(fetcher.calls) < 32


def test_universe_failure_falls_back_to_saved_list(temp_store):
    universe_module.save_universe(UNIVERSE, source="ảnh chụp kiểm thử")

    def failing_universe():
        raise FetchError("không lấy được danh sách", TRANSIENT, symbol="VN30", source="VCI")

    report = run_update(fetcher=make_fetcher(), universe_fetcher=failing_universe, sleep=lambda s: None)
    assert report.universe["status"] == "dùng danh sách đã lưu"
    assert report.success_count == 32


def test_total_api_failure_reports_zero_with_reasons_not_a_crash(temp_store):
    failures = {s: FetchError("connection reset", TRANSIENT, symbol=s, source="VCI") for s in UNIVERSE}
    failures.update(
        {name: FetchError("connection reset", TRANSIENT, symbol=name, source="VCI") for name in ("VNINDEX", "VN30")}
    )
    report = run_update(fetcher=make_fetcher(fail=failures), universe_fetcher=universe_fetcher, sleep=lambda s: None)

    assert report.success_count == 0
    fetch_failures = [f for f in report.failures if f["kind"] != "compute"]
    assert len(fetch_failures) == 32
    assert all(f["message"] for f in report.failures)     # 0/32 luôn kèm nguyên nhân
    # Bước tính chỉ tiêu cũng phải báo lỗi rõ ràng thay vì im lặng.
    assert any(f["kind"] == "compute" for f in report.failures)
    table = quality.failure_table(report.as_dict())
    assert set(table.columns) == {"Mã", "Nguồn", "Loại lỗi", "Nguyên nhân"}


# --- Đọc dữ liệu cho dashboard ------------------------------------------------

def test_snapshot_has_no_data_path(temp_store):
    state = build_snapshot(UNIVERSE)
    assert state["ready"] is False
    assert "VNINDEX" in state["reason"]


def test_snapshot_after_a_successful_update(temp_store):
    run_update(fetcher=make_fetcher(), universe_fetcher=universe_fetcher, sleep=lambda s: None)
    from src import portfolio_risk, regime as regime_module

    state = build_snapshot()
    assert state["ready"] is True
    assert state["breadth"]["sufficient"] is True
    assert state["breadth"]["universe_size"] == 30
    assert not state["missing_symbols"]

    # Feature Layer -> Market Regime Layer -> Portfolio Risk Layer
    regime = regime_module.build_regime(
        state["trend"], state["stress"], state["breadth"],
        state["dispersion"], state["concentration"],
    )
    assert regime["regime"] in {
        "THUẬN LỢI", "CẢNH BÁO", "CHUYỂN TIẾP", "CHỊU ÁP LỰC", "CĂNG THẲNG",
    }
    assert portfolio_risk.guidance(regime)["risk_budget"]


def test_dashboard_works_when_only_the_index_is_available(temp_store):
    """Thiếu toàn bộ cổ phiếu VN30 phải suy giảm êm, không được sập."""
    storage.write_frame(standardize_ohlcv(synthetic_ohlcv(400)), storage.index_path(config.VNINDEX_DATASET))
    state = build_snapshot(UNIVERSE)

    assert state["ready"] is True
    assert state["breadth"]["state"] == "CHƯA CÓ DỮ LIỆU"
    assert state["breadth"]["data_state"] == "no_data"
    assert state["breadth"]["sufficient"] is False
    assert len(state["missing_symbols"]) == 30


def test_corrupted_parquet_is_ignored_rather_than_crashing(temp_store):
    path = storage.stock_path("S01")
    path.write_bytes(b"not a parquet file")
    assert storage.load_stock("S01") is None


def test_rebuild_features_needs_no_network(temp_store):
    storage.write_frame(standardize_ohlcv(synthetic_ohlcv(400)), storage.index_path(config.VNINDEX_DATASET))
    for symbol in UNIVERSE[:25]:
        storage.write_frame(
            standardize_ohlcv(synthetic_ohlcv(300, seed=abs(hash(symbol)) % 400)),
            storage.stock_path(symbol),
        )
    snapshot = rebuild_features(UNIVERSE)
    assert snapshot["universe_size"] == 30
    assert snapshot["max_valid_symbols"] == 25
    assert len(snapshot["missing_symbols"]) == 5
    assert config.VNINDEX_FEATURES_FILE.exists()


def test_universe_snapshot_records_when_it_was_taken(temp_store):
    saved = universe_module.save_universe(["AAA", "BBB"], source="kiểm thử", as_of="2026-08-21")
    loaded = universe_module.load_universe()
    assert loaded["symbols"] == ["AAA", "BBB"]
    assert loaded["as_of"] == "2026-08-21"
    assert "quá khứ" in saved["note"]


def test_written_json_is_strictly_valid(temp_store):
    """json.dumps mặc định ghi NaN, không phải JSON hợp lệ."""
    import json

    from src import storage as storage_module

    path = temp_store / "x.json"
    storage_module.write_json(path, {"a": float("nan"), "b": float("inf"), "c": 1.5, "d": {"e": float("nan")}})
    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda c: pytest.fail(f"NaN/Inf: {c}"))
    assert payload == {"a": None, "b": None, "c": 1.5, "d": {"e": None}}


def test_update_log_and_snapshot_are_valid_json(temp_store):
    import json

    run_update(fetcher=make_fetcher(), universe_fetcher=universe_fetcher, sleep=lambda s: None)
    for path in (config.UPDATE_LOG_FILE, config.VN30_SNAPSHOT_FILE):
        json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda c: pytest.fail(f"NaN/Inf: {c}"))
