"""Khởi tạo dữ liệu lần đầu, cập nhật tăng dần và các trạng thái ở giữa.

Đây là bộ test cho đúng lỗi đã gặp: ``data/raw/stocks`` trống nên Breadth,
Dispersion và Risk Concentration không chạy được.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config, quality, storage
from src import breadth as breadth_module
from src import universe as universe_module
from src.features import build_snapshot
from src.schema import standardize_ohlcv
from src.updater import (
    MODE_FIRST_RUN,
    MODE_INCREMENTAL,
    MODE_PARTIAL,
    SYNC_FAILED,
    SYNC_SUCCESS,
    plan_bootstrap,
    record_sync,
    run_update,
)
from src.vnstock_data import RATE_LIMITED, TRANSIENT, FetchError, FetchResult

UNIVERSE = [f"S{i:02d}" for i in range(30)]


def make_fetcher(fail: dict | None = None, sessions: int = 290):
    """Bộ giả lập API; ``fail`` ánh xạ mã -> lỗi sẽ ném ra."""
    fail = fail or {}
    calls: list[dict] = []

    def fetcher(symbol, start, end=None, asset_type="stock", sleep=None, **kwargs):
        calls.append({"symbol": symbol, "start": start, "asset_type": asset_type})
        if symbol in fail:
            raise fail[symbol]
        rng = np.random.default_rng(abs(hash(symbol)) % 997)
        index = pd.bdate_range("2025-06-19", periods=sessions)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, sessions)))
        frame = pd.DataFrame(
            {"time": index, "open": close, "high": close * 1.01,
             "low": close * 0.99, "close": close, "volume": 1}
        )
        return FetchResult(symbol=symbol, frame=standardize_ohlcv(frame), source="VCI", attempts=1)

    fetcher.calls = calls
    return fetcher


def universe_fetcher():
    return list(UNIVERSE)


def stock_files() -> list[str]:
    return sorted(p.name for p in config.STOCK_DIR.glob("*.parquet"))


# --- Phát hiện chế độ chạy ----------------------------------------------------

def test_empty_stock_directory_is_detected_as_a_first_run(temp_store):
    plan = plan_bootstrap(UNIVERSE)
    assert plan.first_run is True
    assert plan.mode == MODE_FIRST_RUN
    assert plan.full_history == UNIVERSE
    assert plan.incremental == []


def test_partial_data_asks_for_full_history_only_where_it_is_missing(temp_store):
    for symbol in UNIVERSE[:27]:
        storage.write_frame(
            standardize_ohlcv(_frame(300, symbol)), storage.stock_path(symbol)
        )
    plan = plan_bootstrap(UNIVERSE)
    assert plan.first_run is False
    assert plan.mode == MODE_PARTIAL
    assert len(plan.incremental) == 27
    assert plan.full_history == UNIVERSE[27:]


def test_short_history_files_are_topped_up_not_refetched_from_scratch(temp_store):
    for symbol in UNIVERSE:
        storage.write_frame(standardize_ohlcv(_frame(40, symbol)), storage.stock_path(symbol))
    plan = plan_bootstrap(UNIVERSE)
    assert plan.mode == MODE_PARTIAL
    assert plan.short_history == UNIVERSE
    assert plan.full_history == []          # đã có tệp thì nối tiếp, không tải lại từ đầu
    assert plan.incremental == UNIVERSE


def test_complete_data_is_an_incremental_run(temp_store):
    for symbol in UNIVERSE:
        storage.write_frame(standardize_ohlcv(_frame(300, symbol)), storage.stock_path(symbol))
    plan = plan_bootstrap(UNIVERSE)
    assert plan.mode == MODE_INCREMENTAL
    assert plan.full_history == []


def _frame(sessions: int, symbol: str) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(symbol)) % 997)
    index = pd.bdate_range("2025-06-19", periods=sessions)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, sessions)))
    return pd.DataFrame(
        {"time": index, "open": close, "high": close * 1.01,
         "low": close * 0.99, "close": close, "volume": 1}
    )


# --- Lần chạy đầu tiên --------------------------------------------------------

def test_first_run_creates_every_stock_file_under_raw_stocks(temp_store):
    report = run_update(fetcher=make_fetcher(), universe_fetcher=universe_fetcher, sleep=lambda s: None)

    assert report.first_run is True
    assert report.mode == MODE_FIRST_RUN
    assert report.index_success == report.index_total == 2
    assert report.stock_success == report.stock_total == 30
    assert stock_files() == [f"{s}.parquet" for s in UNIVERSE]
    # Giá cổ phiếu chỉ được nằm ở data/raw/stocks, không rơi ra data/raw.
    assert storage.legacy_stock_files() == []
    assert report.data_complete is True


def test_first_run_saves_the_universe_before_fetching_prices(temp_store):
    fetcher = make_fetcher()
    run_update(fetcher=fetcher, universe_fetcher=universe_fetcher, sleep=lambda s: None)

    saved = universe_module.load_universe()
    assert saved["symbols"] == UNIVERSE
    assert saved["as_of"]
    assert saved["is_fallback"] is False


def test_first_run_requests_full_history_for_every_symbol(temp_store):
    fetcher = make_fetcher()
    run_update(fetcher=fetcher, universe_fetcher=universe_fetcher, sleep=lambda s: None)

    stock_calls = [c for c in fetcher.calls if c["asset_type"] == "stock"]
    assert len(stock_calls) == 30
    expected = config.stock_history_start()
    assert {c["start"] for c in stock_calls} == {expected}


def test_first_run_makes_breadth_dispersion_and_concentration_work(temp_store):
    run_update(fetcher=make_fetcher(), universe_fetcher=universe_fetcher, sleep=lambda s: None)
    state = build_snapshot()

    assert state["breadth"]["data_state"] == breadth_module.DATA_OK
    assert state["breadth"]["sufficient"] is True
    assert state["breadth"]["valid_symbols"] == 30
    assert 0 <= state["breadth"]["score"] <= 100
    assert np.isfinite(state["dispersion"]["value"])
    assert np.isfinite(state["concentration"]["top_shares"][5])


def test_first_run_writes_every_expected_file(temp_store):
    report = run_update(fetcher=make_fetcher(), universe_fetcher=universe_fetcher, sleep=lambda s: None)
    audit = storage.verify_data_files(UNIVERSE)

    assert audit["missing"] == []
    assert report.files_written == report.files_expected == 36   # 2 chỉ số + 30 mã + 4 tệp phụ


# --- Cập nhật tăng dần --------------------------------------------------------

def test_second_run_is_incremental_for_every_symbol(temp_store):
    run_update(fetcher=make_fetcher(), universe_fetcher=universe_fetcher, sleep=lambda s: None)

    second = make_fetcher()
    report = run_update(fetcher=second, universe_fetcher=universe_fetcher, sleep=lambda s: None)

    assert report.first_run is False
    assert report.mode == MODE_INCREMENTAL
    assert {d["mode"] for d in report.datasets} == {"incremental"}
    stock_calls = [c for c in second.calls if c["asset_type"] == "stock"]
    assert all(c["start"] > config.stock_history_start() for c in stock_calls)


def test_a_mixed_run_only_backfills_the_symbols_that_need_it(temp_store):
    run_update(fetcher=make_fetcher(), universe_fetcher=universe_fetcher, sleep=lambda s: None)
    for symbol in UNIVERSE[:3]:
        storage.stock_path(symbol).unlink()

    fetcher = make_fetcher()
    report = run_update(fetcher=fetcher, universe_fetcher=universe_fetcher, sleep=lambda s: None)

    assert report.mode == MODE_PARTIAL
    modes = {d["name"]: d["mode"] for d in report.datasets if d["asset_type"] == "stock"}
    assert [s for s, m in modes.items() if m == "full_history"] == UNIVERSE[:3]
    assert len([m for m in modes.values() if m == "incremental"]) == 27
    assert report.stock_success == 30


# --- Hỏng giữa chừng ----------------------------------------------------------

def test_one_failing_symbol_does_not_stop_the_others(temp_store):
    failure = FetchError("connection reset", TRANSIENT, symbol="S07", source="VCI")
    report = run_update(
        fetcher=make_fetcher({"S07": failure}), universe_fetcher=universe_fetcher, sleep=lambda s: None
    )

    assert report.stock_success == 29
    assert report.stock_failed == ["S07"]
    assert report.stock_missing == ["S07"]
    assert "S07.parquet" not in stock_files()
    assert len(stock_files()) == 29

    # Breadth vẫn chạy trên số mã thực tế có dữ liệu.
    state = build_snapshot()
    assert state["breadth"]["valid_symbols"] == 29
    assert state["breadth"]["universe_size"] == 30
    assert state["breadth"]["sufficient"] is True


def test_rate_limit_keeps_what_was_already_fetched(temp_store):
    failure = FetchError("429 Too Many Requests", RATE_LIMITED, symbol="S05", source="VCI")
    report = run_update(
        fetcher=make_fetcher({"S05": failure}), universe_fetcher=universe_fetcher, sleep=lambda s: None
    )

    assert report.rate_limited is True
    assert report.stock_success == 5           # S00..S04 đã lấy xong
    assert len(stock_files()) == 5             # không rollback
    assert "tiếp tục" in report.aborted_reason.lower()


def test_a_run_after_a_rate_limit_resumes_instead_of_restarting(temp_store):
    failure = FetchError("429 Too Many Requests", RATE_LIMITED, symbol="S05", source="VCI")
    run_update(fetcher=make_fetcher({"S05": failure}), universe_fetcher=universe_fetcher, sleep=lambda s: None)

    plan = plan_bootstrap(UNIVERSE)
    assert plan.mode == MODE_PARTIAL
    assert len(plan.incremental) == 5          # giữ lại phần đã lấy
    assert len(plan.full_history) == 25        # chỉ lấy đủ lịch sử phần còn thiếu

    fetcher = make_fetcher()
    report = run_update(fetcher=fetcher, universe_fetcher=universe_fetcher, sleep=lambda s: None)
    assert report.stock_success == 30
    assert report.stock_missing == []
    assert len(stock_files()) == 30


def test_a_run_that_fetches_nothing_never_reports_completion(temp_store):
    failures = {
        symbol: FetchError("connection reset", TRANSIENT, symbol=symbol, source="VCI")
        for symbol in UNIVERSE + ["VNINDEX", "VN30"]
    }
    report = run_update(
        fetcher=make_fetcher(failures), universe_fetcher=universe_fetcher, sleep=lambda s: None
    )

    assert report.success_count == 0
    assert report.data_complete is False
    assert report.completed is False
    assert stock_files() == []


# --- Nhật ký cập nhật ---------------------------------------------------------

def test_update_log_carries_every_documented_field(temp_store):
    report = run_update(fetcher=make_fetcher(), universe_fetcher=universe_fetcher, sleep=lambda s: None)
    record_sync(report, SYNC_SUCCESS, files=36, message="ok")

    log = json.loads(config.UPDATE_LOG_FILE.read_text(encoding="utf-8"))
    for key in (
        "started_at", "finished_at", "first_run", "mode",
        "index_success", "index_total", "stock_success", "stock_total",
        "stock_missing", "stock_failed", "files_written", "files_expected",
        "sync_status", "sync_files", "failures",
    ):
        assert key in log, f"Thiếu trường {key} trong update_log.json"

    assert log["first_run"] is True
    assert log["index_success"] == 2 and log["index_total"] == 2
    assert log["stock_success"] == 30 and log["stock_total"] == 30
    assert log["files_written"] == log["files_expected"] == 36
    assert log["sync_status"] == SYNC_SUCCESS
    assert log["sync_files"] == 36


def test_completion_requires_a_successful_sync(temp_store):
    report = run_update(fetcher=make_fetcher(), universe_fetcher=universe_fetcher, sleep=lambda s: None)
    assert report.data_complete is True
    assert report.completed is False           # chưa đồng bộ thì chưa hoàn tất

    record_sync(report, SYNC_FAILED, 0, "403 từ GitHub")
    assert report.completed is False
    assert report.data_complete is True        # dữ liệu vẫn lấy được

    record_sync(report, SYNC_SUCCESS, 36, "ok")
    assert report.completed is True


# --- Chất lượng dữ liệu -------------------------------------------------------

@pytest.mark.parametrize("available", [0, 15, 29, 30])
def test_quality_reports_the_real_counts_at_every_stage(temp_store, available):
    storage.write_frame(
        standardize_ohlcv(_frame(400, "VNINDEX")), storage.index_path(config.VNINDEX_DATASET)
    )
    for symbol in UNIVERSE[:available]:
        storage.write_frame(standardize_ohlcv(_frame(300, symbol)), storage.stock_path(symbol))
    universe_module.save_universe(UNIVERSE, source="kiểm thử")

    counts = quality.status_counts(UNIVERSE)
    assert counts[quality.STATUS_COMPLETE] == available
    assert counts[quality.STATUS_MISSING] == 30 - available

    state = build_snapshot(UNIVERSE)
    assert state["ready"] is True              # không bao giờ sập
    assert len(state["missing_symbols"]) == 30 - available

    expected_state = {
        0: breadth_module.DATA_NONE,
        15: breadth_module.DATA_INSUFFICIENT,
        29: breadth_module.DATA_OK,
        30: breadth_module.DATA_OK,
    }[available]
    assert state["breadth"]["data_state"] == expected_state


def test_quality_separates_short_history_from_a_missing_file(temp_store):
    storage.write_frame(standardize_ohlcv(_frame(300, "S00")), storage.stock_path("S00"))
    storage.write_frame(standardize_ohlcv(_frame(40, "S01")), storage.stock_path("S01"))

    table = quality.dataset_rows(["S00", "S01", "S02"])
    status = dict(zip(table["Mã"], table["Trạng thái"]))
    assert status["S00"] == quality.STATUS_COMPLETE
    assert status["S01"] == quality.STATUS_SHORT
    assert status["S02"] == quality.STATUS_MISSING


def test_quality_marks_a_symbol_that_failed_in_the_last_run(temp_store):
    log = {"failures": [{"symbol": "S00", "kind": "transient", "message": "reset"}], "stock_failed": ["S00"]}
    table = quality.dataset_rows(["S00", "S01"], log)
    status = dict(zip(table["Mã"], table["Trạng thái"]))
    assert status["S00"] == quality.STATUS_ERROR
    assert status["S01"] == quality.STATUS_MISSING


def test_legacy_stock_files_are_surfaced(temp_store):
    """Chỉ được tồn tại một đường dẫn cho giá cổ phiếu."""
    storage.write_frame(standardize_ohlcv(_frame(50, "ACB")), config.RAW_DIR / "acb.parquet")
    legacy = storage.legacy_stock_files()
    assert [p.name for p in legacy] == ["acb.parquet"]
