"""Pipeline cập nhật dữ liệu. Đây là nơi DUY NHẤT gọi API trong ứng dụng.

Luồng::

    kiểm tra dữ liệu đã lưu
    -> xác định ngày cuối cùng của từng tập
    -> gọi API phần còn thiếu (tuần tự, có nghỉ, có backoff)
    -> chuẩn hóa
    -> kiểm tra dữ liệu
    -> hợp nhất, loại ngày trùng
    -> lưu Parquet
    -> tính lại chỉ tiêu
    -> lưu processed data + nhật ký chất lượng

Đồng bộ GitHub được thực hiện ở tầng gọi (``app.py`` hoặc script), sau khi
pipeline này kết thúc, để một lần cập nhật chỉ tạo đúng một commit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Sequence

import pandas as pd

from src import breadth as breadth_module
from src import concentration as concentration_module
from src import config
from src import dispersion as dispersion_module
from src import portfolio as portfolio_module
from src import regime as regime_module
from src import roro as roro_module
from src import storage
from src import universe as universe_module
from src import volatility as volatility_module
from src.schema import DATE_COLUMN, to_close_panel
from src.vnstock_data import (
    RATE_LIMITED,
    FetchError,
    default_start,
    fetch_history,
    fetch_vn30_constituents,
    friendly_message,
    vnstock_version,
)

PHASE_UNIVERSE = "universe"
PHASE_INDEX = "index"
PHASE_STOCKS = "stocks"
PHASE_FEATURES = "features"


@dataclass
class UpdateReport:
    started_at: str = ""
    finished_at: str = ""
    source: str = ""
    vnstock_version: str = ""
    total_count: int = 0
    success_count: int = 0
    datasets: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    universe: dict = field(default_factory=dict)
    rate_limited: bool = False
    aborted_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "source": self.source,
            "vnstock_version": self.vnstock_version,
            "total_count": self.total_count,
            "success_count": self.success_count,
            "failure_count": len(self.failures),
            "datasets": self.datasets,
            "failures": self.failures,
            "universe": self.universe,
            "rate_limited": self.rate_limited,
            "aborted_reason": self.aborted_reason,
        }


def _noop(*_args, **_kwargs) -> None:
    return None


def _update_one(
    name: str,
    symbol: str,
    asset_type: str,
    path,
    end: str,
    fetcher: Callable,
    sleep: Callable[[float], None],
    min_rows: int,
) -> dict:
    """Cập nhật tăng dần một tập dữ liệu và trả về báo cáo chất lượng."""
    last = storage.last_stored_date(path)
    start = default_start(asset_type, last)
    result = fetcher(symbol, start=start, end=end, asset_type=asset_type, sleep=sleep)
    report = storage.merge_and_write(path, result.frame, min_rows=min_rows)
    report.update(
        name=name,
        symbol=symbol,
        asset_type=asset_type,
        source=result.source,
        attempts=result.attempts,
        requested_start=start,
        requested_end=end,
        previous_last_date=None if last is None else pd.Timestamp(last).strftime("%Y-%m-%d"),
        rows_fetched=int(len(result.frame)),
    )
    return report


def run_update(
    refresh_universe: bool = True,
    progress: Callable[[str, float, str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    fetcher: Callable = fetch_history,
    universe_fetcher: Callable = fetch_vn30_constituents,
    end: str | None = None,
) -> UpdateReport:
    """Chạy toàn bộ pipeline cập nhật.

    ``fetcher`` và ``universe_fetcher`` được tách ra để kiểm thử được mà không
    cần mạng.
    """
    notify = progress or _noop
    report = UpdateReport(
        started_at=storage.utc_now_iso(),
        source=config.PRIMARY_SOURCE,
        vnstock_version=vnstock_version(),
    )
    storage.ensure_dirs()
    end = end or date.today().isoformat()

    # 1. Danh sách VN30 hiện tại -------------------------------------------------
    notify(PHASE_UNIVERSE, 0.02, "Đang xác định danh sách VN30 hiện tại")
    current = universe_module.load_universe()
    symbols = current["symbols"]
    if refresh_universe:
        try:
            fetched = universe_fetcher()
            saved = universe_module.save_universe(
                fetched, source=f"vnstock Listing({config.PRIMARY_SOURCE}).symbols_by_group('VN30')"
            )
            symbols = saved["symbols"]
            report.universe = {"status": "cập nhật", **saved}
        except FetchError as exc:
            report.universe = {
                "status": "dùng danh sách đã lưu",
                "symbols": symbols,
                "as_of": current.get("as_of", ""),
                "source": current.get("source", ""),
                "error": str(exc)[:300],
            }
            if exc.kind == RATE_LIMITED:
                report.rate_limited = True
                report.aborted_reason = friendly_message(RATE_LIMITED)
                report.finished_at = storage.utc_now_iso()
                storage.write_json(config.UPDATE_LOG_FILE, report.as_dict())
                return report
    else:
        report.universe = {"status": "giữ nguyên", **current}

    # 2. Chỉ số ------------------------------------------------------------------
    index_jobs = [
        ("VNINDEX", "VNINDEX", config.VNINDEX_DATASET),
        ("VN30", "VN30", config.VN30_INDEX_DATASET),
    ]
    jobs_total = len(index_jobs) + len(symbols)
    report.total_count = jobs_total
    done = 0

    for label, symbol, dataset in index_jobs:
        notify(PHASE_INDEX, done / jobs_total, f"Chỉ số {label}")
        try:
            info = _update_one(
                label, symbol, "index", storage.index_path(dataset), end, fetcher, sleep, min_rows=100
            )
            report.datasets.append(info)
            report.success_count += 1
        except FetchError as exc:
            report.failures.append(exc.as_dict())
            if exc.kind == RATE_LIMITED:
                report.rate_limited = True
                report.aborted_reason = friendly_message(RATE_LIMITED)
                break
        except Exception as exc:
            report.failures.append(
                {"symbol": label, "source": config.PRIMARY_SOURCE, "kind": "unexpected",
                 "message": f"{type(exc).__name__}: {str(exc)[:250]}"}
            )
        done += 1
        sleep(config.REQUEST_DELAY_SECONDS)

    # 3. Cổ phiếu VN30 hiện tại ---------------------------------------------------
    if not report.rate_limited:
        for i, symbol in enumerate(symbols, start=1):
            notify(PHASE_STOCKS, done / jobs_total, f"Cổ phiếu {symbol} ({i}/{len(symbols)})")
            try:
                info = _update_one(
                    symbol, symbol, "stock", storage.stock_path(symbol), end, fetcher, sleep, min_rows=20
                )
                report.datasets.append(info)
                report.success_count += 1
            except FetchError as exc:
                report.failures.append(exc.as_dict())
                if exc.kind == RATE_LIMITED:
                    report.rate_limited = True
                    report.aborted_reason = (
                        friendly_message(RATE_LIMITED)
                        + f" Đã dừng sau {i}/{len(symbols)} mã để không làm nặng thêm."
                    )
                    break
            except Exception as exc:
                report.failures.append(
                    {"symbol": symbol, "source": config.PRIMARY_SOURCE, "kind": "unexpected",
                     "message": f"{type(exc).__name__}: {str(exc)[:250]}"}
                )
            done += 1
            if i < len(symbols):
                sleep(config.REQUEST_DELAY_SECONDS)

    # 4. Tính lại chỉ tiêu --------------------------------------------------------
    notify(PHASE_FEATURES, 0.92, "Đang tính lại các chỉ tiêu")
    try:
        rebuild_features(symbols)
    except Exception as exc:
        report.failures.append(
            {"symbol": "features", "source": "local", "kind": "compute",
             "message": f"{type(exc).__name__}: {str(exc)[:250]}"}
        )

    report.finished_at = storage.utc_now_iso()
    storage.write_json(config.UPDATE_LOG_FILE, report.as_dict())
    notify(PHASE_FEATURES, 1.0, "Hoàn tất")
    return report


def rebuild_features(symbols: Sequence[str] | None = None) -> dict:
    """Tính lại toàn bộ chỉ tiêu từ dữ liệu đã lưu. Không gọi mạng."""
    symbols = list(symbols or universe_module.symbols())
    storage.ensure_dirs()

    index_frame = storage.load_index(config.VNINDEX_DATASET)
    if index_frame is None or index_frame.empty:
        raise RuntimeError("Chưa có dữ liệu VNINDEX để tính chỉ tiêu")

    trend = roro_module.trend_frame(index_frame["close"])
    stress = volatility_module.stress_frame(index_frame)
    features = pd.concat(
        [index_frame.reset_index(drop=True), trend.reset_index(drop=True), stress.reset_index(drop=True)],
        axis=1,
    )
    features.to_parquet(config.VNINDEX_FEATURES_FILE, index=False)

    frames, _missing = storage.load_stocks(symbols)
    panel = to_close_panel(frames)
    breadth = breadth_module.compute_breadth(frames, symbols)
    dispersion = dispersion_module.compute_dispersion(panel)
    concentration = concentration_module.compute_concentration(panel)

    snapshot = {
        "generated_at": storage.utc_now_iso(),
        "as_of": None if breadth.get("as_of") is None else pd.Timestamp(breadth["as_of"]).strftime("%Y-%m-%d"),
        "universe_size": breadth["universe_size"],
        "valid_symbols": breadth.get("valid_symbols", 0),
        "min_valid_symbols": breadth.get("min_valid_symbols", 0),
        "max_valid_symbols": breadth.get("max_valid_symbols", 0),
        "missing_symbols": breadth["missing_symbols"],
        "breadth_score": breadth["score"],
        "breadth_state": breadth["state"],
        "breadth_components": {
            key: {"pct": value["pct"], "valid": value["valid"], "window": value["window"]}
            for key, value in breadth["components"].items()
        },
        "dispersion": {
            "value": dispersion["value"],
            "percentile": dispersion["percentile"],
            "state": dispersion["state"],
            "windows": dispersion["windows"],
            "historical_basis": dispersion["historical_basis"],
        },
        "concentration": {
            "hhi": concentration["hhi"],
            "effective_names": concentration["effective_names"],
            "top_shares": concentration["top_shares"],
            "percentile": concentration["percentile"],
            "state": concentration["state"],
            "contributors": concentration["contributors"],
            "proxy_note": concentration["proxy_note"],
        },
    }
    storage.write_json(config.VN30_SNAPSHOT_FILE, snapshot)
    return snapshot


def build_market_state(symbols: Sequence[str] | None = None) -> dict:
    """Toàn bộ dữ liệu dashboard cần, đọc từ đĩa. KHÔNG gọi API."""
    meta = universe_module.load_universe()
    symbols = list(symbols or meta["symbols"])

    index_frame = storage.load_index(config.VNINDEX_DATASET)
    if index_frame is None or index_frame.empty:
        return {"ready": False, "reason": "Chưa có dữ liệu VNINDEX trong kho dữ liệu.", "universe": meta}

    trend = roro_module.trend_snapshot(index_frame["close"])
    stress = volatility_module.stress_snapshot(index_frame)

    frames, missing = storage.load_stocks(symbols)
    panel = to_close_panel(frames)
    breadth = breadth_module.compute_breadth(frames, symbols)
    dispersion = dispersion_module.compute_dispersion(panel)
    concentration = concentration_module.compute_concentration(panel)

    regime = regime_module.build_regime(trend, stress, breadth, dispersion, concentration)
    return {
        "ready": True,
        "index": index_frame,
        "as_of": pd.to_datetime(index_frame[DATE_COLUMN]).max(),
        "trend": trend,
        "stress": stress,
        "breadth": breadth,
        "dispersion": dispersion,
        "concentration": concentration,
        "regime": regime,
        "portfolio": portfolio_module.guidance(regime),
        "universe": meta,
        "missing_symbols": missing,
        "update_log": storage.read_json(config.UPDATE_LOG_FILE) or {},
    }
