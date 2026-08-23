"""Market Data Layer: pipeline cập nhật. Nơi DUY NHẤT khởi động lời gọi API.

Luồng đúng theo thứ tự::

    1. Kiểm tra dữ liệu đã lưu, xác định ngày cuối cùng của từng tập
    2. Lấy VNINDEX phần còn thiếu
    3. Lấy danh sách VN30 hiện tại từ API
    4. Lấy dữ liệu từng mã VN30, tuần tự, có nghỉ
    5. Chuẩn hóa
    6. Gộp, loại ngày trùng
    7. Tính lại features
    8. Lưu Parquet + processed data
    9. Đồng bộ GitHub (do tầng gọi thực hiện, sau khi pipeline kết thúc,
       để một lần cập nhật chỉ tạo đúng một commit)

Module này không import Streamlit và không biết gì về giao diện.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from src import config, features, storage
from src import universe as universe_module
from src.logging_config import get_logger
from src.vnstock_data import (
    ASSET_INDEX,
    ASSET_STOCK,
    RATE_LIMITED,
    FetchError,
    default_start,
    fetch_history,
    fetch_index_members,
    friendly_message,
    vnstock_version,
)

logger = get_logger(__name__)

# Năm giai đoạn hiển thị trên thanh tiến trình.
PHASE_INDEX = "vnindex"
PHASE_UNIVERSE = "universe"
PHASE_STOCKS = "stocks"
PHASE_FEATURES = "features"
PHASE_SYNC = "github"

PHASE_LABELS = {
    PHASE_INDEX: "VNINDEX",
    PHASE_UNIVERSE: "VN30 Universe",
    PHASE_STOCKS: "VN30 Stocks",
    PHASE_FEATURES: "Features",
    PHASE_SYNC: "GitHub",
}


@dataclass
class UpdateReport:
    """Kết quả một lần cập nhật, cũng là nhật ký chất lượng dữ liệu."""

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

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def completed(self) -> bool:
        """Chỉ coi là hoàn tất khi mọi nguồn đều thành công."""
        return self.total_count > 0 and self.success_count == self.total_count

    def as_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "source": self.source,
            "vnstock_version": self.vnstock_version,
            "total_count": self.total_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "completed": self.completed,
            "datasets": self.datasets,
            "failures": self.failures,
            "universe": self.universe,
            "rate_limited": self.rate_limited,
            "aborted_reason": self.aborted_reason,
        }


ProgressCallback = Callable[[str, float, str], None]


def _noop(*_args, **_kwargs) -> None:
    return None


def _update_one(
    name: str,
    symbol: str,
    asset_type: str,
    path: Path,
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


def _record_failure(report: UpdateReport, symbol: str, exc: BaseException) -> bool:
    """Ghi lại một lỗi. Trả về True nếu phải dừng cả lượt chạy."""
    if isinstance(exc, FetchError):
        report.failures.append(exc.as_dict())
        if exc.kind == RATE_LIMITED:
            report.rate_limited = True
            report.aborted_reason = friendly_message(RATE_LIMITED)
            logger.error("Dừng cập nhật vì bị giới hạn truy cập tại %s", symbol)
            return True
        return False

    report.failures.append(
        {
            "symbol": symbol,
            "source": config.PRIMARY_SOURCE,
            "kind": "unexpected",
            "message": f"{type(exc).__name__}: {str(exc)[:250]}",
        }
    )
    logger.exception("Lỗi ngoài dự kiến khi cập nhật %s", symbol)
    return False


def run_update(
    refresh_universe: bool = True,
    progress: ProgressCallback | None = None,
    sleep: Callable[[float], None] = time.sleep,
    fetcher: Callable = fetch_history,
    universe_fetcher: Callable = fetch_index_members,
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
    logger.info("Bắt đầu cập nhật dữ liệu tới %s", end)

    # --- 1. VNINDEX và chỉ số VN30 ------------------------------------------
    index_jobs = [
        ("VNINDEX", "VNINDEX", config.VNINDEX_DATASET),
        ("VN30", "VN30", config.VN30_INDEX_DATASET),
    ]
    stopped = False
    for position, (label, symbol, dataset) in enumerate(index_jobs):
        notify(PHASE_INDEX, position / len(index_jobs), f"Chỉ số {label}")
        try:
            info = _update_one(
                label, symbol, ASSET_INDEX, storage.index_path(dataset),
                end, fetcher, sleep, min_rows=100,
            )
            report.datasets.append(info)
            report.success_count += 1
        except Exception as exc:
            stopped = _record_failure(report, label, exc)
            if stopped:
                break
        sleep(config.REQUEST_DELAY_SECONDS)
    notify(PHASE_INDEX, 1.0, "Xong chỉ số")

    # --- 2. Danh sách VN30 hiện tại ------------------------------------------
    notify(PHASE_UNIVERSE, 0.3, "Đang lấy danh sách VN30 hiện tại")
    current = universe_module.load_universe()
    symbols = current["symbols"]
    if stopped:
        report.universe = {"status": "bỏ qua", **current}
    elif refresh_universe:
        try:
            fetched = universe_fetcher()
            saved = universe_module.save_universe(
                fetched,
                source=f"vnstock Listing('{config.PRIMARY_SOURCE}').symbols_by_group('VN30')",
            )
            symbols = saved["symbols"]
            report.universe = {"status": "cập nhật từ API", **saved}
        except FetchError as exc:
            report.universe = {
                "status": "dùng danh sách đã lưu",
                "symbols": symbols,
                "as_of": current.get("as_of", ""),
                "source": current.get("source", ""),
                "is_fallback": current.get("is_fallback", True),
                "error": str(exc)[:300],
            }
            report.failures.append(exc.as_dict())
            logger.warning("Không lấy được danh sách VN30, dùng ảnh chụp đã lưu")
            if exc.kind == RATE_LIMITED:
                report.rate_limited = True
                report.aborted_reason = friendly_message(RATE_LIMITED)
                stopped = True
    else:
        report.universe = {"status": "giữ nguyên", **current}
    notify(PHASE_UNIVERSE, 1.0, f"{len(symbols)} mã VN30")

    report.total_count = len(index_jobs) + len(symbols)

    # --- 3. Cổ phiếu VN30 hiện tại -------------------------------------------
    if not stopped:
        for i, symbol in enumerate(symbols, start=1):
            notify(PHASE_STOCKS, (i - 1) / len(symbols), f"{symbol} ({i}/{len(symbols)})")
            try:
                info = _update_one(
                    symbol, symbol, ASSET_STOCK, storage.stock_path(symbol),
                    end, fetcher, sleep, min_rows=20,
                )
                report.datasets.append(info)
                report.success_count += 1
            except Exception as exc:
                if _record_failure(report, symbol, exc):
                    report.aborted_reason += (
                        f" Đã dừng sau {i}/{len(symbols)} mã để không làm nặng thêm."
                    )
                    break
            if i < len(symbols):
                sleep(config.REQUEST_DELAY_SECONDS)
    notify(PHASE_STOCKS, 1.0, "Xong cổ phiếu")

    # --- 4. Tính lại chỉ tiêu -------------------------------------------------
    notify(PHASE_FEATURES, 0.4, "Đang tính lại các chỉ tiêu")
    try:
        features.rebuild(symbols)
    except Exception as exc:
        report.failures.append(
            {
                "symbol": "features",
                "source": "local",
                "kind": "compute",
                "message": f"{type(exc).__name__}: {str(exc)[:250]}",
            }
        )
        logger.exception("Không tính lại được chỉ tiêu")
    notify(PHASE_FEATURES, 1.0, "Xong tính toán")

    report.finished_at = storage.utc_now_iso()
    storage.write_json(config.UPDATE_LOG_FILE, report.as_dict())
    logger.info(
        "Kết thúc cập nhật: %d/%d nguồn thành công",
        report.success_count,
        report.total_count,
    )
    return report


def rebuild_features(symbols: Sequence[str] | None = None) -> dict:
    """Tính lại chỉ tiêu từ dữ liệu đã lưu. Không gọi mạng."""
    return features.rebuild(symbols)
