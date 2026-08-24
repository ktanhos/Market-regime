"""Market Data Layer: pipeline cập nhật. Nơi DUY NHẤT khởi động lời gọi API.

Hai chế độ chạy, phân biệt tự động:

* **Khởi tạo lần đầu** — chưa có tệp nào trong ``data/raw/stocks``. Mỗi mã VN30
  được lấy đủ lịch sử (430 ngày lịch) để tính được MA200 của chính nó.
* **Cập nhật tăng dần** — đã có tệp. Chỉ lấy từ ngày cuối cùng trừ cửa sổ chồng
  lấn, không tải lại toàn bộ lịch sử.

Hai chế độ này có thể trộn lẫn: 27 mã đã có tệp thì cập nhật tăng dần, 3 mã còn
thiếu thì lấy đủ lịch sử. Quyết định nằm ở từng mã, không phải ở cả lượt chạy.

Thứ tự các bước::

    1. Lấy danh sách VN30 hiện tại   -> data/reference/vn30_universe.json
    2. Cập nhật VNINDEX               -> data/raw/vnindex.parquet
    3. Cập nhật chỉ số VN30           -> data/raw/vn30.parquet
    4. Cập nhật từng cổ phiếu VN30    -> data/raw/stocks/<MÃ>.parquet
    5. Tính lại features              -> data/processed/*
    6. Đối chiếu tệp thực tế với tệp kỳ vọng
    7. Đồng bộ GitHub (tầng gọi thực hiện, một lần cập nhật = một commit)

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

# Năm giai đoạn hiển thị trên thanh tiến trình, đúng thứ tự chạy.
PHASE_UNIVERSE = "universe"
PHASE_INDEX = "vnindex"
PHASE_VN30_INDEX = "vn30_index"
PHASE_STOCKS = "stocks"
PHASE_FEATURES = "features"
PHASE_SYNC = "github"

PHASE_ORDER = (PHASE_UNIVERSE, PHASE_INDEX, PHASE_VN30_INDEX, PHASE_STOCKS, PHASE_FEATURES, PHASE_SYNC)
PHASE_LABELS = {
    PHASE_UNIVERSE: "VN30 Universe",
    PHASE_INDEX: "VNINDEX",
    PHASE_VN30_INDEX: "VN30 Index",
    PHASE_STOCKS: "VN30 Stocks",
    PHASE_FEATURES: "Features",
    PHASE_SYNC: "GitHub",
}

SYNC_SUCCESS = "success"
SYNC_SKIPPED = "skipped"
SYNC_FAILED = "failed"
SYNC_NOT_RUN = "not_run"

MODE_FIRST_RUN = "first_run"
MODE_PARTIAL = "partial_bootstrap"
MODE_INCREMENTAL = "incremental"


@dataclass
class BootstrapPlan:
    """Mã nào cần lấy đủ lịch sử, mã nào chỉ cần cập nhật tăng dần."""

    symbols: list[str] = field(default_factory=list)
    full_history: list[str] = field(default_factory=list)
    incremental: list[str] = field(default_factory=list)
    short_history: list[str] = field(default_factory=list)

    @property
    def first_run(self) -> bool:
        """Chưa có tệp cổ phiếu nào: đây là lần khởi tạo dữ liệu đầu tiên."""
        return bool(self.symbols) and not self.incremental and not self.short_history

    @property
    def mode(self) -> str:
        if self.first_run:
            return MODE_FIRST_RUN
        if self.full_history or self.short_history:
            return MODE_PARTIAL
        return MODE_INCREMENTAL

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "first_run": self.first_run,
            "symbols": list(self.symbols),
            "full_history": list(self.full_history),
            "incremental": list(self.incremental),
            "short_history": list(self.short_history),
        }


def plan_bootstrap(symbols: Sequence[str]) -> BootstrapPlan:
    """Quyết định cho từng mã: lấy đủ lịch sử hay chỉ cập nhật tăng dần."""
    symbols = sorted({s.upper() for s in symbols})
    inventory = storage.stock_inventory(symbols)
    plan = BootstrapPlan(symbols=symbols)
    for symbol in symbols:
        status = inventory[symbol]["status"]
        if status == storage.STOCK_MISSING:
            plan.full_history.append(symbol)
        elif status == storage.STOCK_SHORT:
            # Có tệp nhưng thiếu lịch sử: nối tiếp phần còn thiếu, không xóa đi làm lại.
            plan.short_history.append(symbol)
            plan.incremental.append(symbol)
        else:
            plan.incremental.append(symbol)
    return plan


@dataclass
class UpdateReport:
    """Kết quả một lần cập nhật, cũng là nhật ký chất lượng dữ liệu."""

    started_at: str = ""
    finished_at: str = ""
    source: str = ""
    vnstock_version: str = ""
    mode: str = MODE_INCREMENTAL
    first_run: bool = False

    index_success: int = 0
    index_total: int = 0
    stock_success: int = 0
    stock_total: int = 0
    stock_missing: list[str] = field(default_factory=list)
    stock_failed: list[str] = field(default_factory=list)

    files_written: int = 0
    files_expected: int = 0
    missing_files: list[str] = field(default_factory=list)

    sync_status: str = SYNC_NOT_RUN
    sync_files: int = 0
    sync_message: str = ""

    datasets: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    universe: dict = field(default_factory=dict)
    bootstrap: dict = field(default_factory=dict)
    rate_limited: bool = False
    aborted_reason: str = ""

    @property
    def total_count(self) -> int:
        return self.index_total + self.stock_total

    @property
    def success_count(self) -> int:
        return self.index_success + self.stock_success

    @property
    def failure_count(self) -> int:
        return len(self.failures)

    @property
    def data_complete(self) -> bool:
        """Mọi nguồn thành công và mọi tệp kỳ vọng đều tồn tại."""
        return (
            self.total_count > 0
            and self.success_count == self.total_count
            and self.files_written == self.files_expected
        )

    @property
    def completed(self) -> bool:
        """Chỉ hoàn tất khi dữ liệu đủ VÀ đã đồng bộ được lên GitHub."""
        return self.data_complete and self.sync_status == SYNC_SUCCESS

    def as_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "source": self.source,
            "vnstock_version": self.vnstock_version,
            "mode": self.mode,
            "first_run": self.first_run,
            "index_success": self.index_success,
            "index_total": self.index_total,
            "stock_success": self.stock_success,
            "stock_total": self.stock_total,
            "stock_missing": list(self.stock_missing),
            "stock_failed": list(self.stock_failed),
            "files_written": self.files_written,
            "files_expected": self.files_expected,
            "missing_files": list(self.missing_files),
            "sync_status": self.sync_status,
            "sync_files": self.sync_files,
            "sync_message": self.sync_message,
            "total_count": self.total_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "data_complete": self.data_complete,
            "completed": self.completed,
            "datasets": self.datasets,
            "failures": self.failures,
            "universe": self.universe,
            "bootstrap": self.bootstrap,
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
    """Cập nhật một tập dữ liệu và trả về báo cáo chất lượng.

    Ngày bắt đầu do ``default_start`` quyết định: chưa có tệp thì lấy đủ lịch
    sử, đã có tệp thì nối tiếp từ ngày cuối trừ cửa sổ chồng lấn.
    """
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
        mode="full_history" if last is None else "incremental",
        requested_start=start,
        requested_end=end,
        previous_last_date=None if last is None else pd.Timestamp(last).strftime("%Y-%m-%d"),
        rows_fetched=int(len(result.frame)),
        path=str(path),
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
        logger.warning("Bỏ qua %s: %s", symbol, str(exc)[:200])
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


def _resolve_universe(
    report: UpdateReport,
    refresh_universe: bool,
    universe_fetcher: Callable,
) -> tuple[list[str], bool]:
    """Bước 1: lấy và lưu danh sách VN30 hiện tại. Trả về (symbols, stopped)."""
    current = universe_module.load_universe()
    symbols = current["symbols"]

    if not refresh_universe:
        report.universe = {"status": "giữ nguyên", **current}
        return symbols, False

    try:
        fetched = universe_fetcher()
        saved = universe_module.save_universe(
            fetched,
            source=f"vnstock Listing('{config.PRIMARY_SOURCE}').symbols_by_group('VN30')",
        )
        report.universe = {"status": "cập nhật từ API", **saved}
        logger.info("Danh sách VN30 cập nhật từ API: %d mã", len(saved["symbols"]))
        return saved["symbols"], False
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
            return symbols, True
        return symbols, False


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

    legacy = storage.legacy_stock_files()
    if legacy:
        logger.warning(
            "Còn %d tệp ở bố cục cũ data/raw/*.parquet: %s",
            len(legacy),
            ", ".join(p.name for p in legacy),
        )

    # --- Bước 1: danh sách VN30 hiện tại -------------------------------------
    notify(PHASE_UNIVERSE, 0.2, "Đang lấy danh sách thành phần VN30")
    symbols, stopped = _resolve_universe(report, refresh_universe, universe_fetcher)
    notify(PHASE_UNIVERSE, 1.0, f"{len(symbols)} mã VN30")

    plan = plan_bootstrap(symbols)
    report.bootstrap = plan.as_dict()
    report.mode = plan.mode
    report.first_run = plan.first_run
    report.stock_total = len(symbols)
    if plan.first_run:
        logger.info("Đang khởi tạo dữ liệu VN30 lần đầu cho %d mã", len(symbols))
    elif plan.full_history:
        logger.info(
            "Khởi tạo bổ sung: %d mã lấy đủ lịch sử, %d mã cập nhật tăng dần",
            len(plan.full_history), len(plan.incremental),
        )

    # --- Bước 2 và 3: hai chỉ số ---------------------------------------------
    index_jobs = [
        (PHASE_INDEX, "VNINDEX", "VNINDEX", config.VNINDEX_DATASET),
        (PHASE_VN30_INDEX, "VN30", "VN30", config.VN30_INDEX_DATASET),
    ]
    report.index_total = len(index_jobs)

    for phase, label, symbol, dataset in index_jobs:
        if stopped:
            break
        notify(phase, 0.3, f"Đang cập nhật {label}")
        try:
            info = _update_one(
                label, symbol, ASSET_INDEX, storage.index_path(dataset),
                end, fetcher, sleep, min_rows=100,
            )
            report.datasets.append(info)
            report.index_success += 1
        except Exception as exc:
            stopped = _record_failure(report, label, exc)
        notify(phase, 1.0, f"Xong {label}")
        if not stopped:
            sleep(config.REQUEST_DELAY_SECONDS)

    # --- Bước 4: từng cổ phiếu VN30 ------------------------------------------
    if not stopped and symbols:
        for i, symbol in enumerate(symbols, start=1):
            mode = "lịch sử đầy đủ" if symbol in plan.full_history else "cập nhật"
            notify(PHASE_STOCKS, (i - 1) / len(symbols), f"{symbol} {i}/{len(symbols)} ({mode})")
            try:
                info = _update_one(
                    symbol, symbol, ASSET_STOCK, storage.stock_path(symbol),
                    end, fetcher, sleep, min_rows=20,
                )
                report.datasets.append(info)
                report.stock_success += 1
            except Exception as exc:
                report.stock_failed.append(symbol)
                if _record_failure(report, symbol, exc):
                    # Rate limit: giữ nguyên các mã đã lấy được, không rollback.
                    report.aborted_reason += (
                        f" Đã dừng sau {i}/{len(symbols)} mã. "
                        "Các mã đã lấy được vẫn được giữ lại, lần cập nhật sau sẽ tiếp tục phần còn thiếu."
                    )
                    break
            if i < len(symbols):
                sleep(config.REQUEST_DELAY_SECONDS)
    notify(PHASE_STOCKS, 1.0, f"{report.stock_success}/{report.stock_total} mã")

    # --- Bước 5: tính lại chỉ tiêu -------------------------------------------
    notify(PHASE_FEATURES, 0.4, "Đang tính Breadth, Dispersion và Risk Concentration")
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

    # --- Bước 6: đối chiếu tệp thực tế ---------------------------------------
    report.stock_missing = [
        symbol for symbol in symbols if not storage.stock_path(symbol).exists()
    ]
    report.finished_at = storage.utc_now_iso()
    _refresh_file_audit(report, symbols)
    logger.info(
        "Kết thúc cập nhật [%s]: chỉ số %d/%d, cổ phiếu %d/%d, tệp %d/%d",
        report.mode, report.index_success, report.index_total,
        report.stock_success, report.stock_total,
        report.files_written, report.files_expected,
    )
    return report


def _refresh_file_audit(report: UpdateReport, symbols: Sequence[str]) -> None:
    """Đếm lại tệp thực sự có trên đĩa rồi ghi nhật ký.

    Nhật ký được ghi trước khi đếm vì chính nó nằm trong danh sách tệp kỳ vọng:
    ở lần khởi tạo đầu tiên tệp này chưa tồn tại và sẽ bị đếm thiếu. Sau khi
    đếm xong thì ghi lại lần nữa với con số cuối cùng.
    """
    storage.write_json(config.UPDATE_LOG_FILE, report.as_dict())
    audit = storage.verify_data_files(symbols)
    report.files_written = audit["files_written"]
    report.files_expected = audit["files_expected"]
    report.missing_files = audit["missing_names"]
    storage.write_json(config.UPDATE_LOG_FILE, report.as_dict())


def record_sync(report: UpdateReport, status: str, files: int = 0, message: str = "") -> UpdateReport:
    """Ghi kết quả đồng bộ GitHub vào báo cáo và nhật ký.

    Được gọi sau ``run_update`` vì việc đồng bộ do tầng gọi thực hiện: đó là thứ
    giữ cho một lần cập nhật chỉ tạo đúng một commit.
    """
    report.sync_status = status
    report.sync_files = int(files)
    report.sync_message = message
    symbols = report.bootstrap.get("symbols") or universe_module.symbols()
    _refresh_file_audit(report, symbols)
    logger.info("Đồng bộ GitHub: %s (%d tệp)", status, files)
    return report


def rebuild_features(symbols: Sequence[str] | None = None) -> dict:
    """Tính lại chỉ tiêu từ dữ liệu đã lưu. Không gọi mạng."""
    return features.rebuild(symbols)
