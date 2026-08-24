"""Báo cáo chất lượng dữ liệu.

Mỗi lần cập nhật đều phải trả lời được: dữ liệu tới ngày nào, bao nhiêu phiên,
bao nhiêu mã thành công, bao nhiêu mã thất bại và vì sao, nguồn nào, lúc nào,
khoảng dữ liệu là gì, số dòng trước và sau khi hợp nhất.
"""

from __future__ import annotations

import pandas as pd

from src import config, storage
from src.schema import DATE_COLUMN, inspect_frame


# Bốn trạng thái của một tệp dữ liệu, theo đúng thứ tự nghiêm trọng tăng dần.
STATUS_COMPLETE = "Đầy đủ"
STATUS_SHORT = "Thiếu lịch sử"
STATUS_MISSING = "Không có tệp"
STATUS_ERROR = "Lỗi cập nhật"


def _failed_symbols(log: dict | None) -> set[str]:
    if not log:
        return set()
    failed = {str(item.get("symbol", "")).upper() for item in (log.get("failures") or [])}
    failed |= {str(s).upper() for s in (log.get("stock_failed") or [])}
    return {s for s in failed if s}


def dataset_rows(universe: list[str], log: dict | None = None) -> pd.DataFrame:
    """Tình trạng từng tệp dữ liệu đang có trên đĩa.

    Phân biệt rõ "không có tệp", "có tệp nhưng thiếu lịch sử" và "lỗi ở lần cập
    nhật gần nhất" thay vì gộp tất cả thành một nhãn thiếu dữ liệu.
    """
    log = log if log is not None else storage.read_json(config.UPDATE_LOG_FILE)
    failed = _failed_symbols(log)
    rows: list[dict] = []

    for dataset, label in ((config.VNINDEX_DATASET, "VNINDEX"), (config.VN30_INDEX_DATASET, "VN30")):
        rows.append(
            _row(label, "Chỉ số", storage.load_index(dataset), failed, min_sessions=100)
        )

    for symbol in sorted({s.upper() for s in universe}):
        rows.append(
            _row(
                symbol, "Cổ phiếu VN30", storage.load_stock(symbol), failed,
                min_sessions=config.MIN_SESSIONS_FOR_FULL_HISTORY,
            )
        )

    return pd.DataFrame(rows)


def _row(
    name: str,
    kind: str,
    frame: pd.DataFrame | None,
    failed: set[str],
    min_sessions: int,
) -> dict:
    has_error = name.upper() in failed
    if frame is None or frame.empty:
        return {
            "Mã": name,
            "Loại": kind,
            "Trạng thái": STATUS_ERROR if has_error else STATUS_MISSING,
            "Số phiên": 0,
            "Từ ngày": "",
            "Đến ngày": "",
            "Ngày trùng": 0,
            "Ghi chú": "Lần cập nhật gần nhất thất bại" if has_error else "Chưa có tệp dữ liệu",
        }

    report = inspect_frame(frame)
    if has_error:
        status = STATUS_ERROR
        note = "Có dữ liệu cũ nhưng lần cập nhật gần nhất thất bại"
    elif report.rows < min_sessions:
        status = STATUS_SHORT
        note = f"Mới có {report.rows} phiên, cần tối thiểu {min_sessions}"
    else:
        status = STATUS_COMPLETE
        note = "; ".join(report.warnings)

    return {
        "Mã": name,
        "Loại": kind,
        "Trạng thái": status,
        "Số phiên": report.rows,
        "Từ ngày": report.first_date.strftime("%d/%m/%Y"),
        "Đến ngày": report.last_date.strftime("%d/%m/%Y"),
        "Ngày trùng": report.duplicate_dates,
        "Ghi chú": note,
    }


def status_counts(universe: list[str], log: dict | None = None) -> dict[str, int]:
    """Đếm số mã cổ phiếu theo từng trạng thái."""
    table = dataset_rows(universe, log)
    stocks = table[table["Loại"] == "Cổ phiếu VN30"]
    counts = {
        STATUS_COMPLETE: 0, STATUS_SHORT: 0, STATUS_MISSING: 0, STATUS_ERROR: 0,
    }
    for status, count in stocks["Trạng thái"].value_counts().items():
        counts[str(status)] = int(count)
    return counts


def index_summary() -> dict:
    """Số phiên và khoảng dữ liệu của hai chỉ số."""
    rows = {}
    for dataset, label in ((config.VNINDEX_DATASET, "VNINDEX"), (config.VN30_INDEX_DATASET, "VN30")):
        frame = storage.load_index(dataset)
        if frame is None or frame.empty:
            rows[label] = {"sessions": 0, "first_date": None, "last_date": None}
            continue
        dates = pd.to_datetime(frame[DATE_COLUMN])
        rows[label] = {
            "sessions": int(len(frame)),
            "first_date": dates.min(),
            "last_date": dates.max(),
        }
    return rows


def coverage_summary(universe: list[str]) -> dict:
    """Tổng quan độ phủ dữ liệu, dùng cho thẻ Chất lượng dữ liệu."""
    universe = sorted({s.upper() for s in universe})
    frames, missing = storage.load_stocks(universe)
    indices = index_summary()
    index_frame = storage.load_index(config.VNINDEX_DATASET)

    last_dates = [
        pd.to_datetime(f[DATE_COLUMN]).max() for f in frames.values() if f is not None and not f.empty
    ]
    index_last = None
    index_rows = 0
    index_first = None
    if index_frame is not None and not index_frame.empty:
        dates = pd.to_datetime(index_frame[DATE_COLUMN])
        index_last = dates.max()
        index_first = dates.min()
        index_rows = int(len(index_frame))

    spread_days = None
    if last_dates:
        spread_days = int((max(last_dates) - min(last_dates)).days)

    log = storage.read_json(config.UPDATE_LOG_FILE) or {}
    universe_meta = storage.read_json(config.UNIVERSE_FILE) or {}
    audit = storage.verify_data_files(universe)

    return {
        "indices": indices,
        "statuses": status_counts(universe, log),
        "files_written": audit["files_written"],
        "files_expected": audit["files_expected"],
        "legacy_files": [p.name for p in storage.legacy_stock_files()],
        "never_updated": not log,
        "first_run": bool(log.get("first_run")) if log else True,
        "sync_status": log.get("sync_status", ""),
        "sync_files": log.get("sync_files", 0),
        "index_success": log.get("index_success"),
        "index_total": log.get("index_total"),
        "stock_success": log.get("stock_success"),
        "stock_total": log.get("stock_total"),
        "index_first_date": index_first,
        "index_last_date": index_last,
        "index_sessions": index_rows,
        "stock_symbols_expected": len(universe),
        "stock_symbols_available": len(frames),
        "stock_symbols_missing": missing,
        "stock_last_date": max(last_dates) if last_dates else None,
        "stock_date_spread_days": spread_days,
        "last_update": log.get("finished_at", ""),
        "last_update_source": log.get("source", ""),
        "last_update_success": log.get("success_count"),
        "last_update_total": log.get("total_count"),
        "universe_as_of": universe_meta.get("as_of", ""),
        "universe_source": universe_meta.get("source", ""),
        "universe_is_fallback": bool(universe_meta.get("is_fallback", True)),
    }


def failure_table(log: dict | None) -> pd.DataFrame:
    """Bảng các mã lỗi trong lần cập nhật gần nhất, kèm nguyên nhân."""
    if not log:
        return pd.DataFrame()
    failures = log.get("failures") or []
    if not failures:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "Mã": item.get("symbol", ""),
                "Nguồn": item.get("source", ""),
                "Loại lỗi": item.get("kind", ""),
                "Nguyên nhân": item.get("message", ""),
            }
            for item in failures
        ]
    )
