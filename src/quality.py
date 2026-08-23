"""Báo cáo chất lượng dữ liệu.

Mỗi lần cập nhật đều phải trả lời được: dữ liệu tới ngày nào, bao nhiêu phiên,
bao nhiêu mã thành công, bao nhiêu mã thất bại và vì sao, nguồn nào, lúc nào,
khoảng dữ liệu là gì, số dòng trước và sau khi hợp nhất.
"""

from __future__ import annotations

import pandas as pd

from src import config, storage
from src.schema import DATE_COLUMN, inspect_frame


def dataset_rows(universe: list[str]) -> pd.DataFrame:
    """Tình trạng từng tệp dữ liệu đang có trên đĩa."""
    rows: list[dict] = []

    for dataset, label in ((config.VNINDEX_DATASET, "VNINDEX"), (config.VN30_INDEX_DATASET, "VN30")):
        frame = storage.load_index(dataset)
        rows.append(_row(label, "Chỉ số", frame))

    for symbol in sorted({s.upper() for s in universe}):
        rows.append(_row(symbol, "Cổ phiếu VN30", storage.load_stock(symbol)))

    return pd.DataFrame(rows)


def _row(name: str, kind: str, frame: pd.DataFrame | None) -> dict:
    if frame is None or frame.empty:
        return {
            "Mã": name,
            "Loại": kind,
            "Trạng thái": "Thiếu dữ liệu",
            "Số phiên": 0,
            "Từ ngày": "",
            "Đến ngày": "",
            "Ngày trùng": 0,
            "Ghi chú": "Chưa có tệp dữ liệu",
        }
    report = inspect_frame(frame)
    return {
        "Mã": name,
        "Loại": kind,
        "Trạng thái": "Có dữ liệu",
        "Số phiên": report.rows,
        "Từ ngày": report.first_date.strftime("%d/%m/%Y"),
        "Đến ngày": report.last_date.strftime("%d/%m/%Y"),
        "Ngày trùng": report.duplicate_dates,
        "Ghi chú": "; ".join(report.warnings),
    }


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

    return {
        "indices": indices,
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
