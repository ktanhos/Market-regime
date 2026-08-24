"""Lớp lưu trữ cục bộ. Không có bất kỳ lời gọi mạng nào trong module này.

Dashboard chỉ được phép đi qua lớp này. Mọi thứ liên quan tới API nằm ở
``src.vnstock_data`` và chỉ được gọi từ ``src.updater``.

Bố cục kho dữ liệu::

    data/raw/vnindex.parquet          chỉ số VNINDEX
    data/raw/vn30.parquet             chỉ số VN30
    data/raw/stocks/<MÃ>.parquet      giá cổ phiếu thành phần VN30 hiện tại
    data/reference/vn30_universe.json ảnh chụp danh sách VN30 tại thời điểm cập nhật
    data/processed/...                kết quả đã tính sẵn + nhật ký cập nhật
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src import config
from src.logging_config import get_logger
from src.schema import DATE_COLUMN, merge_history, standardize_ohlcv, validate_frame

logger = get_logger(__name__)


def ensure_dirs() -> None:
    for directory in (config.RAW_DIR, config.STOCK_DIR, config.PROCESSED_DIR, config.REFERENCE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def index_path(dataset: str) -> Path:
    return config.RAW_DIR / f"{dataset.lower()}.parquet"


def stock_path(symbol: str) -> Path:
    return config.STOCK_DIR / f"{symbol.upper()}.parquet"


def dataset_path(name: str, kind: str) -> Path:
    if kind == "index":
        return index_path(name)
    if kind == "stock":
        return stock_path(name)
    raise ValueError("kind phải là 'index' hoặc 'stock'")


def read_frame(path: Path) -> pd.DataFrame | None:
    """Đọc một tệp parquet đã lưu. Trả về None nếu chưa có tệp."""
    if not path.exists():
        return None
    try:
        return standardize_ohlcv(pd.read_parquet(path))
    except Exception as exc:
        # Tệp hỏng không được phép làm sập dashboard, nhưng phải để lại dấu vết
        # thay vì biến mất im lặng.
        logger.error("Không đọc được %s: %s: %s", path, type(exc).__name__, exc)
        return None


def load_index(dataset: str = config.VNINDEX_DATASET) -> pd.DataFrame | None:
    return read_frame(index_path(dataset))


def load_stock(symbol: str) -> pd.DataFrame | None:
    return read_frame(stock_path(symbol))


def load_stocks(symbols) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Đọc nhiều mã. Trả về (dữ liệu đọc được, danh sách mã thiếu tệp)."""
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in symbols:
        frame = load_stock(symbol)
        if frame is None or frame.empty:
            missing.append(symbol.upper())
        else:
            frames[symbol.upper()] = frame
    return frames, missing


def write_frame(df: pd.DataFrame, path: Path, min_rows: int = 1) -> dict:
    """Ghi parquet sau khi kiểm tra. Trả về báo cáo chất lượng."""
    ensure_dirs()
    report = validate_frame(df, min_rows=min_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return report.as_dict()


def merge_and_write(path: Path, new: pd.DataFrame, min_rows: int = 1) -> dict:
    """Hợp nhất dữ liệu mới vào tệp đang có rồi ghi lại.

    Trả về số dòng trước và sau khi hợp nhất để phục vụ báo cáo chất lượng.
    """
    old = read_frame(path)
    rows_before = 0 if old is None else int(len(old))
    combined = merge_history(old, new)
    rows_after = int(len(combined))
    report = write_frame(combined, path, min_rows=min_rows)
    report["rows_before_merge"] = rows_before
    report["rows_after_merge"] = rows_after
    report["rows_added"] = rows_after - rows_before
    return report


def last_stored_date(path: Path) -> pd.Timestamp | None:
    frame = read_frame(path)
    if frame is None or frame.empty:
        return None
    return pd.to_datetime(frame[DATE_COLUMN]).max()


def available_stock_symbols() -> list[str]:
    """Các mã đã có tệp giá trong ``data/raw/stocks``."""
    if not config.STOCK_DIR.exists():
        return []
    return sorted(p.stem.upper() for p in config.STOCK_DIR.glob("*.parquet"))


def stock_file_count() -> int:
    return len(available_stock_symbols())


def legacy_stock_files() -> list[Path]:
    """Tệp giá cổ phiếu còn sót ở bố cục cũ ``data/raw/<mã>.parquet``.

    Chỉ được tồn tại đúng một đường dẫn cho giá cổ phiếu. Kiến trúc cũ ghi thẳng
    vào ``data/raw`` nên một bản cài đặt cũ có thể còn tệp thừa ở đó; chúng phải
    hiện ra chứ không được âm thầm nằm lại.
    """
    if not config.RAW_DIR.exists():
        return []
    reserved = {config.VNINDEX_DATASET.lower(), config.VN30_INDEX_DATASET.lower()}
    return sorted(
        path for path in config.RAW_DIR.glob("*.parquet") if path.stem.lower() not in reserved
    )


# Trạng thái dữ liệu của một mã.
STOCK_OK = "ok"                    # có tệp và đủ lịch sử
STOCK_SHORT = "short_history"      # có tệp nhưng chưa đủ phiên
STOCK_MISSING = "missing_file"     # chưa có tệp


def stock_inventory(symbols: Iterable[str], min_sessions: int | None = None) -> dict[str, dict]:
    """Tình trạng từng mã: có tệp chưa, bao nhiêu phiên, đến ngày nào.

    Đây là căn cứ để phân biệt "chưa có tệp" với "có tệp nhưng thiếu lịch sử",
    hai tình huống rất khác nhau nhưng trước đây bị gộp làm một.
    """
    min_sessions = min_sessions or config.MIN_SESSIONS_FOR_FULL_HISTORY
    inventory: dict[str, dict] = {}
    for symbol in sorted({s.upper() for s in symbols}):
        frame = load_stock(symbol)
        if frame is None or frame.empty:
            inventory[symbol] = {
                "status": STOCK_MISSING, "rows": 0, "last_date": None, "first_date": None,
            }
            continue
        dates = pd.to_datetime(frame[DATE_COLUMN])
        rows = int(len(frame))
        inventory[symbol] = {
            "status": STOCK_OK if rows >= min_sessions else STOCK_SHORT,
            "rows": rows,
            "first_date": dates.min(),
            "last_date": dates.max(),
        }
    return inventory


def expected_data_files(symbols: Iterable[str]) -> list[Path]:
    """Danh sách tệp mà một bộ dữ liệu đầy đủ phải có."""
    files = [
        index_path(config.VNINDEX_DATASET),
        index_path(config.VN30_INDEX_DATASET),
        config.UNIVERSE_FILE,
        config.VNINDEX_FEATURES_FILE,
        config.VN30_SNAPSHOT_FILE,
        config.UPDATE_LOG_FILE,
    ]
    files.extend(stock_path(symbol) for symbol in sorted({s.upper() for s in symbols}))
    return files


def verify_data_files(symbols: Iterable[str]) -> dict:
    """Đối chiếu tệp kỳ vọng với tệp thực sự tồn tại trên đĩa."""
    expected = expected_data_files(symbols)
    present = [path for path in expected if path.exists()]
    absent = [path for path in expected if not path.exists()]
    return {
        "expected": expected,
        "present": present,
        "missing": absent,
        "files_expected": len(expected),
        "files_written": len(present),
        "missing_names": [p.name for p in absent],
    }


# --- JSON nhỏ (danh sách VN30, nhật ký cập nhật, ảnh chụp chỉ tiêu) -----------

def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Không đọc được %s: %s: %s", path, type(exc).__name__, exc)
        return None


def _jsonable(value):
    """Chuyển NaN/Inf và các kiểu numpy về giá trị JSON hợp lệ.

    ``json.dumps`` mặc định ghi ra ``NaN``, không phải JSON đúng chuẩn và các
    trình đọc khác sẽ từ chối tệp.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def write_json(path: Path, payload: dict) -> Path:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def data_files() -> list[Path]:
    """Danh sách tệp dữ liệu cần đồng bộ lên GitHub."""
    ensure_dirs()
    files: list[Path] = []
    for directory in (config.RAW_DIR, config.PROCESSED_DIR, config.REFERENCE_DIR):
        for pattern in ("*.parquet", "*.json"):
            files.extend(sorted(directory.rglob(pattern)))
    return sorted(set(files))
