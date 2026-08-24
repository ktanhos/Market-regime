"""Breadth của rổ VN30 **hiện tại**.

Chỉ mô tả trạng thái hiện tại của 30 mã đang thuộc VN30. Lịch sử giá 200 phiên
của một cổ phiếu chỉ dùng để tính MA200 của chính cổ phiếu đó; nó không hàm ý
rằng cổ phiếu đó đã thuộc VN30 trong suốt 200 phiên ấy.

Vì không có dữ liệu thành phần VN30 lịch sử, module này **không** dựng chuỗi
breadth lịch sử.

Thành phần tối thiểu:

* % số mã trên MA20 / MA50 / MA200
* % số mã tăng trong 1 / 5 / 20 phiên gần nhất

Mỗi thành phần đi kèm số mã hợp lệ trên tổng số mã. Không mặc định 30/30.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.schema import to_close_panel

# Bốn tình huống rất khác nhau, trước đây bị gộp thành một nhãn duy nhất.
BREADTH_NO_DATA = "CHƯA CÓ DỮ LIỆU"        # chưa có tệp giá nào
BREADTH_INSUFFICIENT = "DỮ LIỆU CHƯA ĐỦ"   # có tệp nhưng thiếu lịch sử
BREADTH_STALE = "DỮ LIỆU KHÔNG ĐỒNG BỘ"    # ngày dữ liệu giữa các mã chênh lệch lớn
BREADTH_UNKNOWN = BREADTH_NO_DATA          # tương thích ngược

DATA_OK = "ok"
DATA_NONE = "no_data"
DATA_INSUFFICIENT = "insufficient"
DATA_STALE = "stale"


def _pct_above_ma(panel: pd.DataFrame, window: int) -> dict:
    """% số mã có giá đóng cửa cuối cùng nằm trên MA ``window`` của chính nó."""
    hits = 0
    valid: list[str] = []
    for symbol in panel.columns:
        series = panel[symbol].dropna()
        if len(series) < window:
            continue
        ma = float(series.iloc[-window:].mean())
        last = float(series.iloc[-1])
        if not np.isfinite(ma) or not np.isfinite(last) or ma <= 0:
            continue
        valid.append(symbol)
        if last > ma:
            hits += 1
    total = len(valid)
    return {
        "window": window,
        "pct": float(hits / total * 100) if total else np.nan,
        "hits": hits,
        "valid": total,
        "symbols": valid,
    }


def _pct_advancing(panel: pd.DataFrame, window: int) -> dict:
    """% số mã có lợi suất dương qua ``window`` phiên."""
    hits = 0
    valid: list[str] = []
    for symbol in panel.columns:
        series = panel[symbol].dropna()
        if len(series) < window + 1:
            continue
        past = float(series.iloc[-(window + 1)])
        last = float(series.iloc[-1])
        if not np.isfinite(past) or past <= 0:
            continue
        valid.append(symbol)
        if last > past:
            hits += 1
    total = len(valid)
    return {
        "window": window,
        "pct": float(hits / total * 100) if total else np.nan,
        "hits": hits,
        "valid": total,
        "symbols": valid,
    }


def classify_breadth(score: float) -> str:
    if score is None or pd.isna(score):
        return BREADTH_NO_DATA
    for threshold, label in config.BREADTH_BANDS:
        if score < threshold:
            return label
    return config.BREADTH_TOP_LABEL


def _date_spread(panel: pd.DataFrame) -> dict:
    """Độ lệch ngày dữ liệu giữa các mã, tính theo số phiên trong bảng."""
    if panel.empty:
        return {"stale_symbols": [], "max_gap_sessions": 0}
    positions = {timestamp: i for i, timestamp in enumerate(panel.index)}
    latest = len(panel.index) - 1
    stale: list[str] = []
    worst = 0
    for symbol in panel.columns:
        series = panel[symbol].dropna()
        if series.empty:
            continue
        gap = latest - positions[series.index[-1]]
        worst = max(worst, gap)
        if gap > config.MAX_STALE_SESSIONS:
            stale.append(symbol)
    return {"stale_symbols": sorted(stale), "max_gap_sessions": int(worst)}


def data_state(loaded_symbols: int, strictest_valid: int, stale_symbols: list[str]) -> tuple[str, str]:
    """Phân biệt bốn tình huống dữ liệu. Trả về (mã trạng thái, nhãn hiển thị).

    ``strictest_valid`` là số mã đủ dữ liệu cho chỉ tiêu khắt khe nhất (MA200).
    Ba mươi tệp mỗi tệp chỉ có 30 phiên vẫn tính được MA20 nhưng không tính được
    MA200, nên đó là "chưa đủ lịch sử" chứ không phải một điểm breadth thấp.
    """
    if loaded_symbols == 0:
        return DATA_NONE, BREADTH_NO_DATA
    if strictest_valid < config.BREADTH_MIN_VALID_SYMBOLS:
        return DATA_INSUFFICIENT, BREADTH_INSUFFICIENT
    if stale_symbols:
        return DATA_STALE, BREADTH_STALE
    return DATA_OK, ""


def symbol_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Bảng chi tiết từng mã: vị trí so với MA và lợi suất các khung thời gian."""
    rows = []
    for symbol in sorted(panel.columns):
        series = panel[symbol].dropna()
        row: dict = {"Mã": symbol, "Số phiên": int(len(series))}
        if series.empty:
            rows.append(row)
            continue
        last = float(series.iloc[-1])
        row["Giá"] = round(last, 2)
        for window in config.BREADTH_MA_WINDOWS:
            key = f"Trên MA{window}"
            if len(series) >= window:
                row[key] = "Có" if last > float(series.iloc[-window:].mean()) else "Không"
            else:
                row[key] = "Thiếu dữ liệu"
        for window in config.BREADTH_RETURN_WINDOWS:
            key = f"Lợi suất {window} phiên %"
            if len(series) >= window + 1:
                past = float(series.iloc[-(window + 1)])
                row[key] = round((last / past - 1) * 100, 2) if past > 0 else np.nan
            else:
                row[key] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_breadth(frames: dict[str, pd.DataFrame], universe: list[str]) -> dict:
    """Tính toàn bộ chỉ tiêu breadth cho rổ VN30 hiện tại."""
    universe = sorted({s.upper() for s in universe})
    panel = to_close_panel(frames)

    result: dict = {
        "universe_size": len(universe),
        "loaded_symbols": int(panel.shape[1]) if not panel.empty else 0,
        "missing_symbols": [s for s in universe if s not in panel.columns],
        "components": {},
        "score": np.nan,
        "state": BREADTH_NO_DATA,
        "data_state": DATA_NONE,
        "stale_symbols": [],
        "max_gap_sessions": 0,
        "valid_symbols": 0,
        "min_valid_symbols": 0,
        "max_valid_symbols": 0,
        "as_of": None,
        "table": pd.DataFrame(),
        "sufficient": False,
        "panel": panel,
    }
    if panel.empty:
        return result

    result["as_of"] = pd.Timestamp(panel.index.max())
    components: dict[str, dict] = {}
    for window in config.BREADTH_MA_WINDOWS:
        components[f"ma{window}"] = _pct_above_ma(panel, window)
    for window in config.BREADTH_RETURN_WINDOWS:
        components[f"ret{window}"] = _pct_advancing(panel, window)
    result["components"] = components

    # Số mã hợp lệ hiển thị cho người dùng: số mã đủ dữ liệu cho chỉ tiêu
    # khắt khe nhất đang được tính (MA dài nhất).
    strictest = components.get(f"ma{max(config.BREADTH_MA_WINDOWS)}", {})
    result["valid_symbols"] = int(strictest.get("valid", 0))
    result["min_valid_symbols"] = int(min((c["valid"] for c in components.values()), default=0))
    result["max_valid_symbols"] = int(max((c["valid"] for c in components.values()), default=0))
    spread = _date_spread(panel)
    result.update(spread)
    state_code, state_label = data_state(
        result["loaded_symbols"], result["min_valid_symbols"], spread["stale_symbols"]
    )
    result["data_state"] = state_code
    # Dữ liệu lệch ngày vẫn tính được số liệu, chỉ cần cảnh báo kèm theo.
    result["sufficient"] = state_code in (DATA_OK, DATA_STALE)

    values = [c["pct"] for c in components.values() if pd.notna(c["pct"])]
    if values:
        result["score"] = float(np.mean(values))
    result["state"] = classify_breadth(result["score"]) if state_code == DATA_OK else state_label

    result["table"] = symbol_table(panel)
    result["panel"] = panel
    return result
