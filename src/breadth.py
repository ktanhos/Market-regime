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

BREADTH_UNKNOWN = "CHƯA ĐỦ DỮ LIỆU"


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
        return BREADTH_UNKNOWN
    for threshold, label in config.BREADTH_BANDS:
        if score < threshold:
            return label
    return config.BREADTH_TOP_LABEL


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
        "state": BREADTH_UNKNOWN,
        "as_of": None,
        "table": pd.DataFrame(),
        "sufficient": False,
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

    values = [c["pct"] for c in components.values() if pd.notna(c["pct"])]
    if values:
        result["score"] = float(np.mean(values))
        result["state"] = classify_breadth(result["score"])

    # Số mã hợp lệ hiển thị cho người dùng: số mã đủ dữ liệu cho chỉ tiêu
    # khắt khe nhất đang được tính (MA dài nhất).
    strictest = components.get(f"ma{max(config.BREADTH_MA_WINDOWS)}", {})
    result["valid_symbols"] = int(strictest.get("valid", 0))
    result["min_valid_symbols"] = int(min((c["valid"] for c in components.values()), default=0))
    result["max_valid_symbols"] = int(max((c["valid"] for c in components.values()), default=0))
    result["sufficient"] = result["max_valid_symbols"] >= config.BREADTH_MIN_VALID_SYMBOLS
    result["table"] = symbol_table(panel)
    result["panel"] = panel
    return result
