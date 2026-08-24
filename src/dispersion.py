"""Phân hóa: độ khác biệt lợi suất giữa các cổ phiếu VN30 hiện tại.

Đo bằng độ lệch chuẩn theo lát cắt ngang (cross-sectional) của lợi suất 1/5/20
phiên. Khung 20 phiên là chỉ báo chính.

Không gắn nhãn "cao / thấp / nguy hiểm" bằng ngưỡng tùy ý. Nhãn được suy ra từ
phân vị của chính chuỗi quan sát được.

Cảnh báo phương pháp luận: chuỗi tham chiếu được tính trên **rổ VN30 hiện tại**
kéo ngược về quá khứ, không phải rổ VN30 thực tế của từng ngày trong quá khứ.
Vì vậy phân vị ở đây chỉ là bối cảnh mô tả, không phải thống kê lịch sử của chỉ
số VN30. Trường ``historical_basis`` luôn ghi rõ điều này.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

DISPERSION_UNKNOWN = "CHƯA ĐỦ DỮ LIỆU"
DISPERSION_LOW = "PHÂN HÓA THẤP"
DISPERSION_NORMAL = "PHÂN HÓA TRUNG BÌNH"
DISPERSION_HIGH = "PHÂN HÓA CAO"

HISTORICAL_BASIS_NOTE = (
    "Phân vị tính trên rổ VN30 hiện tại kéo ngược về quá khứ, "
    "không phải thành phần VN30 thực tế của từng ngày. Chỉ dùng để mô tả bối cảnh."
)


def cross_sectional_dispersion(panel: pd.DataFrame, window: int, min_symbols: int = 10) -> pd.Series:
    """Độ lệch chuẩn theo lát cắt ngang của lợi suất ``window`` phiên."""
    if panel.empty:
        return pd.Series(dtype="float64")
    # fill_method=None: một mã nghỉ giao dịch phải cho NaN, không được pad giá cũ
    # thành lợi suất 0%. Pad làm độ lệch chuẩn theo lát cắt ngang thấp đi giả tạo.
    returns = panel.pct_change(window, fill_method=None)
    counts = returns.notna().sum(axis=1)
    dispersion = returns.std(axis=1, ddof=1) * 100
    return dispersion.where(counts >= min_symbols).rename(f"dispersion_{window}d")


def percentile_of_last(series: pd.Series, lookback: int | None = None) -> float:
    """Phân vị của giá trị cuối cùng trong ``lookback`` quan sát gần nhất."""
    lookback = lookback or config.DISPERSION_CONTEXT_SESSIONS
    clean = series.dropna()
    if len(clean) < 2:
        return np.nan
    window = clean.iloc[-lookback:]
    last = float(window.iloc[-1])
    return float((window <= last).mean() * 100)


def classify_context(percentile: float) -> str:
    if percentile is None or pd.isna(percentile):
        return DISPERSION_UNKNOWN
    if percentile >= config.CONTEXT_HIGH_PERCENTILE:
        return DISPERSION_HIGH
    if percentile <= config.CONTEXT_LOW_PERCENTILE:
        return DISPERSION_LOW
    return DISPERSION_NORMAL


def compute_dispersion(panel: pd.DataFrame) -> dict:
    result: dict = {
        "primary_window": config.DISPERSION_PRIMARY_WINDOW,
        "windows": {},
        "value": np.nan,
        "percentile": np.nan,
        "state": DISPERSION_UNKNOWN,
        "historical_basis": HISTORICAL_BASIS_NOTE,
        "context_sessions": 0,
        "series": pd.Series(dtype="float64"),
    }
    if panel is None or panel.empty:
        return result

    for window in config.BREADTH_RETURN_WINDOWS:
        series = cross_sectional_dispersion(panel, window)
        clean = series.dropna()
        result["windows"][window] = {
            "value": float(clean.iloc[-1]) if len(clean) else np.nan,
            "percentile": percentile_of_last(series),
            "observations": int(len(clean)),
        }

    primary = cross_sectional_dispersion(panel, config.DISPERSION_PRIMARY_WINDOW)
    clean = primary.dropna()
    if len(clean):
        result["value"] = float(clean.iloc[-1])
        result["percentile"] = percentile_of_last(primary)
        result["state"] = classify_context(result["percentile"])
        result["context_sessions"] = int(min(len(clean), config.DISPERSION_CONTEXT_SESSIONS))
        result["series"] = primary.dropna().iloc[-config.DISPERSION_CONTEXT_SESSIONS:]
    return result
