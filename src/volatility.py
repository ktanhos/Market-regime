"""Stress: biến động Parkinson và chế độ căng thẳng của VNINDEX.

Ước lượng Parkinson dùng **trung bình của bình phương log biên độ**, không phải
phương sai quanh trung bình. Công thức chính xác đang dùng::

    vol = sqrt( mean_{22 phiên}( ln(H/L)^2 ) / (4 * ln 2) * 252 ) * 100

Chỉ một công thức duy nhất được dùng trong toàn bộ dự án. Không trộn giữa
"variance" và "mean squared range".

Chế độ căng thẳng không dùng ngưỡng tuyệt đối kiểu 20/25/35. Nó dựa trên phân vị
của chính chuỗi biến động VNINDEX trong 252 phiên gần nhất, tức là so sánh mức
biến động hiện tại với chính thị trường này trong một năm qua.

``stress_index`` là chỉ số căng thẳng biến động dạng proxy 0-100. Đây KHÔNG phải
VIX. Không có dữ liệu phái sinh (basis VN30F1M) trong mô hình vì nguồn dữ liệu
phái sinh chưa đủ ổn định để đưa vào, và không có xác suất nào được gán cho nó.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

STRESS_UNKNOWN = "CHƯA ĐỦ DỮ LIỆU"


def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int | None = None,
    annualization: int | None = None,
) -> pd.Series:
    """Biến động Parkinson đã quy năm, đơn vị phần trăm."""
    window = window or config.PARKINSON_WINDOW
    annualization = annualization or config.ANNUALIZATION
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    ratio = (high / low).where((high > 0) & (low > 0))
    log_range = np.log(ratio)
    mean_squared_range = log_range.pow(2).rolling(window, min_periods=window).mean()
    vol = np.sqrt(mean_squared_range / (4 * np.log(2)) * annualization) * 100
    return vol.rename("parkinson_vol")


def close_to_close_volatility(close: pd.Series, window: int | None = None) -> pd.Series:
    window = window or config.PARKINSON_WINDOW
    returns = pd.to_numeric(close, errors="coerce").pct_change()
    return (
        returns.rolling(window, min_periods=window).std()
        * np.sqrt(config.ANNUALIZATION)
        * 100
    ).rename("close_vol")


def rolling_percentile(series: pd.Series, window: int | None = None, min_periods: int | None = None) -> pd.Series:
    """Thứ hạng phân vị của giá trị hiện tại trong cửa sổ trượt, thang 0-100."""
    window = window or config.STRESS_PERCENTILE_WINDOW
    min_periods = min_periods or config.STRESS_MIN_PERIODS
    return (
        series.rolling(window, min_periods=min_periods)
        .rank(pct=True)
        .mul(100)
        .rename("percentile")
    )


def rolling_zscore(series: pd.Series, window: int | None = None, min_periods: int | None = None) -> pd.Series:
    window = window or config.STRESS_PERCENTILE_WINDOW
    min_periods = min_periods or config.STRESS_MIN_PERIODS
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std().replace(0, np.nan)
    return ((series - mean) / std).rename("zscore")


def classify_stress(percentile: float) -> str:
    if percentile is None or pd.isna(percentile):
        return STRESS_UNKNOWN
    for threshold, label in config.STRESS_BANDS:
        if percentile < threshold:
            return label
    return config.STRESS_TOP_LABEL


def stress_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Tính biến động và chế độ căng thẳng từ khung OHLC đã chuẩn hóa."""
    vol = parkinson_volatility(df["high"], df["low"])
    smoothed = vol.ewm(span=config.STRESS_EMA_SPAN, min_periods=config.STRESS_EMA_SPAN).mean()
    percentile = rolling_percentile(smoothed)
    out = pd.DataFrame(
        {
            "parkinson_vol": vol,
            "stress_index": smoothed.rename("stress_index"),
            "stress_percentile": percentile,
            "stress_zscore": rolling_zscore(smoothed),
            "close_vol": close_to_close_volatility(df["close"]),
        }
    )
    out["stress_state"] = [classify_stress(p) for p in out["stress_percentile"]]
    return out


def stress_snapshot(df: pd.DataFrame) -> dict:
    frame = stress_frame(df)
    if frame.empty:
        return {"state": STRESS_UNKNOWN}
    last = frame.iloc[-1]

    def value(name):
        v = last.get(name)
        return float(v) if v is not None and pd.notna(v) else np.nan

    return {
        "state": last["stress_state"],
        "parkinson_vol": value("parkinson_vol"),
        "stress_index": value("stress_index"),
        "percentile": value("stress_percentile"),
        "zscore": value("stress_zscore"),
        "close_vol": value("close_vol"),
        "label": "Chỉ số căng thẳng biến động (proxy)",
        "series": frame,
    }
