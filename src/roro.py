"""Trend: sức mạnh động lượng đa khung và chỉ báo RORO của VNINDEX.

Công thức giữ nguyên thiết kế gốc::

    Strength = ROC63*0.4 + ROC126*0.2 + ROC189*0.2 + ROC252*0.2   (đơn vị %)
    RORO     = Strength - trung bình động 49 phiên của Strength

RORO là thước đo xu hướng **tương đối** so với chính nó trong 49 phiên gần nhất.
RORO > 0 không có nghĩa là "Risk On" tuyệt đối, nên phân loại dùng ba mức với
vùng trung tính rộng bằng ``RORO_NEUTRAL_SIGMA`` lần độ lệch chuẩn 252 phiên
của chính chuỗi RORO. Ngưỡng vì thế tự điều chỉnh theo biên độ dữ liệu quan sát
được thay vì là một con số cố định.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

TREND_POSITIVE = "TÍCH CỰC"
TREND_NEUTRAL = "TRUNG TÍNH"
TREND_WEAK = "SUY YẾU"
TREND_UNKNOWN = "CHƯA ĐỦ DỮ LIỆU"


def calculate_strength(close: pd.Series) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    total = None
    for horizon, weight in config.RORO_HORIZONS:
        part = close.pct_change(horizon) * weight
        total = part if total is None else total + part
    return (total * 100).rename("strength")


def calculate_roro(close: pd.Series, baseline_window: int | None = None) -> pd.DataFrame:
    baseline_window = baseline_window or config.RORO_BASELINE_WINDOW
    strength = calculate_strength(close)
    baseline = strength.rolling(baseline_window, min_periods=baseline_window).mean()
    roro = (strength - baseline).rename("roro")
    band = (
        roro.rolling(config.RORO_SIGMA_WINDOW, min_periods=60).std()
        * config.RORO_NEUTRAL_SIGMA
    ).rename("roro_band")
    return pd.DataFrame({"strength": strength, "roro_baseline": baseline, "roro": roro, "roro_band": band})


def classify_roro(roro: float, band: float | None = None) -> str:
    """Ba mức: TÍCH CỰC / TRUNG TÍNH / SUY YẾU."""
    if roro is None or pd.isna(roro):
        return TREND_UNKNOWN
    if band is None or pd.isna(band) or band <= 0:
        band = 0.0
    if roro > band:
        return TREND_POSITIVE
    if roro < -band:
        return TREND_WEAK
    return TREND_NEUTRAL


def trend_frame(close: pd.Series) -> pd.DataFrame:
    frame = calculate_roro(close)
    frame["trend_state"] = [
        classify_roro(r, b) for r, b in zip(frame["roro"], frame["roro_band"])
    ]
    return frame


def trend_snapshot(close: pd.Series) -> dict:
    frame = trend_frame(close)
    if frame.empty:
        return {"state": TREND_UNKNOWN}
    last = frame.iloc[-1]
    roro = float(last["roro"]) if pd.notna(last["roro"]) else np.nan
    band = float(last["roro_band"]) if pd.notna(last["roro_band"]) else np.nan
    return {
        "state": last["trend_state"],
        "roro": roro,
        "band": band,
        "strength": float(last["strength"]) if pd.notna(last["strength"]) else np.nan,
        "series": frame,
    }
