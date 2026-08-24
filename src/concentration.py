"""Tập trung rủi ro của rổ VN30 hiện tại.

Đây là **proxy về mức tập trung rủi ro biến động**, không phải đóng góp rủi ro
thực của một danh mục. Không có trọng số danh mục nên không thể nói đây là rủi
ro của nhà đầu tư nào cả. Vốn hóa cũng không được dùng vì repository chưa có
nguồn dữ liệu vốn hóa đáng tin cậy.

Cách tính:

1. Biến động 20 phiên đã quy năm của từng mã.
2. Chuẩn hóa thành tỷ trọng ``w_i = vol_i / sum(vol)`` -> risk contribution proxy.
3. Top 5 / Top 10 risk share, chỉ số Herfindahl và số mã đóng góp hiệu dụng
   ``1 / HHI``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.dispersion import HISTORICAL_BASIS_NOTE, classify_context, percentile_of_last

CONCENTRATION_UNKNOWN = "CHƯA ĐỦ DỮ LIỆU"
CONCENTRATION_LOW = "TẬP TRUNG THẤP"
CONCENTRATION_NORMAL = "TẬP TRUNG TRUNG BÌNH"
CONCENTRATION_HIGH = "TẬP TRUNG CAO"

_STATE_MAP = {
    "PHÂN HÓA CAO": CONCENTRATION_HIGH,
    "PHÂN HÓA THẤP": CONCENTRATION_LOW,
    "PHÂN HÓA TRUNG BÌNH": CONCENTRATION_NORMAL,
    "CHƯA ĐỦ DỮ LIỆU": CONCENTRATION_UNKNOWN,
}


def realised_volatility(panel: pd.DataFrame, window: int | None = None) -> pd.DataFrame:
    """Biến động quy năm theo cửa sổ trượt cho từng mã, đơn vị phần trăm."""
    window = window or config.VOLATILITY_WINDOW
    # fill_method=None: phiên thiếu dữ liệu phải là NaN chứ không phải lợi suất 0%,
    # nếu không biến động của mã đó bị ước lượng thấp đi.
    returns = panel.pct_change(fill_method=None)
    return returns.rolling(window, min_periods=window).std() * np.sqrt(config.ANNUALIZATION) * 100


def risk_weights(vol_row: pd.Series) -> pd.Series:
    clean = vol_row.dropna()
    clean = clean[clean > 0]
    total = float(clean.sum())
    if total <= 0:
        return pd.Series(dtype="float64")
    return (clean / total).sort_values(ascending=False)


def concentration_metrics(weights: pd.Series) -> dict:
    if weights.empty:
        return {
            "hhi": np.nan,
            "effective_names": np.nan,
            "top_shares": {},
            "contributors": 0,
        }
    hhi = float((weights ** 2).sum())
    return {
        "hhi": hhi,
        "effective_names": float(1 / hhi) if hhi > 0 else np.nan,
        "top_shares": {
            n: float(weights.nlargest(n).sum() * 100) for n in config.CONCENTRATION_TOP_N
        },
        "contributors": int(len(weights)),
    }


def _top_share_series(vol_frame: pd.DataFrame, top_n: int, min_symbols: int = 10) -> pd.Series:
    rows = {}
    for timestamp, row in vol_frame.iterrows():
        weights = risk_weights(row)
        if len(weights) < min_symbols:
            continue
        rows[timestamp] = float(weights.nlargest(top_n).sum() * 100)
    return pd.Series(rows, dtype="float64").sort_index()


def compute_concentration(panel: pd.DataFrame) -> dict:
    result: dict = {
        "window": config.VOLATILITY_WINDOW,
        "hhi": np.nan,
        "effective_names": np.nan,
        "top_shares": {},
        "contributors": 0,
        "percentile": np.nan,
        "state": CONCENTRATION_UNKNOWN,
        "historical_basis": HISTORICAL_BASIS_NOTE,
        "table": pd.DataFrame(),
        "series": pd.Series(dtype="float64"),
        "proxy_note": (
            "Tỷ trọng rủi ro suy ra từ biến động 20 phiên của từng mã. "
            "Đây là proxy về mức tập trung, không phải đóng góp rủi ro của một danh mục cụ thể."
        ),
    }
    if panel is None or panel.empty:
        return result

    vol_frame = realised_volatility(panel).dropna(how="all")
    if vol_frame.empty:
        return result

    weights = risk_weights(vol_frame.iloc[-1])
    if weights.empty:
        return result

    result.update(concentration_metrics(weights))

    top_n = config.CONCENTRATION_TOP_N[0]
    series = _top_share_series(vol_frame.iloc[-config.DISPERSION_CONTEXT_SESSIONS:], top_n)
    if len(series) > 1:
        result["series"] = series
        result["percentile"] = percentile_of_last(series)
        result["state"] = _STATE_MAP.get(classify_context(result["percentile"]), CONCENTRATION_UNKNOWN)

    last_vol = vol_frame.iloc[-1]
    result["table"] = (
        pd.DataFrame(
            {
                "Mã": weights.index,
                "Biến động 20 phiên %": [round(float(last_vol[s]), 2) for s in weights.index],
                "Tỷ trọng rủi ro %": (weights * 100).round(2).values,
            }
        )
        .sort_values("Tỷ trọng rủi ro %", ascending=False)
        .reset_index(drop=True)
    )
    return result
