from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from vnstock import Market, register_user

st.set_page_config(page_title="Vietnam Market Regime", page_icon="📊", layout="wide")


def normalize_ohlcv(df):
    if df is None or df.empty:
        raise ValueError("API trả về dữ liệu rỗng")
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in out.columns]
    date_col = "time" if "time" in out.columns else "date" if "date" in out.columns else None
    if date_col is None or "close" not in out.columns:
        raise ValueError("Dữ liệu phải có cột thời gian và close")
    out = out.rename(columns={date_col: "date"})
    for col in ["open", "high", "low", "close", "volume"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def fetch_index_history(market, symbol, start_date, end_date):
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    pieces = []
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=119), end_ts)
        df = market.index(symbol).ohlcv(
            start=cursor.strftime("%Y-%m-%d"),
            end=chunk_end.strftime("%Y-%m-%d"),
            interval="1D",
        )
        pieces.append(normalize_ohlcv(df))
        cursor = chunk_end + pd.Timedelta(days=1)
    out = pd.concat(pieces, ignore_index=True)
    return out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def calculate_features(df, level=49, park_window=22, stress_window=252):
    out = df.copy()
    close = out["close"]
    strength = (
        close.pct_change(63) * 0.4
        + close.pct_change(126) * 0.2
        + close.pct_change(189) * 0.2
        + close.pct_change(252) * 0.2
    ) * 100
    out["strength"] = strength
    out["roro_equal"] = strength.rolling(level, min_periods=level).mean()
    out["roro"] = strength - out["roro_equal"]
    out["trend_state"] = np.where(out["roro"] > 0, "RISK ON", np.where(out["roro"] < 0, "RISK OFF", "TRUNG TÍNH"))

    if {"high", "low"}.issubset(out.columns):
        log_range = np.log(out["high"] / out["low"])
        out["parkinson_vol"] = np.sqrt(log_range.pow(2).rolling(park_window, min_periods=park_window).mean() / (4 * np.log(2)) * 252) * 100
        base = out["parkinson_vol"].rolling(stress_window, min_periods=max(60, park_window)).mean()
        std = out["parkinson_vol"].rolling(stress_window, min_periods=max(60, park_window)).std()
        out["stress_z"] = (out["parkinson_vol"] - base) / std.replace(0, np.nan)
        out["stress_score"] = 50 + 10 * out["stress_z"].clip(-5, 5)
    else:
        out["parkinson_vol"] = np.nan
        out["stress_z"] = np.nan
        out["stress_score"] = np.nan
    return out


def latest_text(value, fmt):
    return "Chưa đủ dữ liệu" if pd.isna(value) else fmt.format(value)


def show_market(name, df):
    latest = df.iloc[-1]
    st.subheader(name)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Đóng cửa", latest_text(latest["close"], "{:,.2f}"))
    c2.metric("RORO", latest_text(latest["roro"], "{:.2f}"))
    c3.metric("Xu hướng", latest["trend_state"])
    c4.metric("Điểm căng thẳng", latest_text(latest["stress_score"], "{:.1f}"))

    chart = df.set_index("date")[["close", "roro", "parkinson_vol", "stress_score"]]
    tabs = st.tabs(["Giá", "Trend", "Stress"])
    with tabs[0]:
        st.line_chart(chart[["close"]].dropna(), use_container_width=True)
    with tabs[1]:
        st.line_chart(chart[["roro"]].dropna(), use_container_width=True)
        st.caption("RORO dương thể hiện môi trường Risk On, RORO âm thể hiện môi trường Risk Off.")
    with tabs[2]:
        st.line_chart(chart[["parkinson_vol", "stress_score"]].dropna(how="all"), use_container_width=True)


st.title("Vietnam Market Regime")
st.caption("Khung phân tích chế độ thị trường Việt Nam")
st.write("Phiên bản nghiên cứu: nhập API key, lấy dữ liệu trong phiên và tính trực tiếp. API key không được hiển thị hoặc lưu vào mã nguồn.")

with st.sidebar:
    st.header("Cấu hình dữ liệu")
    api_key = st.text_input("API key VNStock", type="password")
    start_date = st.date_input("Từ ngày", value=date(2015, 1, 1))
    end_date = st.date_input("Đến ngày", value=date.today())
    run = st.button("Lấy dữ liệu và phân tích", type="primary", use_container_width=True)

if run:
    if start_date >= end_date:
        st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        st.stop()
    if not api_key.strip():
        st.error("Vui lòng nhập API key VNStock.")
        st.stop()

    try:
        with st.status("Đang kết nối và lấy dữ liệu...", expanded=True) as status:
            register_user(api_key=api_key.strip())
            market = Market()
            results = {}
            for symbol in ["VNINDEX", "VN30"]:
                st.write(f"Đang lấy dữ liệu {symbol}")
                raw = fetch_index_history(market, symbol, start_date, end_date)
                results[symbol] = calculate_features(raw)
                st.write(f"{symbol}: {len(raw):,} phiên")
            status.update(label="Hoàn tất phân tích", state="complete")

        st.session_state["market_regime_results"] = results
        st.success("Đã lấy dữ liệu và tính toán thành công.")
    except Exception as exc:
        st.error(f"Không thể hoàn tất quá trình lấy dữ liệu: {type(exc).__name__}: {exc}")
        st.stop()

results = st.session_state.get("market_regime_results")
if results:
    for name in ["VNINDEX", "VN30"]:
        if name in results:
            show_market(name, results[name])

    st.divider()
    st.header("Các lớp mô hình")
    for title, description, status in [
        ("Trend", "RORO và động lượng đa kỳ hạn", "Đang tính"),
        ("Stress", "Parkinson Volatility và điểm căng thẳng", "Đang tính"),
        ("Breadth", "Độ rộng toàn thị trường", "Chưa tích hợp"),
        ("Regime", "Tổng hợp trạng thái", "Đang ở giai đoạn thiết kế"),
    ]:
        st.subheader(title)
        st.write(description)
        st.caption(status)
else:
    st.info("Nhập API key VNStock ở thanh bên, chọn khoảng thời gian và bấm Lấy dữ liệu và phân tích.")
