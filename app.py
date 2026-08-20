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
    start_ts, end_ts = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
    pieces, cursor = [], start_ts
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=119), end_ts)
        df = market.index(symbol).ohlcv(start=cursor.strftime("%Y-%m-%d"), end=chunk_end.strftime("%Y-%m-%d"), interval="1D")
        pieces.append(normalize_ohlcv(df))
        cursor = chunk_end + pd.Timedelta(days=1)
    return pd.concat(pieces, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def calculate_features(df, level=49, park_window=22, stress_window=252):
    out = df.copy()
    close = out["close"]
    strength = (close.pct_change(63) * .4 + close.pct_change(126) * .2 + close.pct_change(189) * .2 + close.pct_change(252) * .2) * 100
    out["strength"] = strength
    out["roro_equal"] = strength.rolling(level, min_periods=level).mean()
    out["roro"] = strength - out["roro_equal"]
    out["trend_state"] = np.select([out["roro"] > 0, out["roro"] < 0], ["RISK ON", "RISK OFF"], default="TRUNG TÍNH")
    out["trend_persistence"] = (out["roro"] > 0).astype(int).groupby((out["roro"] > 0).ne((out["roro"] > 0).shift()).cumsum()).cumsum()
    out.loc[out["roro"] <= 0, "trend_persistence"] = (out["roro"] < 0).astype(int).groupby((out["roro"] < 0).ne((out["roro"] < 0).shift()).cumsum()).cumsum()

    if {"high", "low"}.issubset(out.columns):
        log_range = np.log(out["high"] / out["low"])
        out["parkinson_vol"] = np.sqrt(log_range.pow(2).rolling(park_window, min_periods=park_window).mean() / (4 * np.log(2)) * 252) * 100
        out["stress_base"] = out["parkinson_vol"].rolling(stress_window, min_periods=max(60, park_window)).mean()
        out["stress_std"] = out["parkinson_vol"].rolling(stress_window, min_periods=max(60, park_window)).std()
        out["stress_z"] = (out["parkinson_vol"] - out["stress_base"]) / out["stress_std"].replace(0, np.nan)
        out["stress_score"] = 50 + 10 * out["stress_z"].clip(-5, 5)
        out["stress_state"] = np.select([out["stress_score"] >= 65, out["stress_score"] >= 55, out["stress_score"] <= 40], ["CAO", "TĂNG", "THẤP"], default="BÌNH THƯỜNG")
    else:
        for col in ["parkinson_vol", "stress_base", "stress_std", "stress_z", "stress_score"]:
            out[col] = np.nan
        out["stress_state"] = "CHƯA ĐỦ DỮ LIỆU"

    return out


def classify_regime(row):
    if pd.isna(row["roro"]) or pd.isna(row["stress_score"]):
        return "CHƯA ĐỦ DỮ LIỆU", "Chưa đủ lịch sử để phân loại"
    if row["roro"] > 0 and row["stress_score"] < 55:
        return "EXPANSION", "Xu hướng thuận lợi, căng thẳng thấp"
    if row["roro"] > 0 and row["stress_score"] >= 55:
        return "FRAGILE RALLY", "Xu hướng tăng nhưng căng thẳng cao"
    if row["roro"] <= 0 and row["stress_score"] >= 65:
        return "HIGH STRESS RISK OFF", "Xu hướng bất lợi và căng thẳng cao"
    if row["roro"] <= 0 and row["stress_score"] < 55:
        return "CORRECTION", "Xu hướng bất lợi nhưng chưa ở trạng thái căng thẳng cực cao"
    return "TRANSITION", "Thị trường đang chuyển trạng thái"


def latest_text(value, fmt):
    return "Chưa đủ dữ liệu" if pd.isna(value) else fmt.format(value)


def show_market(name, df):
    latest = df.iloc[-1]
    regime, description = classify_regime(latest)
    previous = df.iloc[-2] if len(df) > 1 else latest
    st.subheader(name)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Đóng cửa", latest_text(latest["close"], "{:,.2f}"), latest_text(latest["close"] - previous["close"], "{:+,.2f}"))
    c2.metric("RORO", latest_text(latest["roro"], "{:.2f}"))
    c3.metric("Xu hướng", latest["trend_state"])
    c4.metric("Căng thẳng", latest_text(latest["stress_score"], "{:.1f}"))
    c5.metric("Regime sơ bộ", regime)
    st.caption(description)

    chart = df.set_index("date")
    tabs = st.tabs(["Tổng quan", "Trend", "Stress", "Kiểm định"])
    with tabs[0]:
        st.line_chart(chart[["close"]].dropna(), use_container_width=True)
        st.dataframe(df.tail(10)[["date", "close", "roro", "trend_state", "parkinson_vol", "stress_score", "stress_state"]], use_container_width=True, hide_index=True)
    with tabs[1]:
        st.line_chart(chart[["roro"]].dropna(), use_container_width=True)
        st.caption("RORO dương là Risk On. RORO âm là Risk Off. Cần đọc cùng Stress để đánh giá chất lượng của trạng thái.")
        st.metric("Số phiên trạng thái hiện tại", int(latest["trend_persistence"]) if not pd.isna(latest["trend_persistence"]) else 0)
    with tabs[2]:
        st.line_chart(chart[["parkinson_vol", "stress_score"]].dropna(how="all"), use_container_width=True)
        st.caption("Điểm Stress hiện được chuẩn hóa quanh mức lịch sử của chính thị trường, chưa phải thước đo xác suất thua lỗ.")
    with tabs[3]:
        test = df.copy()
        for horizon in [5, 10, 20]:
            test[f"forward_{horizon}"] = test["close"].shift(-horizon) / test["close"] - 1
        rows = []
        for regime_name in ["EXPANSION", "FRAGILE RALLY", "CORRECTION", "HIGH STRESS RISK OFF", "TRANSITION"]:
            mask = test.apply(lambda x: classify_regime(x)[0] == regime_name, axis=1)
            item = {"Regime": regime_name, "Số quan sát": int(mask.sum())}
            for horizon in [5, 10, 20]:
                values = test.loc[mask, f"forward_{horizon}"].dropna()
                item[f"T{horizon}"] = values.mean() * 100 if len(values) else np.nan
                item[f"Tỷ lệ dương T{horizon}"] = (values > 0).mean() * 100 if len(values) else np.nan
            rows.append(item)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Đây là kiểm định mô tả trên cùng tập dữ liệu, chưa phải kiểm định ngoài mẫu hay bằng chứng dự báo.")


st.title("Vietnam Market Regime")
st.caption("Khung phân tích chế độ thị trường Việt Nam")
st.write("Nhập API key, lấy dữ liệu trong phiên, tính Trend và Stress, sau đó kiểm tra đặc điểm lợi suất tương lai theo từng trạng thái.")

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
            market, results = Market(), {}
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
    st.header("Lộ trình mô hình")
    st.write("Trend và Stress đã có thể quan sát và kiểm định mô tả. Breadth sẽ được xây ở lớp tiếp theo bằng dữ liệu cổ phiếu toàn thị trường. Regime hiện tại chỉ là phân loại nghiên cứu sơ bộ từ hai lớp đầu vào, chưa phải tín hiệu giao dịch.")
else:
    st.info("Nhập API key VNStock ở thanh bên, chọn khoảng thời gian và bấm Lấy dữ liệu và phân tích.")
