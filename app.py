from datetime import date
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

from vnstock import register_user
from vnstock.api.quote import Quote

st.set_page_config(page_title="Vietnam Market Regime", page_icon="📊", layout="wide")

VN30_FALLBACK = ["ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG","MBB","MSN","MWG","PLX","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VRE","VIX"]
REQUEST_INTERVAL_SECONDS = 1.1


def clean(raw, symbol):
    if raw is None or raw.empty:
        raise ValueError(f"Không có dữ liệu cho {symbol}")
    df = raw.copy()
    df.rename(columns={"date": "time", "Date": "time", "Time": "time"}, inplace=True)
    required = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột {missing} cho {symbol}")
    df = df[required].copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["time", "close"], inplace=True)
    df.sort_values("time", inplace=True)
    df.drop_duplicates("time", keep="last", inplace=True)
    return df


def configure_vnstock(api_key):
    key = (api_key or os.getenv("VNSTOCK_API_KEY", "")).strip()
    if not key:
        raise ValueError("Thiếu VNStock API key")
    os.environ["VNSTOCK_API_KEY"] = key
    register_user(api_key=key)


def rate_limit_gate():
    previous = st.session_state.get("last_vnstock_request", 0.0)
    wait = REQUEST_INTERVAL_SECONDS - (time.monotonic() - previous)
    if wait > 0:
        time.sleep(wait)
    st.session_state["last_vnstock_request"] = time.monotonic()


def safe_error(exc):
    key = os.getenv("VNSTOCK_API_KEY", "")
    return str(exc).replace(key, "[API_KEY]")[:500]


def fetch_equity(symbol, start, end):
    errors = []
    for source in ("KBS", "VCI"):
        rate_limit_gate()
        try:
            quote = Quote(symbol=symbol, source=source)
            raw = quote.history(start=start, end=end, interval="1D")
            return clean(raw, symbol)
        except Exception as exc:
            errors.append(f"{source}: {safe_error(exc)}")
    raise RuntimeError(f"VNStock không lấy được {symbol}: " + " | ".join(errors))


def fetch_index(start, end):
    errors = []
    for source in ("KBS", "VCI"):
        rate_limit_gate()
        try:
            quote = Quote(symbol="VNINDEX", source=source)
            raw = quote.history(start=start, end=end, interval="1D")
            return clean(raw, "VNINDEX")
        except Exception as exc:
            errors.append(f"{source}: {safe_error(exc)}")
    raise RuntimeError("VNStock không lấy được VNINDEX: " + " | ".join(errors))


def get_vn30_symbols():
    try:
        from vnstock import Reference
        listing = Reference().equity.list_by_group("VN30")
        cols = {str(c).lower(): c for c in listing.columns}
        symbol_col = cols.get("symbol") or cols.get("ticker") or cols.get("code")
        if symbol_col:
            symbols = sorted(listing[symbol_col].dropna().astype(str).str.upper().unique().tolist())
            if len(symbols) >= 20:
                return symbols[:30], "Danh sách VN30 hiện tại"
    except Exception:
        pass
    return VN30_FALLBACK, "Danh sách VN30 dự phòng"


def build_recent_breadth(api_key, end_date):
    configure_vnstock(api_key)
    symbols, source = get_vn30_symbols()
    start = (pd.Timestamp(end_date) - pd.DateOffset(days=450)).strftime("%Y-%m-%d")
    end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    prices = {}
    failed = []
    progress = st.progress(0, text="Đang lấy dữ liệu Breadth hiện tại")
    status = st.empty()
    for i, symbol in enumerate(symbols, 1):
        status.caption(f"Breadth hiện tại {i}/{len(symbols)}: {symbol}")
        try:
            df = fetch_equity(symbol, start, end)
            if len(df) >= 50:
                prices[symbol] = df.set_index("time")["close"].rename(symbol)
            else:
                failed.append(f"{symbol}: không đủ lịch sử")
        except Exception as exc:
            failed.append(f"{symbol}: {type(exc).__name__}")
        progress.progress(i / len(symbols))
    progress.empty(); status.empty()

    if len(prices) < 20:
        raise RuntimeError("Không đủ dữ liệu VN30 để tính Breadth hiện tại")

    panel = pd.DataFrame(prices).sort_index()
    last_date = panel.index.max()
    last = panel.iloc[-1]
    prev = panel.iloc[-2] if len(panel) >= 2 else pd.Series(index=panel.columns, dtype=float)
    ma20 = panel.rolling(20, min_periods=20).mean().iloc[-1]
    ma50 = panel.rolling(50, min_periods=50).mean().iloc[-1]
    ma200 = panel.rolling(200, min_periods=200).mean().iloc[-1]

    valid20 = last.notna() & ma20.notna()
    valid50 = last.notna() & ma50.notna()
    valid200 = last.notna() & ma200.notna()
    valid_adv = last.notna() & prev.notna()

    ma20_pct = (last[valid20] > ma20[valid20]).mean() * 100 if valid20.any() else np.nan
    ma50_pct = (last[valid50] > ma50[valid50]).mean() * 100 if valid50.any() else np.nan
    ma200_pct = (last[valid200] > ma200[valid200]).mean() * 100 if valid200.any() else np.nan
    adv_pct = (last[valid_adv] > prev[valid_adv]).mean() * 100 if valid_adv.any() else np.nan

    score = float(np.nanmean([ma20_pct, ma50_pct, ma200_pct, adv_pct]))
    state = "RẤT RỘNG" if score >= 70 else "RỘNG" if score >= 55 else "TRUNG TÍNH" if score >= 45 else "HẸP" if score >= 30 else "RẤT HẸP"

    return {
        "date": last_date,
        "symbols": len(symbols),
        "valid_symbols": len(prices),
        "failed": failed,
        "source": source,
        "pct_above_ma20": ma20_pct,
        "pct_above_ma50": ma50_pct,
        "pct_above_ma200": ma200_pct,
        "pct_advancers": adv_pct,
        "breadth_score": score,
        "breadth_state": state,
    }


def fetch_full_history(start_date, end_date):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    cursor = start_ts
    pieces = []
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=119), end_ts)
        pieces.append(fetch_index(cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + pd.Timedelta(days=1)
    return pd.concat(pieces, ignore_index=True).drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)


def calculate_features(df, level=49, park_window=22, stress_window=252):
    out = df.copy()
    close = out["close"]
    out["strength"] = (close.pct_change(63) * 0.4 + close.pct_change(126) * 0.2 + close.pct_change(189) * 0.2 + close.pct_change(252) * 0.2) * 100
    out["roro_equal"] = out["strength"].rolling(level, min_periods=level).mean()
    out["roro"] = out["strength"] - out["roro_equal"]
    out["trend_state"] = np.select([out["roro"] > 0, out["roro"] < 0], ["RISK ON", "RISK OFF"], default="TRUNG TÍNH")
    log_range = np.log(out["high"] / out["low"])
    out["parkinson_vol"] = np.sqrt(log_range.pow(2).rolling(park_window, min_periods=park_window).mean() / (4 * np.log(2)) * 252) * 100
    base = out["parkinson_vol"].rolling(stress_window, min_periods=max(60, park_window)).mean()
    std = out["parkinson_vol"].rolling(stress_window, min_periods=max(60, park_window)).std().replace(0, np.nan)
    out["stress_z"] = (out["parkinson_vol"] - base) / std
    out["stress_score"] = 50 + 10 * out["stress_z"].clip(-5, 5)
    out["regime"] = np.select(
        [out["roro"] > 0, out["stress_z"] >= 1.5, out["stress_z"] < 0.5],
        ["EXPANSION", "HIGH STRESS RISK OFF", "CORRECTION"],
        default="TRANSITION",
    )
    out.loc[(out["roro"] > 0) & (out["stress_z"] >= 0.5), "regime"] = "FRAGILE RALLY"
    return out


def validation(df):
    test = df[["regime", "close"]].copy()
    for horizon in [5, 10, 20]:
        test[f"f{horizon}"] = (test["close"].shift(-horizon) / test["close"] - 1) * 100
    rows = []
    for regime, group in test.groupby("regime"):
        row = {"Regime": regime, "Số quan sát": len(group)}
        for horizon in [5, 10, 20]:
            values = group[f"f{horizon}"].dropna()
            row[f"T{horizon} lợi suất TB %"] = values.mean()
            row[f"Tỷ lệ dương T{horizon} %"] = (values > 0).mean() * 100 if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


st.title("Vietnam Market Regime")
st.caption("Trend và Stress dùng dữ liệu lịch sử. Breadth chỉ đánh giá trạng thái hiện tại của VN30.")

with st.sidebar:
    key = st.text_input("API key VNStock", type="password", value=st.session_state.get("vnstock_api_key", ""))
    if key:
        st.session_state["vnstock_api_key"] = key
    start = st.date_input("Từ ngày", date(2015, 1, 1))
    end = st.date_input("Đến ngày", date.today())
    use_breadth = st.checkbox("Tính Breadth VN30 hiện tại", value=True)
    run = st.button("Lấy dữ liệu và phân tích", type="primary")

api_key = st.session_state.get("vnstock_api_key", "").strip()

if run:
    if not api_key:
        st.error("Chưa có API key VNStock.")
        st.stop()
    if start >= end:
        st.error("Ngày bắt đầu phải trước ngày kết thúc.")
        st.stop()

    try:
        configure_vnstock(api_key)
        with st.status("Đang lấy và tính dữ liệu", expanded=True) as status:
            st.write("Đang lấy VNINDEX")
            vnindex = calculate_features(fetch_full_history(start, end))
            st.write("Đang lấy VN30")
            vn30 = calculate_features(fetch_full_history(start, end))
            breadth = build_recent_breadth(api_key, end) if use_breadth else None
            status.update(label="Hoàn tất", state="complete")
        st.session_state["vnindex"] = vnindex
        st.session_state["vn30"] = vn30
        st.session_state["breadth"] = breadth
    except Exception as exc:
        st.error(f"Không thể hoàn tất: {type(exc).__name__}: {safe_error(exc)}")
        st.stop()

for name in ["vnindex", "vn30"]:
    df = st.session_state.get(name)
    if df is None:
        continue
    st.header(name.upper())
    latest = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Đóng cửa", f"{latest['close']:,.2f}")
    c2.metric("RORO", f"{latest['roro']:.2f}" if pd.notna(latest['roro']) else "N/A")
    c3.metric("Stress", f"{latest['stress_score']:.1f}" if pd.notna(latest['stress_score']) else "N/A")
    c4.metric("Regime sơ bộ", latest["regime"])
    tabs = st.tabs(["Tổng quan", "Trend", "Stress", "Kiểm định"])
    with tabs[0]:
        st.line_chart(df.set_index("time")[["close"]].dropna(), use_container_width=True)
    with tabs[1]:
        st.line_chart(df.set_index("time")[["roro"]].dropna(), use_container_width=True)
    with tabs[2]:
        st.line_chart(df.set_index("time")[["parkinson_vol", "stress_score"]].dropna(how="all"), use_container_width=True)
    with tabs[3]:
        st.dataframe(validation(df).round(4), hide_index=True, use_container_width=True)
        st.caption("T5, T10 và T20 là lợi suất trung bình theo phần trăm. Đây là kiểm định mô tả trên cùng tập dữ liệu.")

breadth = st.session_state.get("breadth")
if breadth:
    st.divider()
    st.header("Breadth VN30 hiện tại")
    st.caption("Ảnh chụp của VN30 hiện tại. Không sử dụng để mô phỏng thành phần VN30 lịch sử từ năm 2015.")
    q = st.columns(5)
    q[0].metric("Trên MA20", f"{breadth['pct_above_ma20']:.1f}%")
    q[1].metric("Trên MA50", f"{breadth['pct_above_ma50']:.1f}%")
    q[2].metric("Trên MA200", f"{breadth['pct_above_ma200']:.1f}%")
    q[3].metric("Số mã tăng", f"{breadth['pct_advancers']:.1f}%")
    q[4].metric("Breadth Score", f"{breadth['breadth_score']:.1f} | {breadth['breadth_state']}")
    st.info(f"Ngày dữ liệu Breadth: {breadth['date'].date()} | Thành phần hiện tại: {breadth['valid_symbols']}/{breadth['symbols']} mã.")
    if breadth["failed"]:
        st.warning("Mã chưa lấy được: " + ", ".join(breadth["failed"][:8]))

st.divider()
st.header("Lộ trình mô hình")
st.write("Trend và Stress được kiểm định lịch sử từ năm 2015. Breadth VN30 hiện tại chỉ mô tả trạng thái hiện tại. Breadth lịch sử sẽ chỉ được xây khi có thành phần VN30 theo từng kỳ rà soát hoặc một vũ trụ thị trường lịch sử phù hợp.")
