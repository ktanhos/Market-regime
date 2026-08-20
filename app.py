from datetime import date
import queue
import threading
import time

import numpy as np
import pandas as pd
import streamlit as st
from vnstock import Market, Reference, register_user

st.set_page_config(page_title="Vietnam Market Regime", page_icon="📊", layout="wide")

VN30_FALLBACK = ["ACB","BCM","BID","BVH","CTG","FPT","GAS","GVR","HDB","HPG","MBB","MSN","MWG","PLX","SAB","SHB","SSB","SSI","STB","TCB","TPB","VCB","VHM","VIB","VIC","VJC","VNM","VPB","VRE","VIX"]


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


def call_with_timeout(func, timeout=45, **kwargs):
    result = queue.Queue(maxsize=1)
    def worker():
        try:
            result.put((True, func(**kwargs)))
        except Exception as exc:
            result.put((False, exc))
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        ok, value = result.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(f"Quá thời gian {timeout} giây")
    if ok:
        return value
    raise value


def fetch_history(fetcher, start_date, end_date, chunk_days=119, timeout=45, retries=2, progress_callback=None):
    pieces = []
    cursor = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    chunks = []
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=chunk_days), end_ts)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)

    errors = []
    for n, (chunk_start, chunk_end) in enumerate(chunks, 1):
        if progress_callback:
            progress_callback(n, len(chunks), chunk_start, chunk_end)
        last_error = None
        for attempt in range(retries):
            try:
                raw = call_with_timeout(
                    fetcher,
                    timeout=timeout,
                    start=chunk_start.strftime("%Y-%m-%d"),
                    end=chunk_end.strftime("%Y-%m-%d"),
                    interval="1D",
                )
                pieces.append(normalize_ohlcv(raw))
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(1.5 * (attempt + 1))
        if last_error is not None:
            errors.append(f"{chunk_start.date()} đến {chunk_end.date()}: {type(last_error).__name__}")

    if not pieces:
        detail = "; ".join(errors[:3])
        raise ConnectionError(f"Không lấy được dữ liệu trong toàn bộ khoảng thời gian. {detail}")

    out = pd.concat(pieces, ignore_index=True)
    out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return out, errors


def get_vn30_symbols():
    try:
        listing = Reference().equity.list_by_group("VN30")
        cols = {str(c).lower(): c for c in listing.columns}
        symbol_col = cols.get("symbol") or cols.get("ticker") or cols.get("code")
        if symbol_col is not None:
            symbols = listing[symbol_col].dropna().astype(str).str.upper().unique().tolist()
            if len(symbols) >= 20:
                return symbols, "Danh sách tham chiếu VN30 hiện tại"
    except Exception:
        pass
    return VN30_FALLBACK, "Danh sách VN30 dự phòng"


def fetch_index_history(market, symbol, start_date, end_date):
    df, _ = fetch_history(market.index(symbol).ohlcv, start_date, end_date)
    return df


def fetch_equity_history(market, symbol, start_date, end_date, progress_callback=None):
    return fetch_history(market.equity(symbol).ohlcv, start_date, end_date, progress_callback=progress_callback)


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
    out["stress_state"] = np.select([out["stress_z"] >= 1.5, out["stress_z"] >= 0.5, out["stress_z"] <= -0.5], ["CAO", "TĂNG", "THẤP"], default="BÌNH THƯỜNG")
    return out


def build_breadth(market, start_date, end_date):
    symbols, source = get_vn30_symbols()
    breadth_start = max(pd.Timestamp(start_date), pd.Timestamp(end_date) - pd.DateOffset(years=3)).date()
    series, failed, partial = {}, [], []
    progress = st.progress(0, text="Đang lấy dữ liệu Breadth")
    status = st.empty()

    for i, symbol in enumerate(symbols, 1):
        status.caption(f"Đang xử lý {i}/{len(symbols)}: {symbol}")
        def update_chunk(n, total, a, b):
            progress.progress((i - 1 + n / total) / len(symbols), text=f"Breadth {i}/{len(symbols)}: {symbol} | đoạn {n}/{total}")
        try:
            df, errors = fetch_equity_history(market, symbol, breadth_start, end_date, update_chunk)
            if len(df) >= 20:
                series[symbol] = df.set_index("date")["close"].rename(symbol)
                if errors:
                    partial.append(f"{symbol}: thiếu {len(errors)} đoạn")
            else:
                failed.append(f"{symbol}: không đủ dữ liệu")
        except Exception as exc:
            failed.append(f"{symbol}: {type(exc).__name__}")
        progress.progress(i / len(symbols), text=f"Breadth {i}/{len(symbols)}: {symbol}")

    progress.empty()
    status.empty()
    if len(series) < 10:
        raise ValueError("Không đủ dữ liệu cổ phiếu để tính Breadth")

    prices = pd.DataFrame(series).sort_index()
    coverage = prices.notna().sum(axis=1).replace(0, np.nan)
    ma20 = prices.rolling(20, min_periods=20).mean()
    ma50 = prices.rolling(50, min_periods=50).mean()
    ma200 = prices.rolling(200, min_periods=200).mean()
    breadth = pd.DataFrame(index=prices.index)
    breadth["breadth_coverage"] = coverage
    breadth["pct_above_ma20"] = (prices > ma20).sum(axis=1) / prices.notna().sum(axis=1).replace(0, np.nan) * 100
    breadth["pct_above_ma50"] = (prices > ma50).sum(axis=1) / prices.notna().sum(axis=1).replace(0, np.nan) * 100
    breadth["pct_above_ma200"] = (prices > ma200).sum(axis=1) / prices.notna().sum(axis=1).replace(0, np.nan) * 100
    ret = prices.pct_change()
    breadth["pct_advancers"] = (ret > 0).sum(axis=1) / ret.notna().sum(axis=1).replace(0, np.nan) * 100
    breadth["breadth_score"] = breadth[["pct_above_ma20", "pct_above_ma50", "pct_above_ma200", "pct_advancers"]].mean(axis=1)
    breadth["breadth_state"] = np.select([breadth["breadth_score"] >= 70, breadth["breadth_score"] >= 55, breadth["breadth_score"] >= 45, breadth["breadth_score"] >= 30], ["RẤT RỘNG", "RỘNG", "TRUNG TÍNH", "HẸP"], default="RẤT HẸP")
    meta = {"symbols": len(series), "failed": failed, "partial": partial, "source": source, "start": breadth_start, "requested": len(symbols)}
    return breadth.reset_index().rename(columns={"index": "date"}), meta


def classify_regime(row):
    if pd.isna(row["roro"]) or pd.isna(row["stress_score"]):
        return "CHƯA ĐỦ DỮ LIỆU"
    breadth = row.get("breadth_state", np.nan)
    if pd.notna(breadth):
        if row["roro"] > 0 and breadth in ["RẤT RỘNG", "RỘNG"] and row["stress_z"] < 0.5: return "EXPANSION"
        if row["roro"] > 0 and (breadth in ["HẸP", "RẤT HẸP"] or row["stress_z"] >= 0.5): return "FRAGILE RALLY"
        if row["roro"] <= 0 and breadth == "RẤT HẸP" and row["stress_z"] >= 1.5: return "HIGH STRESS RISK OFF"
        if row["roro"] <= 0 and breadth in ["HẸP", "RẤT HẸP"]: return "CORRECTION"
        return "TRANSITION"
    if row["roro"] > 0 and row["stress_z"] < 0.5: return "EXPANSION"
    if row["roro"] > 0: return "FRAGILE RALLY"
    if row["stress_z"] >= 1.5: return "HIGH STRESS RISK OFF"
    if row["stress_z"] < 0.5: return "CORRECTION"
    return "TRANSITION"


def merge_regime(df, breadth=None):
    out = df.copy()
    if breadth is not None: out = out.merge(breadth, on="date", how="left")
    else:
        out["breadth_state"] = np.nan
        out["breadth_score"] = np.nan
    out["regime"] = out.apply(classify_regime, axis=1)
    return out


def regime_validation(df):
    test = df[["regime", "close"]].copy()
    for horizon in [5, 10, 20]: test[f"forward_{horizon}"] = (test["close"].shift(-horizon) / test["close"] - 1) * 100
    rows = []
    for regime, group in test.groupby("regime"):
        if regime == "CHƯA ĐỦ DỮ LIỆU": continue
        row = {"Regime": regime, "Số quan sát": len(group)}
        for horizon in [5, 10, 20]:
            x = group[f"forward_{horizon}"].dropna()
            row[f"T{horizon} lợi suất TB %"] = x.mean()
            row[f"Tỷ lệ dương T{horizon} %"] = (x > 0).mean() * 100 if len(x) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def latest_text(value, fmt): return "Chưa đủ dữ liệu" if pd.isna(value) else fmt.format(value)


def show_market(name, df):
    latest = df.iloc[-1]
    st.subheader(name)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Đóng cửa", latest_text(latest["close"], "{:,.2f}"))
    c2.metric("RORO", latest_text(latest["roro"], "{:.2f}"))
    c3.metric("Stress", latest_text(latest["stress_score"], "{:.1f}"))
    c4.metric("Breadth", latest_text(latest.get("breadth_score", np.nan), "{:.1f}"))
    c5.metric("Regime", latest["regime"])
    chart = df.set_index("date")
    tabs = st.tabs(["Tổng quan","Trend","Stress","Breadth","Kiểm định"])
    with tabs[0]: st.line_chart(chart[["close"]].dropna(), use_container_width=True)
    with tabs[1]: st.line_chart(chart[["roro"]].dropna(), use_container_width=True)
    with tabs[2]: st.line_chart(chart[["parkinson_vol","stress_score"]].dropna(how="all"), use_container_width=True)
    with tabs[3]:
        cols=[c for c in ["pct_above_ma20","pct_above_ma50","pct_above_ma200","pct_advancers","breadth_score"] if c in chart.columns]
        if cols: st.line_chart(chart[cols].dropna(how="all"), use_container_width=True)
        else: st.info("Chưa có dữ liệu Breadth.")
    with tabs[4]:
        st.dataframe(regime_validation(df).round(4), use_container_width=True, hide_index=True)
        st.caption("T5, T10, T20 là lợi suất trung bình phần trăm sau 5, 10, 20 phiên. Đây là kiểm định mô tả trên cùng tập dữ liệu.")

st.title("Vietnam Market Regime")
st.caption("Khung phân tích chế độ thị trường Việt Nam")
st.write("Trend và Stress đã có thể quan sát và kiểm định mô tả. Breadth là lớp nghiên cứu tiếp theo. Regime chưa phải tín hiệu giao dịch.")
with st.sidebar:
    st.header("Cấu hình dữ liệu")
    api_key=st.text_input("API key VNStock", type="password")
    start_date=st.date_input("Từ ngày", value=date(2015,1,1))
    end_date=st.date_input("Đến ngày", value=date.today())
    enable_breadth=st.checkbox("Tính Breadth VN30", value=True)
    run=st.button("Lấy dữ liệu và phân tích", type="primary", use_container_width=True)

if run:
    if start_date >= end_date: st.error("Ngày bắt đầu phải trước ngày kết thúc."); st.stop()
    if not api_key.strip(): st.error("Vui lòng nhập API key VNStock."); st.stop()
    try:
        with st.status("Đang kết nối và phân tích", expanded=True) as status:
            register_user(api_key=api_key.strip())
            market=Market()
            raw_results={}
            for symbol in ["VNINDEX","VN30"]:
                st.write(f"Đang lấy dữ liệu {symbol}")
                raw_results[symbol]=calculate_features(fetch_index_history(market,symbol,start_date,end_date))
            breadth=meta=None
            if enable_breadth:
                st.write("Đang xây Breadth từ VN30")
                breadth,meta=build_breadth(market,start_date,end_date)
                st.write(f"Breadth hoàn tất: {meta['symbols']}/{meta['requested']} mã hợp lệ")
                if meta["failed"]: st.warning("Không lấy được: " + ", ".join(meta["failed"]))
                if meta["partial"]: st.info("Dữ liệu một phần: " + ", ".join(meta["partial"]))
            results={name:merge_regime(df,breadth) for name,df in raw_results.items()}
            status.update(label="Hoàn tất phân tích", state="complete")
        st.session_state["market_regime_results"]=results
        st.session_state["breadth_meta"]=meta
        st.success("Đã lấy dữ liệu và tính toán thành công.")
    except Exception as exc:
        st.error(f"Không thể hoàn tất quá trình lấy dữ liệu: {type(exc).__name__}: {exc}")
        st.stop()

results=st.session_state.get("market_regime_results")
if results:
    meta=st.session_state.get("breadth_meta")
    if meta: st.info(f"Breadth: {meta['symbols']}/{meta['requested']} mã. Bắt đầu từ {meta['start']}.")
    for name in ["VNINDEX","VN30"]: show_market(name,results[name])
    st.divider(); st.header("Lộ trình mô hình")
    st.write("Trend và Stress đã có thể quan sát và kiểm định mô tả. Breadth đang được kiểm tra bằng dữ liệu VN30 hiện tại. Bước tiếp theo là xây thành phần lịch sử hoặc Breadth toàn thị trường trước khi cố định Regime.")
else:
    st.info("Nhập API key VNStock ở thanh bên, chọn khoảng thời gian và bấm Lấy dữ liệu và phân tích.")
