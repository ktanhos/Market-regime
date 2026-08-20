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
    return df.dropna(subset=["time", "close"]).sort_values("time").drop_duplicates("time", keep="last")


def configure_vnstock(api_key):
    key = (api_key or os.getenv("VNSTOCK_API_KEY", "")).strip()
    if not key:
        raise ValueError("Thiếu API key VNStock")
    os.environ["VNSTOCK_API_KEY"] = key
    register_user(api_key=key)


def rate_limit_gate():
    previous = st.session_state.get("last_vnstock_request", 0.0)
    wait = REQUEST_INTERVAL_SECONDS - (time.monotonic() - previous)
    if wait > 0:
        time.sleep(wait)
    st.session_state["last_vnstock_request"] = time.monotonic()


def safe_error(exc):
    return str(exc).replace(os.getenv("VNSTOCK_API_KEY", ""), "[API_KEY]")[:500]


def fetch_symbol(symbol, start, end):
    errors = []
    for source in ["KBS", "VCI"]:
        rate_limit_gate()
        try:
            raw = Quote(symbol=symbol, source=source).history(start=start, end=end, interval="1D")
            return clean(raw, symbol)
        except Exception as exc:
            errors.append(f"{source}: {safe_error(exc)}")
    raise RuntimeError(f"Không lấy được {symbol}: " + " | ".join(errors))


def fetch_full_history(symbol, start_date, end_date):
    cursor = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    pieces = []
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=119), end_ts)
        pieces.append(fetch_symbol(symbol, cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + pd.Timedelta(days=1)
    return pd.concat(pieces, ignore_index=True).drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)


def get_vn30_symbols():
    try:
        from vnstock import Reference
        listing = Reference().equity.list_by_group("VN30")
        cols = {str(c).lower(): c for c in listing.columns}
        col = cols.get("symbol") or cols.get("ticker") or cols.get("code")
        if col:
            symbols = sorted(listing[col].dropna().astype(str).str.upper().unique().tolist())
            if len(symbols) >= 20:
                return symbols[:30], "Danh sách VN30 hiện tại"
    except Exception:
        pass
    return VN30_FALLBACK, "Danh sách VN30 dự phòng"


def calculate_features(df, level=49, park_window=22, stress_window=252):
    out = df.copy()
    close = out["close"]
    out["strength"] = (close.pct_change(63) * 0.4 + close.pct_change(126) * 0.2 + close.pct_change(189) * 0.2 + close.pct_change(252) * 0.2) * 100
    out["roro_equal"] = out["strength"].rolling(level, min_periods=level).mean()
    out["roro"] = out["strength"] - out["roro_equal"]
    out["trend_state"] = np.select([out["roro"] > 0, out["roro"] < 0], ["TÍCH CỰC", "SUY YẾU"], default="TRUNG TÍNH")
    log_range = np.log(out["high"] / out["low"])
    out["parkinson_vol"] = np.sqrt(log_range.pow(2).rolling(park_window, min_periods=park_window).mean() / (4 * np.log(2)) * 252) * 100
    base = out["parkinson_vol"].rolling(stress_window, min_periods=max(60, park_window)).mean()
    std = out["parkinson_vol"].rolling(stress_window, min_periods=max(60, park_window)).std().replace(0, np.nan)
    out["stress_z"] = (out["parkinson_vol"] - base) / std
    out["stress_score"] = 50 + 10 * out["stress_z"].clip(-5, 5)
    out["stress_state"] = np.select([out["stress_z"] >= 1.5, out["stress_z"] >= 0.5, out["stress_z"] <= -0.5], ["RẤT CAO", "CAO", "THẤP"], default="BÌNH THƯỜNG")
    return out


def build_recent_breadth(api_key, end_date):
    configure_vnstock(api_key)
    symbols, source = get_vn30_symbols()
    start = (pd.Timestamp(end_date) - pd.DateOffset(days=450)).strftime("%Y-%m-%d")
    end = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    prices, failed = {}, []
    progress = st.progress(0, text="Đang kiểm tra sức khỏe của 30 cổ phiếu VN30")
    status = st.empty()
    for i, symbol in enumerate(symbols, 1):
        status.caption(f"Đang xử lý {i}/{len(symbols)} mã: {symbol}")
        try:
            df = fetch_symbol(symbol, start, end)
            if len(df) >= 200:
                prices[symbol] = df.set_index("time")["close"]
            else:
                failed.append(f"{symbol}: chưa đủ dữ liệu")
        except Exception as exc:
            failed.append(f"{symbol}: {type(exc).__name__}")
        progress.progress(i / len(symbols))
    progress.empty(); status.empty()
    if len(prices) < 20:
        raise RuntimeError("Không đủ dữ liệu để đánh giá độ rộng VN30")
    panel = pd.DataFrame(prices).sort_index()
    last = panel.iloc[-1]
    prev = panel.iloc[-2]
    ma20 = panel.rolling(20, min_periods=20).mean().iloc[-1]
    ma50 = panel.rolling(50, min_periods=50).mean().iloc[-1]
    ma200 = panel.rolling(200, min_periods=200).mean().iloc[-1]
    def pct(mask):
        return mask.mean() * 100 if len(mask) else np.nan
    valid20 = last.notna() & ma20.notna()
    valid50 = last.notna() & ma50.notna()
    valid200 = last.notna() & ma200.notna()
    valid_adv = last.notna() & prev.notna()
    p20 = pct(last[valid20] > ma20[valid20])
    p50 = pct(last[valid50] > ma50[valid50])
    p200 = pct(last[valid200] > ma200[valid200])
    adv = pct(last[valid_adv] > prev[valid_adv])
    score = float(np.nanmean([p20, p50, p200, adv]))
    state = "RẤT KHỎE" if score >= 70 else "KHỎE" if score >= 55 else "CÂN BẰNG" if score >= 45 else "YẾU" if score >= 30 else "RẤT YẾU"
    return {"date": panel.index.max(), "symbols": len(symbols), "valid_symbols": len(prices), "failed": failed, "source": source, "pct_above_ma20": p20, "pct_above_ma50": p50, "pct_above_ma200": p200, "pct_advancers": adv, "breadth_score": score, "breadth_state": state}


def current_regime(trend, stress, breadth=None):
    if breadth:
        b = breadth["breadth_state"]
        if trend == "TÍCH CỰC" and stress == "THẤP" and b in ["RẤT KHỎE", "KHỎE"]: return "THỊ TRƯỜNG TÍCH CỰC", "Xu hướng tốt, biến động thấp và nhiều cổ phiếu cùng tham gia."
        if trend == "TÍCH CỰC" and (stress in ["CAO", "RẤT CAO"] or b in ["YẾU", "RẤT YẾU"]): return "TĂNG NHƯNG CẦN THẬN TRỌNG", "Chỉ số vẫn tích cực nhưng rủi ro hoặc mức độ đồng thuận chưa tốt."
        if trend == "SUY YẾU" and stress in ["CAO", "RẤT CAO"] and b in ["YẾU", "RẤT YẾU"]: return "THỊ TRƯỜNG RỦI RO CAO", "Xu hướng suy yếu, biến động cao và phần lớn cổ phiếu chưa cho tín hiệu tốt."
        if trend == "SUY YẾU" and stress == "THẤP" and b in ["KHỎE", "RẤT KHỎE", "CÂN BẰNG"]: return "ĐANG ỔN ĐỊNH", "Xu hướng lớn chưa tích cực nhưng áp lực thị trường đã giảm và độ rộng đang cải thiện."
    if trend == "TÍCH CỰC" and stress == "THẤP": return "THỊ TRƯỜNG TÍCH CỰC", "Xu hướng đang tốt và biến động ở mức thấp."
    if trend == "SUY YẾU" and stress in ["CAO", "RẤT CAO"]: return "THỊ TRƯỜNG RỦI RO CAO", "Xu hướng suy yếu đi cùng biến động cao."
    return "GIAI ĐOẠN CHUYỂN TIẾP", "Các yếu tố đang cho tín hiệu trái chiều hoặc chưa đủ rõ ràng."


st.title("Vietnam Market Regime")
st.write("Ứng dụng giúp trả lời một câu hỏi đơn giản: thị trường Việt Nam đang ở trạng thái nào và những yếu tố nào đang tạo nên trạng thái đó.")
st.info("Cách hoạt động: ứng dụng nhìn vào xu hướng của chỉ số, mức độ biến động và sức khỏe của các cổ phiếu VN30 hiện tại. Sau đó các yếu tố được ghép lại để mô tả bức tranh thị trường ở thời điểm mới nhất.")

with st.expander("Các thông số trên ứng dụng có ý nghĩa gì?"):
    st.write("Xu hướng cho biết thị trường đang mạnh lên hay suy yếu so với chính giai đoạn trước. Biến động cho biết thị trường đang bình tĩnh hay có nhiều biến động bất thường. Sức khỏe VN30 cho biết trong 30 cổ phiếu lớn, có bao nhiêu mã đang có trạng thái giá tích cực.")
    st.write("Trên MA20, MA50 và MA200 lần lượt cho biết tỷ lệ cổ phiếu đang nằm trên mức giá trung bình của 20, 50 và 200 phiên gần nhất. Số mã tăng cho biết tỷ lệ cổ phiếu tăng trong phiên gần nhất.")
    st.write("Kết luận thị trường là phần tổng hợp để dễ theo dõi. Đây là công cụ mô tả trạng thái hiện tại, không phải khuyến nghị mua bán và không nhằm dự báo chính xác giá ngày mai.")

with st.sidebar:
    st.header("Dữ liệu")
    key = st.text_input("API key VNStock", type="password", value=st.session_state.get("vnstock_api_key", ""))
    if key: st.session_state["vnstock_api_key"] = key
    start = st.date_input("Xem lịch sử từ ngày", date(2015, 1, 1))
    end = st.date_input("Đến ngày", date.today())
    use_breadth = st.checkbox("Đánh giá sức khỏe VN30 hiện tại", value=True)
    run = st.button("Cập nhật và phân tích", type="primary", use_container_width=True)

api_key = st.session_state.get("vnstock_api_key", "").strip()
if run:
    if not api_key: st.error("Vui lòng nhập API key VNStock."); st.stop()
    if start >= end: st.error("Ngày bắt đầu phải trước ngày kết thúc."); st.stop()
    try:
        configure_vnstock(api_key)
        with st.status("Đang cập nhật dữ liệu", expanded=True) as status:
            st.write("Đang lấy dữ liệu VNINDEX")
            vnindex = calculate_features(fetch_full_history("VNINDEX", start, end))
            st.write("Đang lấy dữ liệu VN30")
            vn30 = calculate_features(fetch_full_history("VN30", start, end))
            breadth = build_recent_breadth(api_key, end) if use_breadth else None
            status.update(label="Đã hoàn tất", state="complete")
        st.session_state.update({"vnindex": vnindex, "vn30": vn30, "breadth": breadth})
    except Exception as exc:
        st.error(f"Không thể hoàn tất: {type(exc).__name__}: {safe_error(exc)}"); st.stop()

vnindex = st.session_state.get("vnindex")
vn30 = st.session_state.get("vn30")
breadth = st.session_state.get("breadth")

if vnindex is not None and vn30 is not None:
    latest_i = vnindex.iloc[-1]
    latest_30 = vn30.iloc[-1]
    overall, explanation = current_regime(latest_i["trend_state"], latest_i["stress_state"], breadth)
    st.header("Trạng thái thị trường hiện tại")
    st.subheader(overall)
    st.write(explanation)
    a,b,c,d = st.columns(4)
    a.metric("VNINDEX", f"{latest_i['close']:,.2f}")
    b.metric("Xu hướng", latest_i["trend_state"])
    c.metric("Mức biến động", latest_i["stress_state"])
    d.metric("Sức khỏe VN30", breadth["breadth_state"] if breadth else "Chưa tính")

    tabs = st.tabs(["Tổng quan", "Xu hướng", "Biến động", "VN30", "Dữ liệu lịch sử"])
    with tabs[0]:
        st.write("Ba lớp thông tin được sử dụng để nhìn thị trường từ ba góc khác nhau.")
        x,y,z = st.columns(3)
        x.metric("Xu hướng", latest_i["trend_state"], f"RORO {latest_i['roro']:.2f}")
        y.metric("Biến động", latest_i["stress_state"], f"Điểm {latest_i['stress_score']:.1f}")
        z.metric("Sức khỏe VN30", breadth["breadth_state"] if breadth else "Chưa tính", f"Điểm {breadth['breadth_score']:.1f}" if breadth else None)
    with tabs[1]:
        st.write("Xu hướng được tính từ dữ liệu lịch sử của VNINDEX và VN30. Giá trị dương cho thấy sức mạnh gần đây tốt hơn mức trung bình của chính thị trường.")
        st.line_chart(vnindex.set_index("time")[["roro"]].dropna(), use_container_width=True)
    with tabs[2]:
        st.write("Biến động được tính từ biên độ dao động giá. Điểm cao hơn cho thấy thị trường đang biến động mạnh hơn bình thường.")
        st.line_chart(vnindex.set_index("time")[["parkinson_vol", "stress_score"]].dropna(how="all"), use_container_width=True)
    with tabs[3]:
        st.write("Phần này chỉ đánh giá tình trạng của 30 cổ phiếu VN30 ở thời điểm hiện tại. Mục đích là xem xu hướng của chỉ số có được nhiều cổ phiếu cùng xác nhận hay không.")
        if breadth:
            q = st.columns(5)
            q[0].metric("Trên MA20", f"{breadth['pct_above_ma20']:.1f}%")
            q[1].metric("Trên MA50", f"{breadth['pct_above_ma50']:.1f}%")
            q[2].metric("Trên MA200", f"{breadth['pct_above_ma200']:.1f}%")
            q[3].metric("Mã tăng trong phiên", f"{breadth['pct_advancers']:.1f}%")
            q[4].metric("Sức khỏe chung", f"{breadth['breadth_score']:.1f}", breadth["breadth_state"])
            st.caption(f"Dữ liệu đến ngày {breadth['date'].date()}. Có {breadth['valid_symbols']}/{breadth['symbols']} mã đủ dữ liệu để tính.")
            if breadth["failed"]: st.warning("Một số mã chưa lấy được dữ liệu: " + ", ".join(breadth["failed"]))
        else: st.info("Chưa tính sức khỏe VN30 trong lần chạy này.")
    with tabs[4]:
        st.write("Phần lịch sử giúp quan sát VNINDEX và VN30 đã thay đổi như thế nào theo thời gian. Sức khỏe VN30 ở phía trên chỉ là ảnh chụp của thời điểm hiện tại nên không được đưa ngược vào chuỗi lịch sử.")
        col1,col2 = st.columns(2)
        with col1: st.line_chart(vnindex.set_index("time")[["close"]], use_container_width=True)
        with col2: st.line_chart(vn30.set_index("time")[["close"]], use_container_width=True)
else:
    st.info("Nhập API key, chọn khoảng thời gian và bấm Cập nhật và phân tích để bắt đầu.")

st.divider()
st.header("Ứng dụng đang làm gì")
st.write("Dữ liệu lịch sử được dùng để đo xu hướng và mức biến động của VNINDEX và VN30. Sau đó ứng dụng lấy thêm dữ liệu gần đây của 30 cổ phiếu VN30 để xem sức khỏe thị trường hiện tại. Ba góc nhìn này được tổng hợp thành một trạng thái dễ theo dõi.")
st.caption("Phiên bản hiện tại tập trung vào việc mô tả thị trường đang ở trạng thái nào. Các lớp dữ liệu khác có thể được bổ sung sau khi có dữ liệu lịch sử phù hợp.")