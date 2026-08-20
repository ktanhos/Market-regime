from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"

st.set_page_config(page_title="Vietnam Market Regime", layout="wide")

@st.cache_data(ttl=3600)
def load_features(name):
    path = DATA_DIR / f"{name}_features.parquet"
    return pd.read_parquet(path) if path.exists() else None

def show_market(name, df):
    latest = df.dropna(subset=["close"]).iloc[-1]
    st.subheader(name)
    a,b,c,d = st.columns(4)
    a.metric("Đóng cửa", f"{latest['close']:,.2f}")
    a2 = latest.get("roro")
    b.metric("RORO", "Chưa đủ dữ liệu" if pd.isna(a2) else f"{a2:.2f}")
    c.metric("Xu hướng", str(latest.get("trend_state", "Chưa xác định")))
    a4 = latest.get("stress_score")
    d.metric("Căng thẳng", "Chưa đủ dữ liệu" if pd.isna(a4) else f"{a4:.1f}")
    st.line_chart(df.set_index("date")[["close"]].dropna(), use_container_width=True)

st.title("Vietnam Market Regime")
st.caption("Khung phân tích chế độ thị trường Việt Nam")
st.info("Ứng dụng chỉ đọc dữ liệu đã xử lý. Tầng API được tách riêng khỏi Streamlit.")

vnindex = load_features("vnindex")
vn30 = load_features("vn30")
if vnindex is None and vn30 is None:
    st.warning("Chưa có dữ liệu đã xử lý. Hãy chạy bộ cập nhật dữ liệu rồi đưa kết quả vào kho dữ liệu.")
else:
    if vnindex is not None: show_market("VNINDEX", vnindex)
    if vn30 is not None: show_market("VN30", vn30)

st.divider()
st.header("Các lớp mô hình")
for title, description, status in [
    ("Trend", "RORO và động lượng đa kỳ hạn", "Đang hoạt động"),
    ("Stress", "Parkinson Volatility và điểm căng thẳng", "Đang hoạt động"),
    ("Breadth", "Độ rộng toàn thị trường", "Chưa tích hợp"),
    ("Regime", "Tổng hợp trạng thái", "Chưa kích hoạt"),
]:
    st.subheader(title)
    st.write(description)
    st.caption(status)
