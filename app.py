from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Vietnam Market Regime", page_icon="📊", layout="wide")

DATA_DIR = Path("data/processed")


@st.cache_data
def load_features(name: str):
    path = DATA_DIR / f"{name}_features.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def show_market(name: str, df: pd.DataFrame):
    latest = df.dropna(subset=["close"]).iloc[-1]
    st.subheader(name)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Đóng cửa", f"{latest['close']:,.2f}")
    c2.metric("RORO", "N/A" if pd.isna(latest['roro']) else f"{latest['roro']:.2f}")
    c3.metric("Xu hướng", latest['trend_state'])
    c4.metric("Stress", "N/A" if pd.isna(latest['stress_score']) else f"{latest['stress_score']:.1f}")

    chart = df.set_index("date")[["close"]].dropna()
    st.line_chart(chart)

    with st.expander(f"Chi tiết chỉ báo {name}"):
        st.dataframe(df.tail(30), use_container_width=True)


st.title("Vietnam Market Regime")
st.caption("Khung phân tích chế độ thị trường Việt Nam")
st.info("Phiên bản nền tảng. Dữ liệu hiện hiển thị Trend và Stress. Breadth và Regime tổng hợp sẽ chỉ được kích hoạt sau khi có dữ liệu toàn thị trường và kiểm định lịch sử.")

vnindex = load_features("vnindex")
vn30 = load_features("vn30")

if vnindex is None and vn30 is None:
    st.warning("Chưa có dữ liệu đã xử lý. Pipeline cần cập nhật VNINDEX và VN30 trước.")
    st.code("python scripts/update_data.py\npython scripts/build_features.py")
else:
    if vnindex is not None:
        show_market("VNINDEX", vnindex)
    if vn30 is not None:
        show_market("VN30", vn30)

st.divider()
st.header("Kiến trúc mô hình")
cols = st.columns(4)
for col, title, desc, status in zip(
    cols,
    ["Trend", "Stress", "Breadth", "Regime"],
    ["RORO và động lượng đa kỳ hạn", "Parkinson Volatility và Stress Score", "Độ rộng toàn thị trường", "Phân loại chế độ thị trường"],
    ["Đang hoạt động", "Đang hoạt động", "Chưa tích hợp", "Chưa kích hoạt"],
):
    with col:
        st.subheader(title)
        st.write(desc)
        st.caption(status)
