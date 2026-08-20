from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import PROCESSED_DIR
from src.vnstock_data import incremental_update
from scripts.build_features import build_features

st.set_page_config(page_title="Vietnam Market Regime", page_icon="📊", layout="wide")


@st.cache_data
def load_features(name: str):
    path = PROCESSED_DIR / f"{name}_features.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def refresh_market_data():
    status = st.status("Đang cập nhật dữ liệu thị trường", expanded=True)
    try:
        for symbol, name in [("VNINDEX", "vnindex"), ("VN30", "vn30")]:
            status.write(f"Đang lấy dữ liệu {symbol}")
            incremental_update(symbol=symbol, dataset_name=name, asset_type="index")
            status.write(f"Đang tính chỉ báo {symbol}")
            build_features(name)
        status.update(label="Đã cập nhật dữ liệu thành công", state="complete", expanded=False)
        st.cache_data.clear()
        return True
    except Exception as exc:
        status.update(label="Cập nhật dữ liệu không thành công", state="error", expanded=True)
        st.exception(exc)
        return False


def show_market(name: str, df: pd.DataFrame):
    latest = df.dropna(subset=["close"]).iloc[-1]
    st.subheader(name)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Đóng cửa", f"{latest['close']:,.2f}")
    c2.metric("RORO", "Chưa đủ dữ liệu" if pd.isna(latest['roro']) else f"{latest['roro']:.2f}")
    c3.metric("Xu hướng", latest["trend_state"])
    c4.metric("Stress", "Chưa đủ dữ liệu" if pd.isna(latest['stress_score']) else f"{latest['stress_score']:.1f}")

    chart = df.set_index("date")[["close"]].dropna()
    st.line_chart(chart)

    with st.expander(f"Chi tiết chỉ báo {name}"):
        st.dataframe(df.tail(30), use_container_width=True)


st.title("Vietnam Market Regime")
st.caption("Khung phân tích chế độ thị trường Việt Nam")
st.info("Phiên bản nền tảng. Trend và Stress được tính từ dữ liệu thực tế. Breadth và Regime tổng hợp sẽ được bổ sung sau khi có dữ liệu toàn thị trường và kiểm định lịch sử.")

col1, col2 = st.columns([1, 4])
with col1:
    update_clicked = st.button("Cập nhật dữ liệu", type="primary", use_container_width=True)
with col2:
    st.caption("Nút này gọi VNStock, cập nhật VNINDEX và VN30, sau đó tính lại RORO và Stress trực tiếp trên Streamlit.")

if update_clicked:
    if refresh_market_data():
        st.rerun()

vnindex = load_features("vnindex")
vn30 = load_features("vn30")

if vnindex is None and vn30 is None:
    st.warning("Chưa có dữ liệu. Hãy bấm Cập nhật dữ liệu để chạy pipeline trực tiếp trên Streamlit.")
else:
    if vnindex is not None:
        show_market("VNINDEX", vnindex)
    if vn30 is not None:
        show_market("VN30", vn30)

st.divider()
st.header("Kiến trúc mô hình")
cols = st.columns(4)
for col, title, desc, status_text in zip(
    cols,
    ["Trend", "Stress", "Breadth", "Regime"],
    ["RORO và động lượng đa kỳ hạn", "Parkinson Volatility và Stress Score", "Độ rộng toàn thị trường", "Phân loại chế độ thị trường"],
    ["Đang hoạt động", "Đang hoạt động", "Chưa tích hợp", "Chưa kích hoạt"],
):
    with col:
        st.subheader(title)
        st.write(desc)
        st.caption(status_text)
