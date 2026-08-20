import streamlit as st

st.set_page_config(page_title="Vietnam Market Regime", page_icon="📊", layout="wide")

st.title("Vietnam Market Regime")
st.caption("Khung phân tích chế độ thị trường Việt Nam")

st.info("Phiên bản nền tảng: dữ liệu và mô hình sẽ được kết nối từng lớp. Chưa đưa ra tín hiệu giao dịch.")

st.header("Kiến trúc ban đầu")
cols = st.columns(4)
for col, title, desc in zip(
    cols,
    ["Trend", "Stress", "Breadth", "Regime"],
    ["RORO và động lượng đa kỳ hạn", "Biến động và căng thẳng thị trường", "Độ rộng thị trường", "Tổng hợp trạng thái"],
):
    with col:
        st.subheader(title)
        st.write(desc)

st.header("Trạng thái")
st.warning("Chưa có dữ liệu thị trường. Hãy chạy pipeline dữ liệu trước khi sử dụng dashboard.")
