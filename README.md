# Vietnam Market Regime

Khung nghiên cứu nhận diện chế độ thị trường chứng khoán Việt Nam.

## Mục tiêu

Xây dựng hệ thống theo ba lớp chính:

1. **Trend**: đo động lượng và chế độ Risk On hoặc Risk Off bằng RORO.
2. **Stress**: đo biến động và mức độ căng thẳng thị trường.
3. **Breadth**: đo mức độ lan tỏa của xu hướng giữa các cổ phiếu.

Các lớp trên sẽ được kết hợp thành Market Regime sau khi kiểm định lịch sử. Phiên bản đầu tiên chưa gán ngưỡng tối ưu và chưa tạo tín hiệu mua bán.

## Cấu trúc

```text
Market-regime/
├── app.py
├── requirements.txt
├── data/
├── scripts/
│   └── update_data.py
├── src/
│   ├── data.py
│   ├── vnstock_data.py
│   ├── roro.py
│   ├── volatility.py
│   ├── breadth.py
│   └── regime.py
└── .github/workflows/
    └── syntax-check.yml
```

## Nguyên tắc thiết kế

Dữ liệu VNStock được cô lập trong `src/vnstock_data.py` để khi giao diện hoặc API thay đổi chỉ cần sửa lớp dữ liệu.

Dữ liệu thị trường sau khi chuẩn hóa được lưu dạng Parquet. Streamlit đọc lớp dữ liệu đã xử lý thay vì tải lại toàn bộ lịch sử ở mỗi lần mở ứng dụng.

## Phiên bản đầu tiên

Hiện tại hệ thống mới có khung nền:

`VNStock → Data Layer → Indicators → Regime Engine → Streamlit`

Các ngưỡng trong Regime Engine hiện chỉ là ngưỡng khởi tạo để phục vụ backtest, chưa được coi là ngưỡng chính thức.

## Chạy ứng dụng

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cập nhật dữ liệu

```bash
python scripts/update_data.py
```

Bước tiếp theo là xây pipeline dữ liệu thực tế, tính RORO và volatility cho VNINDEX và VN30, sau đó bổ sung dữ liệu cổ phiếu để tính breadth và kiểm định từng chế độ theo lợi suất tương lai.
