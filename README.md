# Trạng thái thị trường Việt Nam

Ứng dụng Streamlit mô tả **trạng thái hiện tại** của thị trường chứng khoán Việt
Nam và chuyển trạng thái đó thành thông tin tham chiếu cho quản trị rủi ro danh
mục.

Ứng dụng **không** dự báo giá ngày mai và **không** phải hệ thống tín hiệu mua
bán.

## Nguyên tắc kiến trúc

1. **Mở dashboard không gọi API.** Trang chính chỉ đọc dữ liệu đã lưu trong
   `data/`. Toàn bộ lời gọi mạng nằm trong `src/vnstock_data.py` và chỉ được
   kích hoạt từ `src/updater.py` khi người dùng bấm nút.
2. **Một lược đồ dữ liệu duy nhất.** Mọi bảng giá đều là
   `date | open | high | low | close | volume`. Lớp ghi và lớp đọc dùng chung
   `src/schema.py`.
3. **GitHub là kho lưu trữ bền.** Filesystem của Streamlit Cloud là tạm thời:
   container khởi động lại là mất dữ liệu vừa ghi. Mỗi lần cập nhật đẩy toàn bộ
   tệp dữ liệu lên GitHub trong **đúng một commit**.
4. **VN30 hiện tại không phải VN30 lịch sử.** Xem phần dưới.

## Luồng dữ liệu

```text
Streamlit (bấm "Cập nhật dữ liệu")
  → đọc dữ liệu đã lưu, xác định ngày cuối cùng của từng tập
  → gọi API phần còn thiếu (tuần tự, có nghỉ, có backoff, không song song)
  → chuẩn hóa về lược đồ chuẩn
  → kiểm tra chất lượng (ngày trùng, high < low, giá <= 0)
  → hợp nhất với dữ liệu cũ, loại ngày trùng
  → lưu Parquet
  → tính lại toàn bộ chỉ tiêu
  → lưu processed data + nhật ký chất lượng
  → đồng bộ lên GitHub trong một commit
  → dashboard đọc lại dữ liệu mới
```

Bố cục kho dữ liệu:

```text
data/raw/vnindex.parquet             chỉ số VNINDEX, từ 2015
data/raw/vn30.parquet                chỉ số VN30
data/raw/stocks/<MÃ>.parquet         giá cổ phiếu VN30 hiện tại
data/reference/vn30_universe.json    ảnh chụp danh sách VN30 kèm ngày chụp
data/processed/vnindex_features.parquet
data/processed/vn30_snapshot.json    chỉ tiêu VN30 tính sẵn
data/processed/update_log.json       nhật ký và chất lượng lần cập nhật gần nhất
```

## API vnstock đang dùng

Giao diện chính thức hiện hành của vnstock 4.x:

```python
from vnstock import Quote, Listing

Quote(symbol="VNINDEX", source="VCI").history(start="2015-01-01", end="2026-08-22", interval="1D")
Listing(source="VCI").symbols_by_group("VN30")
```

Không dùng `Vnstock().stock(...)`, không dùng `Market(source=...)`, không gọi
`Quote.ohlcv` như một phương thức riêng — trong vnstock 4.x `ohlcv` chỉ là bí
danh của `history`.

Giới hạn nguồn dữ liệu, đọc trực tiếp từ gói đã cài:

| Nguồn | VNINDEX | VN30 | Cổ phiếu |
|-------|---------|------|----------|
| VCI   | có      | có   | có       |
| KBS   | có      | **không** | có  |

`vnstock/explorer/kbs/const.py::_INDEX_MAPPING` chỉ liệt kê VNINDEX, HNXINDEX và
UPCOMINDEX, nên VN30 bắt buộc phải đi qua VCI. Không nguồn nào có endpoint lấy
lịch sử nhiều mã trong một lần gọi, vì vậy các mã được gọi tuần tự.

## VN30 hiện tại, không phải VN30 lịch sử

Repository **không có** dữ liệu thành phần VN30 theo từng thời điểm trong quá
khứ. Do đó:

* Danh sách VN30 chỉ là ảnh chụp tại ngày cập nhật, lưu kèm `as_of`.
* Lịch sử 200 phiên của một cổ phiếu chỉ dùng để tính MA200 của **chính cổ
  phiếu đó**. Nó không hàm ý cổ phiếu đó đã thuộc VN30 trong 200 phiên ấy.
* **Không** dựng breadth lịch sử, **không** dựng VN30 regime lịch sử, **không**
  backtest cấu trúc VN30 theo rổ hiện tại.
* Phân vị của Phân hóa và Tập trung rủi ro được tính trên rổ hiện tại kéo ngược
  về quá khứ và luôn được ghi rõ là bối cảnh mô tả, không phải thống kê lịch sử
  của chỉ số VN30.

## Các chỉ tiêu

**Trend (`src/roro.py`)**

```text
Strength = ROC63*0.4 + ROC126*0.2 + ROC189*0.2 + ROC252*0.2   (%)
RORO     = Strength − trung bình động 49 phiên của Strength
```

Ba mức TÍCH CỰC / TRUNG TÍNH / SUY YẾU. Vùng trung tính rộng bằng 0.5 lần độ
lệch chuẩn 252 phiên của chính chuỗi RORO, nên ngưỡng tự điều chỉnh theo biên độ
dữ liệu quan sát được. RORO > 0 không được coi là "Risk On" tuyệt đối.

**Stress (`src/volatility.py`)**

```text
vol = sqrt( mean_{22 phiên}( ln(H/L)^2 ) / (4·ln2) · 252 ) · 100
```

Dùng **trung bình của bình phương log biên độ**, không phải phương sai quanh
trung bình. Chỉ một công thức duy nhất trong toàn dự án.

Chế độ căng thẳng dựa trên **phân vị** của chính chuỗi biến động VNINDEX trong
252 phiên gần nhất: <20 THẤP, 20–60 BÌNH THƯỜNG, 60–85 CAO, ≥85 RẤT CAO. Không
có ngưỡng tuyệt đối kiểu 20/25/35.

`stress_index` là **chỉ số căng thẳng biến động dạng proxy**, không phải VIX.
Basis phái sinh VN30F1M không được đưa vào mô hình vì chưa có nguồn dữ liệu phái
sinh đủ ổn định. Không có xác suất nào được gán cho chỉ số này.

**Breadth (`src/breadth.py`)** — % số mã trên MA20 / MA50 / MA200 và % số mã
tăng trong 1 / 5 / 20 phiên. Mỗi thành phần hiển thị số mã hợp lệ trên tổng số
mã. Không mặc định 30/30.

**Dispersion (`src/dispersion.py`)** — độ lệch chuẩn theo lát cắt ngang của lợi
suất 1/5/20 phiên, khung 20 phiên là chính. Nhãn suy ra từ phân vị của chính
chuỗi quan sát.

**Risk concentration (`src/concentration.py`)** — biến động 20 phiên của từng mã
chuẩn hóa thành tỷ trọng rủi ro, rồi tính Top 5 / Top 10 risk share, Herfindahl
và số mã đóng góp hiệu dụng `1/HHI`. Đây là **proxy về mức tập trung**, không
phải đóng góp rủi ro của một danh mục cụ thể vì chưa có trọng số danh mục.

**Market Regime (`src/regime.py`)** — bảng quyết định tường minh, không dùng học
máy: THUẬN LỢI / CẢNH BÁO / CHUYỂN TIẾP / CHỊU ÁP LỰC / CĂNG THẲNG, cộng CHƯA ĐỦ
DỮ LIỆU. Trend và Stress quyết định nhãn, Breadth tham gia ở nhánh cần xác nhận
độ lan tỏa. Dispersion và Risk concentration chỉ điều chỉnh **mức độ rủi ro**,
không đổi nhãn chế độ, vì chúng thiếu nền lịch sử đáng tin cậy.

**Quản trị danh mục (`src/portfolio.py`)** — chuyển chế độ thành mức rủi ro tham
chiếu, mức thận trọng, mức đòn bẩy tham chiếu, mức tập trung và khả năng duy trì
tỷ trọng cổ phiếu. Không có khuyến nghị từng cổ phiếu và không có tỷ trọng cố
định kiểu "70% cổ phiếu, 30% tiền" vì chưa có mô hình được kiểm định.

## Chạy ứng dụng

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cập nhật dữ liệu

Trong ứng dụng: thanh bên → **Kiểm tra kết nối API** → **Cập nhật dữ liệu**.

Ngoài ứng dụng:

```bash
python scripts/check_api.py       # thử VNINDEX, FPT và danh sách VN30
python scripts/update_data.py     # chạy toàn bộ pipeline
python scripts/update_data.py --features-only   # chỉ tính lại, không gọi API
python scripts/validate_data.py   # báo cáo chất lượng kho dữ liệu
```

## Đồng bộ GitHub

Token chỉ đến từ `st.secrets["GITHUB_TOKEN"]`, sau đó tới biến môi trường
`GITHUB_TOKEN`. Không có token nào trong mã nguồn.

Chưa cấu hình token thì dashboard vẫn chạy bình thường, chỉ riêng chức năng đồng
bộ bị tắt kèm hướng dẫn.

`.streamlit/secrets.toml`:

```toml
GITHUB_TOKEN = "..."
```

## Kiểm thử

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q
```

Bộ test không chạm mạng. Lớp gọi API được kiểm thử qua bộ giả lập, gồm các tình
huống: nguồn lỗi tạm thời, nguồn không hỗ trợ mã, rate limit, dữ liệu rỗng,
thiếu một mã, ngày trùng lặp, mất toàn bộ dữ liệu và tệp parquet hỏng.

Kết nối thật tới vnstock được xác minh riêng bằng workflow
`.github/workflows/check_api.yml` hoặc `python scripts/check_api.py`.

## Giới hạn hiện tại

* Không có dữ liệu thành phần VN30 lịch sử → không có breadth, dispersion hay
  regime lịch sử của VN30.
* Không có dữ liệu phái sinh ổn định → chỉ số căng thẳng chỉ dùng biến động
  VNINDEX.
* Chưa có dữ liệu vốn hóa đáng tin cậy → tập trung rủi ro dùng proxy biến động.
* Các ngưỡng của Market Regime là quy tắc mô tả tường minh, chưa qua kiểm định
  lợi suất tương lai. Chúng không phải ngưỡng tối ưu và không sinh tín hiệu.
