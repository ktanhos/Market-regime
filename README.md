# Trạng thái thị trường Việt Nam

Ứng dụng Streamlit mô tả **trạng thái hiện tại** của thị trường chứng khoán Việt
Nam và chuyển trạng thái đó thành thông tin tham chiếu cho quản trị rủi ro danh
mục.

Ứng dụng **không** dự báo giá ngày mai và **không** phải hệ thống tín hiệu mua
bán.

## Kiến trúc

```text
Streamlit UI            app.py                     chỉ vẽ, không import vnstock
    ↓
Market Data Layer       schema · storage · vnstock_data · universe · updater
    ↓
Feature Layer           trend · stress · breadth · dispersion · concentration · features
    ↓
Market Regime Layer     regime
    ↓
Portfolio Risk Layer    portfolio_risk
    ↓
Interpretation Layer    narrative                  chỉ diễn giải, không tính lại gì
    ↓
Streamlit UI            app.py
```

Phụ thuộc chỉ đi xuống. Không có import vòng. Ba ràng buộc này được kiểm tra tự
động trong `tests/test_architecture.py` bằng cách phân tích AST của chính mã
nguồn, nên chúng không thể mục nát theo thời gian.

| Module | Tầng | Vai trò |
|---|---|---|
| `src/config.py` | nền | đường dẫn, thứ tự nguồn, tần suất gọi, mọi ngưỡng |
| `src/logging_config.py` | nền | logger dùng chung |
| `src/schema.py` | nền | lược đồ chuẩn, chuẩn hóa, hợp nhất, kiểm tra |
| `src/storage.py` | dữ liệu | Parquet + JSON, không có lời gọi mạng |
| `src/vnstock_data.py` | dữ liệu | module DUY NHẤT import vnstock |
| `src/universe.py` | dữ liệu | ảnh chụp VN30 hiện tại kèm `as_of` |
| `src/updater.py` | dữ liệu | pipeline cập nhật, nơi duy nhất khởi động lời gọi API |
| `src/trend.py` | feature | RORO |
| `src/stress.py` | feature | Market Stress, Volatility Stress Proxy |
| `src/breadth.py` | feature | độ lan tỏa rổ VN30 hiện tại |
| `src/dispersion.py` | feature | phân hóa lợi suất |
| `src/concentration.py` | feature | tập trung rủi ro proxy |
| `src/features.py` | feature | gom toàn bộ chỉ tiêu, đọc từ đĩa |
| `src/quality.py` | feature | báo cáo chất lượng dữ liệu |
| `src/regime.py` | regime | bảng quyết định Market Regime |
| `src/portfolio_risk.py` | danh mục | chuyển chế độ thành tham chiếu rủi ro |
| `src/narrative.py` | diễn giải | dịch các chỉ tiêu sang ngôn ngữ phổ thông: đang ở đâu, vì sao, đã đổi gì, cần theo dõi gì |
| `src/github_store.py` | hạ tầng | đồng bộ một commit mỗi lần cập nhật, chỉ tải lên tệp thực sự đổi nội dung |

## Nguyên tắc kiến trúc

1. **Mở dashboard không gọi API.** Trang chính chỉ đọc dữ liệu đã lưu trong
   `data/`. Toàn bộ lời gọi mạng nằm trong `src/vnstock_data.py` và chỉ được
   kích hoạt từ `src/updater.py` khi người dùng bấm nút. `app.py` không import
   vnstock; điều này được kiểm tra tự động.
2. **Một lược đồ dữ liệu duy nhất.** Mọi bảng giá đều là
   `date | open | high | low | close | volume`. Lớp ghi và lớp đọc dùng chung
   `src/schema.py`.
3. **GitHub là kho lưu trữ bền.** Filesystem của Streamlit Cloud là tạm thời:
   container khởi động lại là mất dữ liệu vừa ghi. Mỗi lần cập nhật đẩy toàn bộ
   tệp dữ liệu lên GitHub trong **đúng một commit**.
4. **VN30 hiện tại không phải VN30 lịch sử.** Xem phần dưới.

## Khởi tạo dữ liệu lần đầu

Kho dữ liệu trong repository chỉ có sẵn hai chỉ số. Giá cổ phiếu VN30 phải được
tải về ở lần chạy đầu tiên, và pipeline tự nhận biết điều đó.

Quyết định nằm ở **từng mã**, không phải ở cả lượt chạy:

| Tình trạng `data/raw/stocks/<MÃ>.parquet` | Hành động |
|---|---|
| chưa có tệp | lấy đủ 430 ngày lịch để tính được MA200 của chính mã đó |
| đã có tệp | chỉ lấy từ ngày cuối cùng trừ 12 ngày chồng lấn |

Nhờ vậy 27 mã đã có và 3 mã còn thiếu sẽ cho 27 lượt cập nhật tăng dần cộng 3
lượt tải đủ lịch sử. Đây cũng là cách một lượt chạy bị API giới hạn truy cập tiếp
tục ở lần sau: phần đã lấy được giữ nguyên, không tải lại từ đầu.

Ba chế độ được ghi vào `update_log.json`:

* `first_run` — chưa có tệp cổ phiếu nào
* `partial_bootstrap` — có một phần, còn thiếu hoặc thiếu lịch sử
* `incremental` — đã đủ, chỉ cập nhật phần mới

Một lượt chạy chỉ được coi là **hoàn tất** khi mọi nguồn thành công, mọi tệp kỳ
vọng đều tồn tại trên đĩa, và đã đồng bộ được lên GitHub.

### Hạn mức truy cập

Nguồn dữ liệu giới hạn số lượt gọi **theo phút**, không phải theo khoảng cách
giữa hai lượt:

| Gói | Hạn mức |
|---|---|
| Khách, không API key | 20 lượt/phút |
| API key miễn phí | 60 lượt/phút |
| Các gói tài trợ | 180 – 600 lượt/phút |

Một lượt khởi tạo cần 33 lượt gọi (1 danh sách + 2 chỉ số + 30 cổ phiếu), nên
`RateLimiter` điều tiết bằng cửa sổ trượt một phút thay vì chỉ nghỉ giữa hai
lượt. Mặc định đặt dưới hạn mức thật để chừa chỗ cho `scripts/check_api.py` chạy
trước đó.

Nếu vẫn bị chặn, pipeline **chờ hết chu kỳ rồi thử lại** tối đa
`MAX_RATE_LIMIT_RETRIES` lần. Chỉ khi hết lượt mới dừng, và phần dữ liệu đã lấy
được vẫn giữ nguyên.

### API key

Khóa được xác định ở đúng một chỗ, theo thứ tự ưu tiên:

1. Khóa nhập trong phiên Streamlit (thanh bên → **KẾT NỐI DỮ LIỆU**)
2. `st.secrets["VNSTOCK_API_KEY"]`
3. Biến môi trường `VNSTOCK_API_KEY`
4. Không có khóa → chạy ở gói Khách

Đã cấu hình Secrets thì **không cần nhập gì trong giao diện**. Ô nhập chỉ để thử
một khóa khác trong phiên hiện tại; giá trị đó nằm trong `st.session_state`,
không ghi xuống tệp, không vào dữ liệu, không lên GitHub.

**Streamlit Cloud** — App settings → Secrets:

```toml
VNSTOCK_API_KEY = "YOUR_KEY"
```

**GitHub Actions** — Repository Settings → Secrets and variables → Actions →
New repository secret → tên `VNSTOCK_API_KEY`. Workflow vẫn chạy bình thường khi
secret chưa tồn tại, chỉ ở hạn mức gói Khách.

**Local**:

```bash
export VNSTOCK_API_KEY="YOUR_KEY"
```

Hạn mức không được suy ra từ việc biến môi trường có tồn tại hay không. Sau khi
áp khóa, hệ thống hỏi lại chính client xem nó đang ở gói nào
(`vnai.beam.auth.authenticator`) rồi mới lấy hạn mức từ đó, nên không thể xảy ra
chuyện chạy ở 60 lượt/phút trong khi client vẫn ở gói Khách. Nếu nguồn vẫn từ
chối dù đã điều tiết, bộ điều tiết tự hạ về mức của gói Khách.

## Luồng dữ liệu

```text
Streamlit (bấm "Cập nhật dữ liệu")
  → đọc dữ liệu đã lưu, xác định ngày cuối cùng của từng tập
  → lấy VNINDEX và VN30 phần còn thiếu
  → lấy danh sách VN30 hiện tại từ API
  → lấy giá từng mã VN30 (tuần tự, có nghỉ, có backoff, không song song)
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
data/raw/vnindex.parquet             chỉ số VNINDEX, nền 8 năm
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

Quote(symbol="VNINDEX", source="VCI").history(start=..., end=..., interval="1D")
Listing(source="VCI").symbols_by_group("VN30")
```

Giao diện mà tầng trên được phép dùng, khai báo trong `src/vnstock_data.py`:

```python
fetch_history(symbol, start, end, asset_type)  -> FetchResult
fetch_index(symbol, start, end)                -> FetchResult
fetch_equity(symbol, start, end)               -> FetchResult
fetch_index_members(index="VN30")              -> list[str]
connectivity_check()                           -> list[ProbeResult]
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

* Danh sách VN30 **lấy từ API** mỗi lần cập nhật. Danh sách trong mã nguồn chỉ
  là fallback, luôn kèm ngày chụp, và khi đang dùng fallback thì dashboard hiện
  cảnh báo rõ chứ không âm thầm.
* Lịch sử 200 phiên của một cổ phiếu chỉ dùng để tính MA200 của **chính cổ
  phiếu đó**. Nó không hàm ý cổ phiếu đó đã thuộc VN30 trong 200 phiên ấy.
* **Không** dựng breadth lịch sử, **không** dựng VN30 regime lịch sử, **không**
  backtest cấu trúc VN30 theo rổ hiện tại.
* Phân vị của Phân hóa và Tập trung rủi ro được tính trên rổ hiện tại kéo ngược
  về quá khứ và luôn được ghi rõ là bối cảnh mô tả, không phải thống kê lịch sử
  của chỉ số VN30.

## Các chỉ tiêu

**Trend (`src/trend.py`)**

```text
Strength = ROC63*0.4 + ROC126*0.2 + ROC189*0.2 + ROC252*0.2   (%)
RORO     = Strength − trung bình động 49 phiên của Strength
```

Ba mức TÍCH CỰC / TRUNG TÍNH / SUY YẾU. Vùng trung tính rộng bằng 0.5 lần độ
lệch chuẩn 252 phiên của chính chuỗi RORO, nên ngưỡng tự điều chỉnh theo biên độ
dữ liệu quan sát được. RORO > 0 không được coi là "Risk On" tuyệt đối.

**Stress (`src/stress.py`)**

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

**Quản trị danh mục (`src/portfolio_risk.py`)** — chuyển chế độ thành mức rủi ro tham
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

`tests/test_bootstrap.py` chạy đủ các tình huống khởi tạo: lần đầu từ thư mục
rỗng, khởi tạo bổ sung, tệp thiếu lịch sử, một mã lỗi, bị giới hạn truy cập giữa
chừng và lượt chạy tiếp theo.

`tests/test_architecture.py` phân tích AST của mã nguồn để bảo đảm: chỉ
`vnstock_data.py` import vnstock, phụ thuộc giữa các tầng chỉ đi xuống, không có
import vòng, không còn lối vào vnstock cũ, không có đường dẫn hardcode, và các
module chết đã bị xóa hẳn.

Kết nối thật tới vnstock được xác minh riêng bằng workflow
`.github/workflows/check_api.yml` hoặc `python scripts/check_api.py`.

## Giới hạn hiện tại

* Giá cổ phiếu VN30 chỉ nằm ở `data/raw/stocks/<MÃ>.parquet`, viết hoa, một
  đường dẫn duy nhất. `scripts/validate_data.py` báo lỗi nếu còn tệp ở bố cục cũ
  `data/raw/<mã>.parquet`.
* Không có dữ liệu thành phần VN30 lịch sử → không có breadth, dispersion hay
  regime lịch sử của VN30.
* Không có dữ liệu phái sinh ổn định → chỉ số căng thẳng chỉ dùng biến động
  VNINDEX.
* Chưa có dữ liệu vốn hóa đáng tin cậy → tập trung rủi ro dùng proxy biến động.
* Các ngưỡng của Market Regime là quy tắc mô tả tường minh, chưa qua kiểm định
  lợi suất tương lai. Chúng không phải ngưỡng tối ưu và không sinh tín hiệu.
