"""Đường dẫn, hằng số và ngưỡng dùng chung.

Toàn bộ tham số có thể tranh luận được tập trung tại đây để dễ kiểm định lại.
Không có ngưỡng nào được suy ra từ mô hình học máy hay từ kết quả backtest chưa công bố.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
STOCK_DIR = RAW_DIR / "stocks"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"

VNINDEX_DATASET = "vnindex"
VN30_INDEX_DATASET = "vn30"

UNIVERSE_FILE = REFERENCE_DIR / "vn30_universe.json"
UPDATE_LOG_FILE = PROCESSED_DIR / "update_log.json"
VN30_SNAPSHOT_FILE = PROCESSED_DIR / "vn30_snapshot.json"
VNINDEX_FEATURES_FILE = PROCESSED_DIR / "vnindex_features.parquet"

GITHUB_REPO = "ktanhos/Market-regime"
GITHUB_BRANCH = "main"

# --- Nguồn dữ liệu -----------------------------------------------------------
# vnstock 4.x: Quote(symbol=..., source=...).history(start=..., end=..., interval="1D")
# VCI hỗ trợ cả chỉ số VN30 và cổ phiếu. KBS chỉ hỗ trợ VNINDEX/HNXINDEX/UPCOMINDEX
# ở nhóm chỉ số nên chỉ dùng làm nguồn dự phòng cho VNINDEX và cổ phiếu.
PRIMARY_SOURCE = "VCI"
FALLBACK_SOURCES = ("KBS",)
INDEX_SOURCES = {
    "VNINDEX": ("VCI", "KBS"),
    "VN30": ("VCI",),
}

# --- Tần suất gọi API --------------------------------------------------------
# Nguồn dữ liệu giới hạn số lượt gọi theo phút. Gói Khách (không API key) chỉ
# cho 20 lượt/phút; API key miễn phí nâng lên 60. Đây là con số quan sát được
# từ chính thông báo của vnstock khi bị chặn, không phải phỏng đoán.
#
# Đặt thấp hơn hạn mức thật để chừa chỗ cho các lượt gọi ngoài pipeline
# (ví dụ scripts/check_api.py chạy ngay trước đó trong cùng một phút).
REQUESTS_PER_MINUTE_GUEST = 15
REQUESTS_PER_MINUTE_WITH_KEY = 45
RATE_LIMIT_WINDOW_SECONDS = 60.0
# Khi vẫn bị chặn dù đã điều tiết: chờ rồi thử lại, không bỏ cuộc ngay.
RATE_LIMIT_COOLDOWN_SECONDS = 60.0
MAX_RATE_LIMIT_RETRIES = 2
REQUEST_DELAY_SECONDS = RATE_LIMIT_WINDOW_SECONDS / REQUESTS_PER_MINUTE_GUEST

MAX_ATTEMPTS_PER_SOURCE = 2   # vnstock đã tự retry 3 lần bên trong mỗi lần gọi
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 16.0

# Biến môi trường / secret chứa API key vnstock (tùy chọn).
VNSTOCK_API_KEY_ENV = "VNSTOCK_API_KEY"


def requests_per_minute(has_api_key: bool = False) -> int:
    return REQUESTS_PER_MINUTE_WITH_KEY if has_api_key else REQUESTS_PER_MINUTE_GUEST


def request_spacing_seconds(has_api_key: bool = False) -> float:
    """Khoảng cách tối thiểu giữa hai lượt gọi để không vượt hạn mức."""
    return RATE_LIMIT_WINDOW_SECONDS / requests_per_minute(has_api_key)

# --- Độ dài lịch sử cần lấy --------------------------------------------------
# VNINDEX là dữ liệu nền dài hạn: đủ cho ROC252, trung bình 49 phiên của Strength
# và phân vị 252 phiên của biến động.
INDEX_HISTORY_YEARS = 8
# Cổ phiếu VN30 chỉ cần đủ để tính MA200 của chính cổ phiếu đó cộng biên an toàn.
# 430 ngày lịch (khoảng 285 phiên) phủ MA200 và lợi suất 20 phiên.
STOCK_HISTORY_CALENDAR_DAYS = 430
# Khi cập nhật tăng dần, lấy chồng lấn để bắt các phiên bị điều chỉnh muộn.
INCREMENTAL_OVERLAP_DAYS = 12
# Số phiên tối thiểu để một mã được coi là đủ lịch sử. Suy ra từ chỉ tiêu khắt
# khe nhất đang tính (MA dài nhất) cộng một biên nhỏ, để bảng chất lượng dữ liệu
# và số mã hợp lệ của breadth không mâu thuẫn nhau.
MIN_SESSIONS_FOR_FULL_HISTORY = 205
# Dưới ngưỡng này thì coi như chưa khởi tạo được dữ liệu VN30.
MIN_STOCKS_FOR_VALID_DATASET = 20
# Chênh lệch ngày dữ liệu giữa các mã vượt ngưỡng này là dữ liệu không đồng bộ.
MAX_STALE_SESSIONS = 5


def index_history_start(today: date | None = None) -> str:
    """Ngày bắt đầu cho lần tải chỉ số đầu tiên."""
    today = today or date.today()
    return (today - timedelta(days=365 * INDEX_HISTORY_YEARS)).isoformat()


def stock_history_start(today: date | None = None) -> str:
    """Ngày bắt đầu cho lần tải cổ phiếu đầu tiên."""
    today = today or date.today()
    return (today - timedelta(days=STOCK_HISTORY_CALENDAR_DAYS)).isoformat()

# --- Ghi log -----------------------------------------------------------------
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_LEVEL = "INFO"

# --- Trend / RORO ------------------------------------------------------------
RORO_HORIZONS = ((63, 0.4), (126, 0.2), (189, 0.2), (252, 0.2))
RORO_BASELINE_WINDOW = 49
# Vùng trung tính = RORO_NEUTRAL_SIGMA * độ lệch chuẩn 252 phiên của chính chuỗi RORO.
RORO_NEUTRAL_SIGMA = 0.5
RORO_SIGMA_WINDOW = 252

# --- Stress ------------------------------------------------------------------
PARKINSON_WINDOW = 22
ANNUALIZATION = 252
STRESS_PERCENTILE_WINDOW = 252
STRESS_MIN_PERIODS = 120
# Ngưỡng là phân vị trên chính phân phối lịch sử của VNINDEX, không phải số tuyệt đối.
STRESS_BANDS = ((20.0, "THẤP"), (60.0, "BÌNH THƯỜNG"), (85.0, "CAO"))
STRESS_TOP_LABEL = "RẤT CAO"
STRESS_EMA_SPAN = 5

# --- Breadth -----------------------------------------------------------------
BREADTH_MA_WINDOWS = (20, 50, 200)
BREADTH_RETURN_WINDOWS = (1, 5, 20)
BREADTH_MIN_VALID_SYMBOLS = 20
BREADTH_BANDS = ((30.0, "RẤT YẾU"), (45.0, "YẾU"), (55.0, "CÂN BẰNG"), (70.0, "KHỎE"))
BREADTH_TOP_LABEL = "RẤT KHỎE"

# --- Dispersion / Concentration ---------------------------------------------
DISPERSION_PRIMARY_WINDOW = 20
DISPERSION_CONTEXT_SESSIONS = 252
VOLATILITY_WINDOW = 20
CONCENTRATION_TOP_N = (5, 10)

# Phân vị dùng để mô tả mức cao/thấp của phân hóa và tập trung rủi ro.
CONTEXT_LOW_PERCENTILE = 20.0
CONTEXT_HIGH_PERCENTILE = 80.0

# --- Màu sắc UI --------------------------------------------------------------
COLOR_GOOD = "#0f7b52"
COLOR_BAD = "#c0392f"
COLOR_WARN = "#b7791f"
COLOR_MUTED = "#6b7280"
