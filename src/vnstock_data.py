"""Lớp cô lập vnstock. Đây là module DUY NHẤT được phép import vnstock.

Giao diện công khai của module, tầng trên chỉ được dùng đúng những hàm này::

    fetch_history(symbol, start, end, asset_type)  -> FetchResult
    fetch_index(symbol, start, end)                -> FetchResult
    fetch_equity(symbol, start, end)               -> FetchResult
    fetch_index_members(index)                     -> list[str]
    connectivity_check()                           -> list[ProbeResult]

``app.py`` không biết gì về cách vnstock hoạt động bên trong.

API đang dùng (vnstock 4.x, giao diện chính thức hiện hành)::

    from vnstock import Quote, Listing
    Quote(symbol="VNINDEX", source="VCI").history(start=..., end=..., interval="1D")
    Listing(source="VCI").symbols_by_group("VN30")

Những API KHÔNG dùng và lý do:

* ``Vnstock().stock(...)`` / ``Vnstock().index(...)`` – lối vào cũ, chỉ là vỏ bọc.
* ``Market(source=...)`` – không phải giao diện lấy giá lịch sử.
* ``Quote.ohlcv`` – trong vnstock 4.x đây chỉ là bí danh của ``Quote.history``
  (``ohlcv = history`` trong ``vnstock/api/quote.py``). Gọi lần lượt cả hai tên
  chỉ nhân đôi số lượt gọi API chứ không tạo thêm cơ hội thành công.

Giới hạn nguồn dữ liệu, đọc trực tiếp từ gói đã cài:

* KBS (``vnstock/explorer/kbs/const.py::_INDEX_MAPPING``) chỉ hỗ trợ VNINDEX,
  HNXINDEX, UPCOMINDEX. KBS **không** lấy được chỉ số VN30.
* VCI (``vnstock/explorer/vci/const.py::_VCI_INDEX_MAPPING``) hỗ trợ VN30.
* Cả hai nguồn trả về cột ``time/open/high/low/close/volume``.
* Không nguồn nào có endpoint lấy lịch sử nhiều mã trong một lần gọi.
  ``Trading.price_board`` nhận danh sách mã nhưng chỉ trả bảng giá tại thời
  điểm hiện tại, không có lịch sử, nên không thay thế được vòng lặp tuần tự.

Hạn mức truy cập, lấy từ chính thông báo của vnstock khi bị chặn::

    Gói Khách (không API key):  20 lượt/phút
    API key miễn phí:           60 lượt/phút

Một lượt khởi tạo cần 33 lượt gọi (1 danh sách + 2 chỉ số + 30 cổ phiếu), nên
phải điều tiết theo phút chứ không chỉ nghỉ giữa hai lượt. ``RateLimiter`` bên
dưới giữ nhịp bằng cửa sổ trượt; nếu vẫn bị chặn thì chờ hết chu kỳ rồi thử lại.
"""

from __future__ import annotations

import os
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator
from datetime import date, timedelta
from typing import Callable, Sequence

import pandas as pd

from src import config
from src import credentials as credentials_module
from src.logging_config import get_logger
from src.schema import CANONICAL_COLUMNS, DataQualityError, standardize_ohlcv

logger = get_logger(__name__)


@dataclass(frozen=True)
class ApiAccess:
    """Mức truy cập thực tế của client trong một lượt chạy.

    ``verified`` chỉ đúng khi client xác nhận nó đang giữ đúng khóa ta đưa vào.
    Hạn mức cao chỉ được áp dụng khi ``verified`` là True: không bao giờ nâng
    lên 60 lượt/phút chỉ vì môi trường có một chuỗi tên VNSTOCK_API_KEY.
    """

    tier: str = "guest"
    observed_limit: int = config.RATE_LIMIT_FALLBACK_GUEST
    effective_limit: int = 0
    configured: bool = False
    verified: bool = False
    source: str = credentials_module.SOURCE_NONE
    note: str = ""

    @property
    def source_label(self) -> str:
        return credentials_module.SOURCE_LABELS.get(self.source, self.source)

    def as_dict(self) -> dict:
        """Chỉ chứa siêu dữ liệu. Không bao giờ chứa giá trị khóa."""
        return {
            "tier": self.tier,
            "observed_limit": self.observed_limit,
            "effective_limit": self.effective_limit,
            "configured": self.configured,
            "verified": self.verified,
            "source": self.source,
            "note": self.note,
        }


def _authenticator():
    """Đối tượng vnstock dùng để quyết định gói và hạn mức."""
    from vnai.beam.auth import authenticator

    return authenticator


def _observed_limit(force_refresh: bool = True) -> tuple[str, int]:
    """Hỏi thẳng client: đang ở gói nào và hạn mức bao nhiêu lượt mỗi phút.

    Đây là nguồn sự thật, không phải phỏng đoán từ biến môi trường.

    ``force_refresh`` là bắt buộc sau khi vừa đổi khóa: ``get_tier`` có bộ nhớ
    đệm theo thời gian, nên nếu không ép làm mới thì nó trả lại gói cũ và hạn
    mức sẽ không bao giờ được nâng lên dù khóa đã được áp dụng.
    """
    try:
        auth = _authenticator()
        tier = str(auth.get_tier(force_refresh=force_refresh))
        limit = int(auth.get_limits(tier).get("min", config.RATE_LIMIT_FALLBACK_GUEST))
        return tier, limit
    except Exception as exc:
        logger.warning("Không đọc được gói truy cập từ vnstock: %s", str(exc)[:200])
        return "unknown", config.RATE_LIMIT_FALLBACK_GUEST


def describe_api_access(credentials: "credentials_module.ApiCredentials | None" = None) -> ApiAccess:
    """Mức truy cập sẽ có nếu chạy ngay bây giờ, KHÔNG thay đổi gì.

    Dùng cho phần hiển thị trạng thái ở thanh bên.
    """
    credentials = credentials or credentials_module.resolve_vnstock_api_key()
    if not credentials.configured:
        tier, limit = _observed_limit(force_refresh=True)
        return ApiAccess(
            tier=tier,
            observed_limit=limit,
            effective_limit=config.effective_rate_limit(limit),
            source=credentials.source,
        )
    # Chưa áp khóa nên chỉ ước lượng theo gói Cộng đồng.
    limit = config.RATE_LIMIT_FALLBACK_WITH_KEY
    return ApiAccess(
        tier="free",
        observed_limit=limit,
        effective_limit=config.effective_rate_limit(limit),
        configured=True,
        verified=False,
        source=credentials.source,
        note="Hạn mức thực tế được xác nhận khi bắt đầu cập nhật.",
    )


@contextmanager
def api_access(
    credentials: "credentials_module.ApiCredentials | None" = None,
) -> "Iterator[ApiAccess]":
    """Cấu hình client với khóa trong suốt một lượt chạy rồi trả lại như cũ.

    Vì sao đặt khóa vào ``os.environ`` chứ không gọi ``register_user``:
    ``vnai.beam.auth.authenticator.get_api_key()`` đọc biến môi trường
    **trước** tệp khóa, nên đây là đường chính thức để client dùng đúng khóa
    này. ``register_user`` thì ghi khóa xuống ``~/.vnstock``, tức là để lại
    khóa của một phiên Streamlit trên đĩa máy chủ — điều không được phép.

    Giá trị cũ được khôi phục khi thoát, nên khóa của một phiên không rò sang
    phiên khác trong cùng tiến trình Streamlit.
    """
    credentials = credentials or credentials_module.resolve_vnstock_api_key()

    if not credentials.configured:
        tier, limit = _observed_limit(force_refresh=True)
        yield ApiAccess(
            tier=tier,
            observed_limit=limit,
            effective_limit=config.effective_rate_limit(limit),
            source=credentials.source,
        )
        return

    if not credentials.usable:
        tier, limit = _observed_limit(force_refresh=True)
        logger.warning(
            "API key từ %s quá ngắn nên bị bỏ qua, chạy ở gói Khách",
            credentials.source_label,
        )
        yield ApiAccess(
            tier=tier,
            observed_limit=limit,
            effective_limit=config.effective_rate_limit(limit),
            configured=True,
            verified=False,
            source=credentials.source,
            note="Khóa quá ngắn nên không được dùng.",
        )
        return

    key_name = config.VNSTOCK_API_KEY_ENV
    previous = os.environ.get(key_name)
    os.environ[key_name] = credentials.key
    try:
        # Xác minh: client có thực sự giữ đúng khóa vừa đưa vào hay không.
        verified = False
        try:
            verified = _authenticator().get_api_key() == credentials.key
        except Exception as exc:
            logger.warning("Không xác minh được API key: %s", str(exc)[:200])

        if not verified:
            tier, limit = _observed_limit(force_refresh=True)
            logger.warning("Client không nhận API key, giữ hạn mức gói Khách")
            yield ApiAccess(
                tier=tier,
                observed_limit=config.RATE_LIMIT_FALLBACK_GUEST,
                effective_limit=config.effective_rate_limit(config.RATE_LIMIT_FALLBACK_GUEST),
                configured=True,
                verified=False,
                source=credentials.source,
                note="Client không nhận khóa nên vẫn dùng hạn mức gói Khách.",
            )
            return

        tier, limit = _observed_limit(force_refresh=True)
        effective = config.effective_rate_limit(limit)
        logger.info(
            "API key từ %s đã được áp dụng · gói %s · hạn mức %d, vận hành %d lượt/phút",
            credentials.source_label, tier, limit, effective,
        )
        yield ApiAccess(
            tier=tier,
            observed_limit=limit,
            effective_limit=effective,
            configured=True,
            verified=True,
            source=credentials.source,
        )
    finally:
        if previous is None:
            os.environ.pop(key_name, None)
        else:
            os.environ[key_name] = previous
        # Bỏ bộ nhớ đệm gói để lần hỏi sau phản ánh đúng trạng thái đã khôi phục.
        _observed_limit(force_refresh=True)


class RateLimiter:
    """Điều tiết số lượt gọi theo cửa sổ trượt một phút.

    Nghỉ cố định giữa hai lượt là không đủ: hạn mức của nguồn tính theo phút,
    nên phải đếm số lượt trong cửa sổ và chờ đúng lúc cần chờ.
    """

    def __init__(
        self,
        max_calls: int | None = None,
        window: float = config.RATE_LIMIT_WINDOW_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_calls = max_calls or config.requests_per_minute(False)
        self.window = window
        self._sleep = sleep
        self._clock = clock
        self._calls: deque[float] = deque()

    def acquire(self) -> float:
        """Chờ tới khi được phép gọi. Trả về số giây đã chờ."""
        now = self._clock()
        while self._calls and now - self._calls[0] >= self.window:
            self._calls.popleft()

        waited = 0.0
        if len(self._calls) >= self.max_calls:
            wait = self.window - (now - self._calls[0]) + 0.05
            if wait > 0:
                logger.info("Điều tiết truy cập: chờ %.1f giây", wait)
                self._sleep(wait)
                waited = wait
                now = self._clock()
                while self._calls and now - self._calls[0] >= self.window:
                    self._calls.popleft()

        self._calls.append(now)
        return waited

    def downgrade(self, max_calls: int | None = None) -> int:
        """Hạ hạn mức khi nguồn vẫn từ chối dù ta đã điều tiết.

        Nghĩa là ước lượng gói truy cập đã sai — ví dụ khóa không được nguồn
        chấp nhận. Không được tiếp tục giả định hạn mức cao.
        """
        target = max_calls or config.requests_per_minute(False)
        if target < self.max_calls:
            logger.warning(
                "Hạ hạn mức vận hành từ %d xuống %d lượt/phút", self.max_calls, target
            )
            self.max_calls = target
        return self.max_calls

    def cool_down(self, seconds: float | None = None) -> None:
        """Chờ hết một chu kỳ sau khi bị nguồn từ chối."""
        seconds = seconds if seconds is not None else config.RATE_LIMIT_COOLDOWN_SECONDS
        logger.warning("Bị giới hạn truy cập, chờ %.0f giây rồi thử lại", seconds)
        self._sleep(seconds)
        self._calls.clear()


ASSET_INDEX = "index"
ASSET_STOCK = "stock"

# --- Phân loại lỗi -----------------------------------------------------------

TRANSIENT = "transient"        # nên thử lại
RATE_LIMITED = "rate_limited"  # dừng lại, không thử lại
PERMANENT = "permanent"        # sai mã / nguồn không hỗ trợ, thử lại vô nghĩa
EMPTY = "empty"                # gọi được nhưng không có dữ liệu
DEPENDENCY = "dependency"      # thiếu thư viện

_RATE_LIMIT_HINTS = (
    "429", "too many requests", "rate limit", "quá nhiều", "giới hạn",
    "vượt quá", "quota", "throttle",
)
_TRANSIENT_HINTS = (
    "timeout", "timed out", "connection", "connectionerror", "temporarily",
    "502", "503", "504", "reset by peer", "retryerror", "max retries",
    "remote end closed", "ssl",
)
_PERMANENT_HINTS = (
    "không được hỗ trợ", "not supported", "invalid group", "không hợp lệ",
    "chỉ nhận giá trị tham số", "unknown symbol", "404",
)


class FetchError(RuntimeError):
    """Lỗi khi lấy dữ liệu, kèm phân loại để tầng trên quyết định hành vi."""

    def __init__(self, message: str, kind: str, symbol: str = "", source: str = ""):
        super().__init__(message)
        self.kind = kind
        self.symbol = symbol
        self.source = source

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "source": self.source,
            "kind": self.kind,
            "message": str(self),
        }


def classify_error(exc: BaseException) -> str:
    """Xếp loại lỗi để quyết định thử lại, đổi nguồn hay dừng hẳn."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if isinstance(exc, DataQualityError):
        return EMPTY
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return DEPENDENCY
    for hint in _RATE_LIMIT_HINTS:
        if hint in text:
            return RATE_LIMITED
    for hint in _PERMANENT_HINTS:
        if hint in text:
            return PERMANENT
    for hint in _TRANSIENT_HINTS:
        if hint in text:
            return TRANSIENT
    if isinstance(exc, ValueError):
        return PERMANENT
    return TRANSIENT


def friendly_message(kind: str) -> str:
    return {
        RATE_LIMITED: "API đang giới hạn số lượt truy cập. Hãy thử lại sau.",
        TRANSIENT: "Kết nối tới nguồn dữ liệu không ổn định.",
        PERMANENT: "Nguồn dữ liệu không hỗ trợ mã này.",
        EMPTY: "Nguồn trả về dữ liệu rỗng cho khoảng thời gian yêu cầu.",
        DEPENDENCY: "Thiếu thư viện vnstock trong môi trường đang chạy.",
    }.get(kind, "Lỗi không xác định khi lấy dữ liệu.")


# --- Truy cập vnstock --------------------------------------------------------

def vnstock_version() -> str:
    try:
        from importlib.metadata import version

        return version("vnstock")
    except Exception as exc:
        logger.warning("Không đọc được phiên bản vnstock: %s", exc)
        return "không xác định"


def _quote_class():
    try:
        from vnstock import Quote
    except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
        raise FetchError(
            f"Không import được vnstock.Quote: {type(exc).__name__}: {exc}",
            DEPENDENCY,
        ) from exc
    return Quote


def _listing_class():
    try:
        from vnstock import Listing
    except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
        raise FetchError(
            f"Không import được vnstock.Listing: {type(exc).__name__}: {exc}",
            DEPENDENCY,
        ) from exc
    return Listing


def sources_for(symbol: str, asset_type: str) -> tuple[str, ...]:
    """Thứ tự nguồn dữ liệu cho một mã, theo khả năng hỗ trợ thực tế."""
    symbol = symbol.upper()
    if asset_type == ASSET_INDEX:
        return config.INDEX_SOURCES.get(symbol, (config.PRIMARY_SOURCE,))
    return (config.PRIMARY_SOURCE,) + tuple(config.FALLBACK_SOURCES)


@dataclass
class FetchResult:
    """Kết quả một lần lấy dữ liệu thành công."""

    symbol: str
    frame: pd.DataFrame
    source: str
    attempts: int

    @property
    def rows(self) -> int:
        return int(len(self.frame))

    @property
    def first_date(self) -> str:
        return pd.to_datetime(self.frame["date"]).min().strftime("%Y-%m-%d")

    @property
    def last_date(self) -> str:
        return pd.to_datetime(self.frame["date"]).max().strftime("%Y-%m-%d")

    @property
    def schema(self) -> list[str]:
        return list(self.frame.columns)


@dataclass
class ProbeResult:
    """Kết quả một phép thử kết nối, dùng cho màn hình chẩn đoán API."""

    name: str
    ok: bool
    source: str = ""
    rows: int = 0
    first_date: str = ""
    last_date: str = ""
    schema: list[str] = field(default_factory=list)
    detail: str = ""
    kind: str = ""
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "Mục kiểm tra": self.name,
            "Kết quả": "SUCCESS" if self.ok else "FAILED",
            "Nguồn": self.source,
            "Số dòng": self.rows,
            "Ngày đầu": self.first_date,
            "Ngày cuối": self.last_date,
            "Schema": ", ".join(self.schema),
            "Lỗi": self.error,
        }


def _single_call(symbol: str, source: str, start: str, end: str) -> pd.DataFrame:
    """Đúng một lời gọi tới giao diện hiện hành của vnstock."""
    Quote = _quote_class()
    quote = Quote(symbol=symbol, source=source)
    raw = quote.history(start=start, end=end, interval="1D")
    return standardize_ohlcv(raw)


def fetch_history(
    symbol: str,
    start: str,
    end: str | None = None,
    asset_type: str = ASSET_STOCK,
    sources: Sequence[str] | None = None,
    max_attempts: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    caller: Callable[..., pd.DataFrame] | None = None,
    limiter: "RateLimiter | None" = None,
) -> FetchResult:
    """Lấy lịch sử giá của một mã: tuần tự, có điều tiết, không thử lại vô hạn.

    Thứ tự xử lý lỗi:

    * ``rate_limited``          -> chờ hết chu kỳ rồi thử lại, tối đa
      ``MAX_RATE_LIMIT_RETRIES`` lần; hết lượt mới báo lỗi lên tầng trên.
    * ``permanent``/``dependency`` -> chuyển sang nguồn kế tiếp, không lặp lại nguồn cũ.
    * ``transient``/``empty``   -> thử lại tối đa ``max_attempts`` lần với backoff mũ.

    ``caller`` chỉ dùng cho kiểm thử: nó thay thế lời gọi mạng thật.
    """
    symbol = symbol.upper()
    end = end or date.today().isoformat()
    sources = tuple(sources or sources_for(symbol, asset_type))
    max_attempts = max_attempts or config.MAX_ATTEMPTS_PER_SOURCE
    call = caller or _single_call

    errors: list[str] = []
    attempts = 0
    rate_limit_retries = 0

    for source in sources:
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            attempts += 1
            if limiter is not None:
                limiter.acquire()
            try:
                frame = call(symbol, source, start, end)
                if frame.empty:
                    raise DataQualityError("Nguồn trả về 0 dòng")
                logger.info(
                    "Lấy %s từ %s: %d dòng (%s lượt gọi)", symbol, source, len(frame), attempts
                )
                return FetchResult(symbol=symbol, frame=frame, source=source, attempts=attempts)
            except FetchError:
                raise
            except Exception as exc:
                kind = classify_error(exc)
                errors.append(f"{source}: {type(exc).__name__}: {str(exc)[:200]}")
                logger.warning("Lỗi lấy %s từ %s [%s]: %s", symbol, source, kind, str(exc)[:200])
                if kind == RATE_LIMITED:
                    if rate_limit_retries < config.MAX_RATE_LIMIT_RETRIES:
                        rate_limit_retries += 1
                        if limiter is not None:
                            # Bị chặn dù đã điều tiết nghĩa là hạn mức ta tưởng
                            # đang có không đúng. Hạ về mức an toàn rồi mới thử lại.
                            limiter.downgrade()
                            limiter.cool_down()
                        else:
                            sleep(config.RATE_LIMIT_COOLDOWN_SECONDS)
                        attempt -= 1   # lần chờ không tính là một lượt thử của nguồn
                        continue
                    raise FetchError(
                        f"{friendly_message(RATE_LIMITED)} ({source})",
                        RATE_LIMITED,
                        symbol=symbol,
                        source=source,
                    ) from exc
                if kind in (PERMANENT, DEPENDENCY):
                    break  # đổi nguồn, thử lại cùng nguồn là vô nghĩa
                if attempt < max_attempts:
                    delay = min(
                        config.BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                        config.BACKOFF_MAX_SECONDS,
                    )
                    sleep(delay)

    kind = TRANSIENT if errors else EMPTY
    raise FetchError(
        f"Không lấy được dữ liệu {symbol}. " + " | ".join(errors[-3:]),
        kind,
        symbol=symbol,
        source=",".join(sources),
    )


def make_limiter(
    sleep: Callable[[float], None] = time.sleep,
    access: ApiAccess | None = None,
) -> RateLimiter:
    """Bộ điều tiết DUY NHẤT cho cả một lượt chạy.

    Hạn mức lấy từ mức truy cập đã được xác minh, không phải từ sự tồn tại của
    một biến môi trường.
    """
    max_calls = access.effective_limit if access is not None else None
    return RateLimiter(max_calls=max_calls, sleep=sleep)


def fetch_index(symbol: str, start: str, end: str | None = None, **kwargs) -> FetchResult:
    """Lịch sử giá của một chỉ số (VNINDEX, VN30...)."""
    return fetch_history(symbol, start=start, end=end, asset_type=ASSET_INDEX, **kwargs)


def fetch_equity(symbol: str, start: str, end: str | None = None, **kwargs) -> FetchResult:
    """Lịch sử giá của một cổ phiếu."""
    return fetch_history(symbol, start=start, end=end, asset_type=ASSET_STOCK, **kwargs)


def default_start(asset_type: str, last_date: pd.Timestamp | None = None) -> str:
    """Ngày bắt đầu cần yêu cầu từ API.

    * Chưa có dữ liệu: chỉ số lấy 8 năm để đủ ROC252 và phân vị 252 phiên;
      cổ phiếu chỉ lấy 430 ngày, vừa đủ tính MA200 của chính nó.
    * Đã có dữ liệu: lấy từ ngày cuối cùng lùi lại ``INCREMENTAL_OVERLAP_DAYS``
      để bắt các phiên bị điều chỉnh muộn, không tải lại toàn bộ lịch sử.
    """
    if last_date is not None and pd.notna(last_date):
        start = pd.Timestamp(last_date) - timedelta(days=config.INCREMENTAL_OVERLAP_DAYS)
        return start.strftime("%Y-%m-%d")
    if asset_type == ASSET_INDEX:
        return config.index_history_start()
    return config.stock_history_start()


def fetch_index_members(index: str = "VN30", source: str = config.PRIMARY_SOURCE) -> list[str]:
    """Danh sách thành phần của một chỉ số **tại thời điểm gọi**.

    Đây là ảnh chụp hiện tại. Nó không nói gì về thành phần của chỉ số trong quá
    khứ và không được dùng để tái tạo lịch sử rổ.

    Gọi: Listing(source).symbols_by_group(index)
    """
    Listing = _listing_class()
    try:
        logger.info("Đang lấy danh sách %s từ %s...", index, source)
        series = Listing(source=source).symbols_by_group(index)
    except Exception as exc:
        kind = classify_error(exc)
        logger.warning(
            "Không lấy được danh sách %s từ %s [%s]: %s",
            index, source, kind, str(exc)[:200],
        )
        raise FetchError(
            f"Không lấy được danh sách {index} từ {source}: {type(exc).__name__}: {str(exc)[:200]}",
            kind,
            symbol=index,
            source=source,
        ) from exc

    symbols = sorted(
        {str(s).strip().upper() for s in pd.Series(series).dropna().tolist() if str(s).strip()}
    )
    if len(symbols) < 20:
        logger.error("Danh sách %s từ %s chỉ trả về %d mã (kỳ vọng ~30)", index, source, len(symbols))
        raise FetchError(
            f"Danh sách {index} từ {source} chỉ có {len(symbols)} mã, không hợp lý (kỳ vọng ~30).",
            EMPTY,
            symbol=index,
            source=source,
        )
    logger.info("Danh sách %s lấy được %d mã từ %s", index, len(symbols), source)
    return symbols


def connectivity_check(
    sleep: Callable[[float], None] = time.sleep,
    limiter: "RateLimiter | None" = None,
) -> list[ProbeResult]:
    """Chẩn đoán API: một chỉ số, một cổ phiếu và danh sách thành phần VN30.

    Nếu ba phép thử này không chạy được thì không nên gọi tiếp 30 mã còn lại.
    """
    start = (date.today() - timedelta(days=30)).isoformat()
    end = date.today().isoformat()
    results: list[ProbeResult] = []
    # Dùng lại bộ điều tiết của tầng gọi nếu có, để ba phép thử này nằm chung
    # một ngân sách với pipeline thay vì tiêu thêm hạn mức riêng.
    limiter = limiter if limiter is not None else RateLimiter(sleep=sleep)

    for symbol, asset_type in (("VNINDEX", ASSET_INDEX), ("FPT", ASSET_STOCK)):
        try:
            result = fetch_history(
                symbol, start=start, end=end, asset_type=asset_type, sleep=sleep, limiter=limiter
            )
            results.append(
                ProbeResult(
                    name=symbol,
                    ok=True,
                    source=result.source,
                    rows=result.rows,
                    first_date=result.first_date,
                    last_date=result.last_date,
                    schema=result.schema,
                    detail=f"{result.rows} phiên tới {result.last_date}",
                )
            )
        except FetchError as exc:
            results.append(
                ProbeResult(name=symbol, ok=False, source=exc.source, kind=exc.kind,
                            error=str(exc)[:300], detail=friendly_message(exc.kind))
            )

    try:
        limiter.acquire()
        members = fetch_index_members("VN30")
        results.append(
            ProbeResult(
                name="VN30 Universe",
                ok=True,
                source=config.PRIMARY_SOURCE,
                rows=len(members),
                schema=["symbol"],
                detail=", ".join(members),
            )
        )
    except FetchError as exc:
        results.append(
            ProbeResult(name="VN30 Universe", ok=False, source=exc.source, kind=exc.kind,
                        error=str(exc)[:300], detail=friendly_message(exc.kind))
        )

    return results


def expected_schema() -> list[str]:
    """Lược đồ mà mọi khung dữ liệu trả về phải tuân theo."""
    return list(CANONICAL_COLUMNS)
