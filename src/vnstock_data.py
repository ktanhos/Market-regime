"""Lớp cô lập vnstock. Đây là module duy nhất được phép import vnstock.

API đang dùng (vnstock 4.x, giao diện chính thức hiện hành)::

    from vnstock import Quote, Listing
    Quote(symbol="VNINDEX", source="VCI").history(start="2015-01-01", end="2026-08-22", interval="1D")
    Listing(source="VCI").symbols_by_group("VN30")

Những API KHÔNG dùng và lý do:

* ``Vnstock().stock(symbol=..., source=...)`` – lối vào cũ, chỉ còn là vỏ bọc.
* ``Market(source=...)`` – không phải giao diện lấy giá lịch sử.
* ``Quote.ohlcv`` – trong vnstock 4.x đây chỉ là bí danh của ``Quote.history``
  (``ohlcv = history`` trong ``vnstock/api/quote.py``). Gọi lần lượt cả hai tên
  như mã cũ chỉ nhân đôi số lượt gọi API chứ không tạo thêm cơ hội thành công.

Giới hạn của nguồn dữ liệu, đọc trực tiếp từ gói đã cài:

* KBS (``vnstock/explorer/kbs/const.py::_INDEX_MAPPING``) chỉ hỗ trợ VNINDEX,
  HNXINDEX, UPCOMINDEX. KBS **không** lấy được chỉ số VN30.
* VCI (``vnstock/explorer/vci/const.py::_VCI_INDEX_MAPPING``) hỗ trợ VN30.
* Cả hai nguồn trả về cột ``time/open/high/low/close/volume``.
* Không nguồn nào có endpoint lấy lịch sử nhiều mã trong một lần gọi.
  ``Trading.price_board`` nhận danh sách mã nhưng chỉ trả bảng giá tại thời
  điểm hiện tại, không có lịch sử, nên không thay thế được vòng lặp tuần tự.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable, Sequence

import pandas as pd

from src import config
from src.schema import DataQualityError, standardize_ohlcv

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
    except Exception:
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
    if asset_type == "index":
        return config.INDEX_SOURCES.get(symbol, (config.PRIMARY_SOURCE,))
    return (config.PRIMARY_SOURCE,) + tuple(config.FALLBACK_SOURCES)


@dataclass
class FetchResult:
    symbol: str
    frame: pd.DataFrame
    source: str
    attempts: int


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
    asset_type: str = "stock",
    sources: Sequence[str] | None = None,
    max_attempts: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    caller: Callable[..., pd.DataFrame] | None = None,
) -> FetchResult:
    """Lấy lịch sử giá của một mã, tuần tự, có backoff, không retry vô hạn.

    ``caller`` chỉ dùng cho kiểm thử: nó thay thế lời gọi mạng thật.
    """
    symbol = symbol.upper()
    end = end or date.today().isoformat()
    sources = tuple(sources or sources_for(symbol, asset_type))
    max_attempts = max_attempts or config.MAX_ATTEMPTS_PER_SOURCE
    call = caller or _single_call

    errors: list[str] = []
    attempts = 0

    for source in sources:
        for attempt in range(1, max_attempts + 1):
            attempts += 1
            try:
                frame = call(symbol, source, start, end)
                if frame.empty:
                    raise DataQualityError("Nguồn trả về 0 dòng")
                return FetchResult(symbol=symbol, frame=frame, source=source, attempts=attempts)
            except FetchError:
                raise
            except Exception as exc:
                kind = classify_error(exc)
                errors.append(f"{source}: {type(exc).__name__}: {str(exc)[:200]}")
                if kind == RATE_LIMITED:
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


def default_start(asset_type: str, last_date: pd.Timestamp | None = None) -> str:
    """Ngày bắt đầu cần yêu cầu từ API.

    * Chưa có dữ liệu: chỉ số lấy từ ``INDEX_HISTORY_START`` để đủ ROC252 và
      z-score 252 phiên; cổ phiếu chỉ lấy đủ để tính MA200 của chính nó.
    * Đã có dữ liệu: lấy từ ngày cuối cùng lùi lại ``INCREMENTAL_OVERLAP_DAYS``.
    """
    if last_date is not None and pd.notna(last_date):
        start = pd.Timestamp(last_date) - timedelta(days=config.INCREMENTAL_OVERLAP_DAYS)
        return start.strftime("%Y-%m-%d")
    if asset_type == "index":
        return config.INDEX_HISTORY_START
    return (date.today() - timedelta(days=config.STOCK_HISTORY_CALENDAR_DAYS)).isoformat()


def fetch_vn30_constituents(source: str = config.PRIMARY_SOURCE) -> list[str]:
    """Danh sách VN30 **tại thời điểm gọi**.

    Đây là ảnh chụp hiện tại. Nó không nói gì về thành phần VN30 trong quá khứ
    và không được dùng để tái tạo lịch sử rổ VN30.
    """
    Listing = _listing_class()
    try:
        series = Listing(source=source).symbols_by_group("VN30")
    except Exception as exc:
        raise FetchError(
            f"Không lấy được danh sách VN30: {type(exc).__name__}: {str(exc)[:200]}",
            classify_error(exc),
            symbol="VN30",
            source=source,
        ) from exc

    symbols = sorted({str(s).strip().upper() for s in pd.Series(series).dropna().tolist() if str(s).strip()})
    if len(symbols) < 20:
        raise FetchError(
            f"Danh sách VN30 trả về chỉ có {len(symbols)} mã, không hợp lý.",
            EMPTY,
            symbol="VN30",
            source=source,
        )
    return symbols


def connectivity_check(sleep: Callable[[float], None] = time.sleep) -> list[dict]:
    """Kiểm tra kết nối tối thiểu: một chỉ số và một cổ phiếu.

    Nếu hai phép thử này không chạy được thì không nên gọi tiếp 30 mã còn lại.
    """
    start = (date.today() - timedelta(days=30)).isoformat()
    end = date.today().isoformat()
    checks = [("VNINDEX", "index"), ("FPT", "stock")]
    results: list[dict] = []

    for i, (symbol, asset_type) in enumerate(checks):
        row = {"symbol": symbol, "asset_type": asset_type}
        try:
            result = fetch_history(symbol, start=start, end=end, asset_type=asset_type, sleep=sleep)
            row.update(
                ok=True,
                source=result.source,
                rows=int(len(result.frame)),
                last_date=pd.to_datetime(result.frame["date"]).max().strftime("%Y-%m-%d"),
                message="",
                kind="",
            )
        except FetchError as exc:
            row.update(
                ok=False,
                source=exc.source,
                rows=0,
                last_date="",
                message=str(exc)[:300],
                kind=exc.kind,
            )
        results.append(row)
        if i < len(checks) - 1:
            sleep(config.REQUEST_DELAY_SECONDS)

    return results
