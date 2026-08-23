"""Lược đồ dữ liệu chuẩn và các phép kiểm tra chất lượng.

Mọi dữ liệu giá đi qua hệ thống đều có đúng một lược đồ:

    date (datetime64[ns]) | open | high | low | close | volume

Đây là điểm mà kiến trúc cũ bị hỏng: lớp ghi dùng cột ``date`` còn lớp đọc dùng
cột ``time`` nên dữ liệu ghi thành công vẫn không đọc được.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

DATE_COLUMN = "date"
PRICE_COLUMNS = ("open", "high", "low", "close")
NUMERIC_COLUMNS = PRICE_COLUMNS + ("volume",)
CANONICAL_COLUMNS = (DATE_COLUMN,) + NUMERIC_COLUMNS

# Tất cả tên cột từng gặp ở các nguồn: vnstock trả về "time", dữ liệu cũ trong
# repository dùng "date", một số nguồn khác dùng "tradingDate"/"TradingDate".
_ALIASES = {
    "time": DATE_COLUMN,
    "date": DATE_COLUMN,
    "datetime": DATE_COLUMN,
    "tradingdate": DATE_COLUMN,
    "trading_date": DATE_COLUMN,
    "timestamp": DATE_COLUMN,
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "nmvolume": "volume",
}


class DataQualityError(ValueError):
    """Dữ liệu không đạt yêu cầu tối thiểu để đưa vào kho."""


@dataclass
class FrameReport:
    """Mô tả một khung dữ liệu sau khi chuẩn hóa."""

    rows: int = 0
    first_date: pd.Timestamp | None = None
    last_date: pd.Timestamp | None = None
    duplicate_dates: int = 0
    missing_close: int = 0
    invalid_hl: int = 0
    non_positive: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "first_date": None if self.first_date is None else self.first_date.strftime("%Y-%m-%d"),
            "last_date": None if self.last_date is None else self.last_date.strftime("%Y-%m-%d"),
            "duplicate_dates": self.duplicate_dates,
            "missing_close": self.missing_close,
            "invalid_hl": self.invalid_hl,
            "non_positive": self.non_positive,
            "warnings": list(self.warnings),
        }


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Đưa một khung dữ liệu bất kỳ về lược đồ chuẩn.

    Sắp xếp theo ngày, bỏ trùng ngày (giữ bản ghi mới nhất), ép kiểu số.
    Không tự bịa dữ liệu: cột thiếu được để trống thay vì điền giá trị giả.
    """
    if df is None:
        raise DataQualityError("Nguồn trả về None thay vì bảng dữ liệu")
    if not isinstance(df, pd.DataFrame):
        raise DataQualityError(f"Nguồn trả về kiểu {type(df).__name__} thay vì DataFrame")
    if df.empty:
        raise DataQualityError("Nguồn trả về bảng rỗng")

    out = df.copy()
    rename: dict = {}
    for column in out.columns:
        key = str(column).strip().lower().replace(" ", "")
        target = _ALIASES.get(key)
        if target and target not in rename.values():
            rename[column] = target
    out = out.rename(columns=rename)

    if DATE_COLUMN not in out.columns:
        raise DataQualityError(
            f"Không tìm thấy cột ngày. Các cột nhận được: {list(df.columns)}"
        )

    out = out.loc[:, [c for c in CANONICAL_COLUMNS if c in out.columns]].copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce")
    out = out.dropna(subset=[DATE_COLUMN])
    # Dữ liệu ngày: bỏ phần giờ để hai nguồn khác nhau khớp được với nhau.
    out[DATE_COLUMN] = out[DATE_COLUMN].dt.normalize()

    for column in NUMERIC_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    if "close" not in out.columns:
        raise DataQualityError("Thiếu cột giá đóng cửa")

    out = out.dropna(subset=["close"])
    if out.empty:
        raise DataQualityError("Không còn dòng nào có giá đóng cửa hợp lệ")

    out = out.sort_values(DATE_COLUMN).drop_duplicates(DATE_COLUMN, keep="last")
    return out.reset_index(drop=True)


def merge_history(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Hợp nhất dữ liệu cũ và mới, ưu tiên bản ghi mới cho ngày trùng nhau."""
    new = standardize_ohlcv(new)
    if old is None or old.empty:
        return new
    old = standardize_ohlcv(old)
    combined = pd.concat([old, new], ignore_index=True)
    return standardize_ohlcv(combined)


def inspect_frame(df: pd.DataFrame) -> FrameReport:
    """Thống kê chất lượng của một khung dữ liệu đã chuẩn hóa."""
    report = FrameReport()
    if df is None or df.empty:
        report.warnings.append("Khung dữ liệu rỗng")
        return report

    dates = pd.to_datetime(df[DATE_COLUMN])
    report.rows = int(len(df))
    report.first_date = dates.min()
    report.last_date = dates.max()
    report.duplicate_dates = int(dates.duplicated().sum())
    report.missing_close = int(df["close"].isna().sum())

    if {"high", "low"}.issubset(df.columns):
        pair = df[["high", "low"]].dropna()
        report.invalid_hl = int((pair["high"] < pair["low"]).sum())
    if "close" in df.columns:
        report.non_positive = int((df["close"].dropna() <= 0).sum())

    if report.duplicate_dates:
        report.warnings.append(f"{report.duplicate_dates} ngày bị trùng")
    if report.invalid_hl:
        report.warnings.append(f"{report.invalid_hl} phiên có high < low")
    if report.non_positive:
        report.warnings.append(f"{report.non_positive} phiên có giá đóng cửa <= 0")
    return report


def validate_frame(df: pd.DataFrame, min_rows: int = 1) -> FrameReport:
    """Kiểm tra dữ liệu trước khi ghi. Ném lỗi nếu không dùng được."""
    report = inspect_frame(df)
    if report.rows < min_rows:
        raise DataQualityError(f"Chỉ có {report.rows} dòng, cần tối thiểu {min_rows}")
    if report.non_positive:
        raise DataQualityError(f"{report.non_positive} phiên có giá đóng cửa không dương")
    if report.invalid_hl:
        raise DataQualityError(f"{report.invalid_hl} phiên có high < low")
    return report


def to_close_panel(series_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Ghép giá đóng cửa của nhiều mã thành một bảng theo ngày.

    Không ffill toàn cục: một mã ngừng giao dịch phải hiện ra là thiếu dữ liệu
    thay vì được kéo dài giá cũ và bị tính như một mã bình thường.
    """
    columns = {}
    for symbol, frame in series_by_symbol.items():
        if frame is None or frame.empty:
            continue
        s = frame.set_index(DATE_COLUMN)["close"]
        columns[symbol] = s[~s.index.duplicated(keep="last")]
    if not columns:
        return pd.DataFrame()
    panel = pd.DataFrame(columns).sort_index()
    return panel.astype("float64")
