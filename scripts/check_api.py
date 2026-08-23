"""Kiểm tra kết nối tới nguồn dữ liệu vnstock.

Chạy hai phép thử tối thiểu: một chỉ số (VNINDEX) và một cổ phiếu (FPT).
Nếu hai phép thử này không chạy được thì không nên gọi tiếp 30 mã còn lại.

    python scripts/check_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.vnstock_data import connectivity_check, fetch_vn30_constituents, vnstock_version


def main() -> int:
    print(f"vnstock {vnstock_version()} · nguồn chính {config.PRIMARY_SOURCE}")
    print("API đang dùng: Quote(symbol=..., source=...).history(start=..., end=..., interval='1D')")
    print("-" * 72)

    failures = 0
    for row in connectivity_check():
        if row["ok"]:
            print(f"[OK]   {row['symbol']:<8} nguồn {row['source']:<4} {row['rows']:>4} phiên, đến {row['last_date']}")
        else:
            failures += 1
            print(f"[LỖI]  {row['symbol']:<8} {row['kind']}: {row['message']}")

    print("-" * 72)
    try:
        symbols = fetch_vn30_constituents()
        print(f"[OK]   VN30 hiện có {len(symbols)} mã: {', '.join(symbols)}")
    except Exception as exc:
        failures += 1
        print(f"[LỖI]  Danh sách VN30: {type(exc).__name__}: {exc}")

    if failures:
        print(f"\n{failures} phép thử thất bại. Không nên chạy cập nhật toàn bộ khi chưa xử lý xong.")
        return 1
    print("\nToàn bộ phép thử đạt. Có thể chạy scripts/update_data.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
