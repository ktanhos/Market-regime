"""Chạy toàn bộ pipeline cập nhật dữ liệu ngoài Streamlit.

    python scripts/update_data.py                 # cập nhật đầy đủ
    python scripts/update_data.py --no-universe   # giữ nguyên danh sách VN30 đã lưu
    python scripts/update_data.py --features-only # chỉ tính lại chỉ tiêu, không gọi API
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import universe as universe_module
from src.updater import rebuild_features, run_update


def main() -> int:
    parser = argparse.ArgumentParser(description="Cập nhật dữ liệu thị trường")
    parser.add_argument("--no-universe", action="store_true", help="Không gọi API lấy lại danh sách VN30")
    parser.add_argument("--features-only", action="store_true", help="Chỉ tính lại chỉ tiêu từ dữ liệu đã lưu")
    args = parser.parse_args()

    if args.features_only:
        snapshot = rebuild_features(universe_module.symbols())
        print(f"Đã tính lại chỉ tiêu. Dữ liệu đến ngày {snapshot.get('as_of')}.")
        return 0

    def progress(phase: str, ratio: float, message: str) -> None:
        print(f"[{ratio * 100:5.1f}%] {message}")

    report = run_update(refresh_universe=not args.no_universe, progress=progress)
    print("-" * 72)
    print(f"Thành công {report.success_count}/{report.total_count} nguồn.")
    if report.rate_limited:
        print(f"CẢNH BÁO: {report.aborted_reason}")
    for failure in report.failures:
        print(f"  LỖI {failure['symbol']:<8} [{failure['kind']}] {failure['message'][:160]}")
    return 0 if report.success_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
