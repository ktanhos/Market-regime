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

from src import features, universe as universe_module
from src.logging_config import setup
from src.updater import PHASE_LABELS, SYNC_VIA_CI, record_sync, run_update


def main() -> int:
    parser = argparse.ArgumentParser(description="Cập nhật dữ liệu thị trường")
    parser.add_argument("--no-universe", action="store_true", help="Không gọi API lấy lại danh sách VN30")
    parser.add_argument("--features-only", action="store_true", help="Chỉ tính lại chỉ tiêu từ dữ liệu đã lưu")
    args = parser.parse_args()

    setup()
    if args.features_only:
        snapshot = features.rebuild(universe_module.symbols())
        print(f"Đã tính lại chỉ tiêu. Dữ liệu đến ngày {snapshot.get('as_of')}.")
        return 0

    def progress(phase: str, ratio: float, message: str) -> None:
        print(f"[{PHASE_LABELS.get(phase, phase):<14}{ratio * 100:5.1f}%] {message}")

    report = run_update(refresh_universe=not args.no_universe, progress=progress)

    print("-" * 78)
    print(f"Chế độ              {report.mode}" + (" (khởi tạo lần đầu)" if report.first_run else ""))
    print(f"Danh sách VN30      {report.universe.get('status', '-')} "
          f"({report.universe.get('as_of', '-')})")
    print(f"Chỉ số              {report.index_success}/{report.index_total} thành công")
    print(f"Cổ phiếu VN30       {report.stock_success}/{report.stock_total} mã thành công")
    print(f"Tệp dữ liệu         {report.files_written}/{report.files_expected}")
    if report.stock_missing:
        print(f"Chưa có tệp         {', '.join(report.stock_missing)}")
    if report.rate_limited:
        print(f"CẢNH BÁO            {report.aborted_reason}")
    for failure in report.failures:
        print(f"  LỖI {failure['symbol']:<10} [{failure['kind']}] {failure['message'][:150]}")

    # Chạy ngoài Streamlit thì bước commit của workflow mới là nơi lưu dữ liệu.
    # Ghi lại điều đó để nhật ký không báo "chưa đồng bộ" một cách sai lệch.
    record_sync(report, SYNC_VIA_CI, report.files_written,
                "Dữ liệu được commit bởi workflow, không qua sync_files.")

    if not report.data_complete:
        print("\nDữ liệu chưa đầy đủ.")
        return 1
    print("\nDữ liệu đầy đủ.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
