"""Kiểm tra kho dữ liệu. Không gọi API.

Dùng cả ở dòng lệnh lẫn trong GitHub Actions: workflow phải thất bại nếu sau khi
chạy cập nhật mà dữ liệu cổ phiếu VN30 vẫn chưa tồn tại. Không được commit trạng
thái 0/30 cổ phiếu rồi báo workflow thành công.

    python scripts/validate_data.py            # báo cáo và kiểm tra
    python scripts/validate_data.py --report   # chỉ in báo cáo, luôn thoát 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, quality, storage
from src import universe as universe_module
from src.logging_config import setup


def validate(minimum: int) -> tuple[bool, list[str]]:
    """Trả về (đạt hay không, danh sách lý do thất bại)."""
    symbols = universe_module.symbols()
    problems: list[str] = []

    for dataset, label in ((config.VNINDEX_DATASET, "VNINDEX"), (config.VN30_INDEX_DATASET, "VN30")):
        frame = storage.load_index(dataset)
        if frame is None or frame.empty:
            problems.append(f"Thiếu dữ liệu chỉ số {label} ({storage.index_path(dataset)})")

    available = storage.available_stock_symbols()
    if len(available) < minimum:
        problems.append(
            f"Chỉ có {len(available)}/{len(symbols)} tệp cổ phiếu trong {config.STOCK_DIR}, "
            f"cần tối thiểu {minimum}. Dữ liệu VN30 chưa được khởi tạo."
        )

    legacy = storage.legacy_stock_files()
    if legacy:
        problems.append(
            "Còn tệp ở bố cục cũ data/raw/: " + ", ".join(p.name for p in legacy)
        )

    for path in (config.UNIVERSE_FILE, config.VN30_SNAPSHOT_FILE, config.VNINDEX_FEATURES_FILE):
        if not path.exists():
            problems.append(f"Thiếu tệp {path.relative_to(config.ROOT)}")

    return not problems, problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra kho dữ liệu")
    parser.add_argument("--report", action="store_true", help="Chỉ in báo cáo, luôn thoát 0")
    parser.add_argument(
        "--min-stocks", type=int, default=config.MIN_STOCKS_FOR_VALID_DATASET,
        help="Số tệp cổ phiếu tối thiểu để coi là hợp lệ",
    )
    args = parser.parse_args()
    setup()

    symbols = universe_module.symbols()
    print(quality.dataset_rows(symbols).to_string(index=False))
    print("-" * 78)

    summary = quality.coverage_summary(symbols)
    counts = summary["statuses"]
    print(f"Danh sách VN30      {len(symbols)} mã (as_of {summary['universe_as_of'] or '-'})")
    print(f"Đầy đủ              {counts[quality.STATUS_COMPLETE]}")
    print(f"Thiếu lịch sử       {counts[quality.STATUS_SHORT]}")
    print(f"Không có tệp        {counts[quality.STATUS_MISSING]}")
    print(f"Lỗi cập nhật        {counts[quality.STATUS_ERROR]}")
    print(f"Tệp dữ liệu         {summary['files_written']}/{summary['files_expected']}")
    print(f"Đồng bộ GitHub      {summary['sync_status'] or 'chưa chạy'}")
    print("-" * 78)

    ok, problems = validate(args.min_stocks)
    if ok:
        print("Kho dữ liệu hợp lệ.")
        return 0

    for problem in problems:
        print(f"[LỖI] {problem}")
    if args.report:
        print("(chế độ --report: không trả mã lỗi)")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
