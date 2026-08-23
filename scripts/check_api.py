"""Chẩn đoán kết nối tới nguồn dữ liệu vnstock.

Ba phép thử tối thiểu: một chỉ số (VNINDEX), một cổ phiếu (FPT) và danh sách
thành phần VN30. Nếu ba phép thử này không chạy được thì không nên gọi tiếp 30
mã còn lại.

    python scripts/check_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.logging_config import setup
from src.vnstock_data import connectivity_check, expected_schema, vnstock_version


def main() -> int:
    setup()
    print(f"vnstock {vnstock_version()} · nguồn chính {config.PRIMARY_SOURCE}")
    print("API: Quote(symbol=..., source=...).history(start=..., end=..., interval='1D')")
    print("     Listing(source=...).symbols_by_group('VN30')")
    print("Schema kỳ vọng: " + ", ".join(expected_schema()))
    print("-" * 78)

    probes = connectivity_check()
    for probe in probes:
        verdict = "SUCCESS" if probe.ok else "FAILED "
        print(f"[{verdict}] {probe.name}")
        if probe.ok:
            print(f"           nguồn {probe.source} · {probe.rows} dòng")
            if probe.last_date:
                print(f"           {probe.first_date} → {probe.last_date}")
            print(f"           schema: {', '.join(probe.schema)}")
            if probe.name == "VN30 Universe":
                print(f"           {probe.detail}")
        else:
            print(f"           [{probe.kind}] {probe.error}")

    print("-" * 78)
    failures = [p for p in probes if not p.ok]
    if failures:
        print(f"{len(failures)}/{len(probes)} phép thử thất bại. "
              "Không nên chạy cập nhật toàn bộ khi chưa xử lý xong.")
        return 1
    print(f"{len(probes)}/{len(probes)} phép thử đạt. Có thể chạy scripts/update_data.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
