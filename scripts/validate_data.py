"""In báo cáo chất lượng của kho dữ liệu hiện tại. Không gọi API.

    python scripts/validate_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import quality, universe as universe_module


def main() -> int:
    symbols = universe_module.symbols()
    table = quality.dataset_rows(symbols)
    print(table.to_string(index=False))
    print("-" * 72)
    summary = quality.coverage_summary(symbols)
    for key, value in summary.items():
        print(f"{key:<28} {value}")
    missing = summary["stock_symbols_missing"]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
