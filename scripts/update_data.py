from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vnstock_data import incremental_update


DATASETS = [
    {"symbol": "VNINDEX", "dataset_name": "vnindex", "asset_type": "index"},
    {"symbol": "VN30", "dataset_name": "vn30", "asset_type": "index"},
]


if __name__ == "__main__":
    for item in DATASETS:
        path = incremental_update(**item)
        print(f"Updated {item['symbol']}: {path}")
