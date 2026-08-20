from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vnstock_data import update_market_dataset


if __name__ == "__main__":
    update_market_dataset("VNINDEX", name="vnindex")
    update_market_dataset("VN30", name="vn30")
    print("Updated VNINDEX and VN30 datasets.")
