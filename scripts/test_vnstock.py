from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vnstock_data import VNStockAdapter


if __name__ == "__main__":
    adapter = VNStockAdapter()

    for symbol in ["VNINDEX", "VN30"]:
        df = adapter.fetch_index(symbol, start="2025-01-01")
        print(f"\n{symbol}")
        print(df.tail())
        print(f"Rows: {len(df)}")
