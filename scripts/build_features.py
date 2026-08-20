from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import PROCESSED_DIR, load_parquet
from src.roro import calculate_roro, classify_roro
from src.volatility import downside_volatility, parkinson_volatility, volatility_stress_score


def build_features(name: str) -> Path:
    df = load_parquet(name)
    roro = calculate_roro(df["close"])
    park = parkinson_volatility(df["high"], df["low"])
    downside = downside_volatility(df["close"])
    stress = volatility_stress_score(park)

    out = df[["date", "open", "high", "low", "close"]].copy()
    out = pd.concat([out, roro, park.rename("parkinson_vol"), downside.rename("downside_vol"), stress.rename("stress_score")], axis=1)
    out["trend_state"] = out["roro"].apply(classify_roro)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}_features.parquet"
    out.to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    for dataset in ["vnindex", "vn30"]:
        try:
            path = build_features(dataset)
            print(f"Built features: {path}")
        except FileNotFoundError:
            print(f"Raw dataset missing: {dataset}")
