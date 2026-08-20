import pandas as pd


def calculate_roro(close: pd.Series, level: int = 49) -> pd.DataFrame:
    """Calculate multi-horizon momentum strength and RORO regime signal."""
    close = pd.to_numeric(close, errors="coerce")
    strength = (
        close.pct_change(63) * 0.4
        + close.pct_change(126) * 0.2
        + close.pct_change(189) * 0.2
        + close.pct_change(252) * 0.2
    ) * 100

    equal = strength.rolling(level, min_periods=level).mean()
    roro = strength - equal

    return pd.DataFrame({"strength": strength, "roro": roro})


def classify_roro(roro: float, neutral_band: float = 0.0) -> str:
    if pd.isna(roro):
        return "Insufficient Data"
    if roro > neutral_band:
        return "Risk On"
    if roro < -neutral_band:
        return "Risk Off"
    return "Neutral"
