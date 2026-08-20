import pandas as pd


def breadth_above_ma(close: pd.DataFrame, window: int = 50) -> pd.Series:
    """Percentage of securities whose close is above their moving average."""
    close = close.apply(pd.to_numeric, errors="coerce")
    ma = close.rolling(window, min_periods=window).mean()
    valid = close.notna() & ma.notna()
    above = (close > ma) & valid
    count = valid.sum(axis=1)
    return above.sum(axis=1).div(count.replace(0, pd.NA)) * 100


def advance_decline(close: pd.DataFrame) -> pd.Series:
    """Daily advance minus decline breadth."""
    returns = close.pct_change()
    advances = (returns > 0).sum(axis=1)
    declines = (returns < 0).sum(axis=1)
    return advances - declines


def breadth_score(close: pd.DataFrame, ma_window: int = 50) -> pd.DataFrame:
    """Minimal breadth feature set for the first regime model version."""
    return pd.DataFrame({
        "breadth_ma50": breadth_above_ma(close, ma_window),
        "advance_decline": advance_decline(close),
    })
