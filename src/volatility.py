import numpy as np
import pandas as pd


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 22, annualization: int = 252) -> pd.Series:
    """Annualized Parkinson volatility in percent."""
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    log_hl = np.log(high / low)
    value = log_hl.pow(2).rolling(window, min_periods=window).mean()
    vol = np.sqrt(value / (4 * np.log(2)) * annualization) * 100
    return vol


def downside_volatility(close: pd.Series, window: int = 22, annualization: int = 252) -> pd.Series:
    """Annualized volatility using only negative daily returns."""
    returns = pd.to_numeric(close, errors="coerce").pct_change()
    downside = returns.where(returns < 0)
    return downside.rolling(window, min_periods=window).std() * np.sqrt(annualization) * 100


def volatility_stress_score(volatility: pd.Series, window: int = 252) -> pd.Series:
    """Rolling percentile rank of volatility, useful for regime classification."""
    return volatility.rolling(window, min_periods=max(30, window // 4)).rank(pct=True) * 100
