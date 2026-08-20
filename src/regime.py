from typing import Dict

import numpy as np


def classify_market_regime(
    trend_state: str,
    stress_score: float,
    breadth_ma50: float,
    stress_high: float = 75.0,
    stress_extreme: float = 90.0,
    breadth_weak: float = 40.0,
    breadth_strong: float = 60.0,
) -> Dict[str, str]:
    """First rule-based regime engine. Thresholds are placeholders for backtesting."""
    if any(value is None for value in (stress_score, breadth_ma50)):
        return {"regime": "Insufficient Data", "reason": "Missing indicators"}

    stress_score = float(stress_score)
    breadth_ma50 = float(breadth_ma50)

    if stress_score >= stress_extreme and trend_state == "Risk Off" and breadth_ma50 < breadth_weak:
        return {"regime": "Panic", "reason": "Trend weak, stress extreme, breadth weak"}

    if stress_score >= stress_high and trend_state == "Risk Off":
        return {"regime": "Risk Off", "reason": "Trend weak with elevated stress"}

    if trend_state == "Risk On" and stress_score < stress_high and breadth_ma50 >= breadth_strong:
        return {"regime": "Risk On", "reason": "Trend positive, stress contained, breadth strong"}

    if trend_state == "Risk On" and (stress_score >= stress_high or breadth_ma50 < breadth_weak):
        return {"regime": "Caution", "reason": "Trend positive but stress or breadth is deteriorating"}

    if trend_state == "Risk Off" and stress_score < stress_high and breadth_ma50 >= breadth_weak:
        return {"regime": "Recovery", "reason": "Trend weak but stress is easing and breadth is stabilizing"}

    return {"regime": "Neutral", "reason": "Mixed market signals"}
