"""Bảng quyết định Market Regime và lớp quản trị danh mục."""

from __future__ import annotations

import numpy as np

from src import regime as regime_module
from src.portfolio_risk import guidance


def test_all_five_regimes_are_reachable():
    cases = {
        ("TÍCH CỰC", "THẤP", 70.0): regime_module.FAVOURABLE,
        ("TÍCH CỰC", "CAO", 70.0): regime_module.WARNING,
        ("TRUNG TÍNH", "BÌNH THƯỜNG", 50.0): regime_module.TRANSITION,
        ("SUY YẾU", "CAO", 40.0): regime_module.UNDER_PRESSURE,
        ("SUY YẾU", "RẤT CAO", 20.0): regime_module.STRESSED,
    }
    for (trend, stress, breadth), expected in cases.items():
        assert regime_module.classify_regime(trend, stress, breadth) == expected


def test_missing_inputs_give_unknown_not_a_guess():
    assert regime_module.classify_regime("CHƯA ĐỦ DỮ LIỆU", "THẤP", 70.0) == regime_module.UNKNOWN
    assert regime_module.classify_regime("TÍCH CỰC", "CHƯA ĐỦ DỮ LIỆU", 70.0) == regime_module.UNKNOWN


def test_unknown_breadth_does_not_force_a_strong_regime():
    assert regime_module.classify_regime("TÍCH CỰC", "THẤP", None) == regime_module.TRANSITION
    assert regime_module.classify_regime("TÍCH CỰC", "THẤP", np.nan) == regime_module.TRANSITION


def test_weak_breadth_downgrades_a_positive_trend():
    assert regime_module.classify_regime("TÍCH CỰC", "THẤP", 30.0) == regime_module.WARNING


def test_risk_level_escalates_only_on_two_context_flags():
    base, _ = regime_module.risk_level(regime_module.TRANSITION, "PHÂN HÓA TRUNG BÌNH", "TẬP TRUNG TRUNG BÌNH")
    assert base == regime_module.RISK_MEDIUM

    one, _ = regime_module.risk_level(regime_module.TRANSITION, "PHÂN HÓA CAO", "TẬP TRUNG TRUNG BÌNH")
    assert one == regime_module.RISK_MEDIUM

    two, reasons = regime_module.risk_level(regime_module.TRANSITION, "PHÂN HÓA CAO", "TẬP TRUNG CAO")
    assert two == regime_module.RISK_HIGH
    assert len(reasons) == 2


def test_insufficient_breadth_is_treated_conservatively():
    level, reasons = regime_module.risk_level(
        regime_module.TRANSITION, "PHÂN HÓA CAO", "TẬP TRUNG TRUNG BÌNH", breadth_sufficient=False
    )
    assert level == regime_module.RISK_HIGH
    assert any("không đủ" in r.lower() for r in reasons)


def test_build_regime_wires_every_input():
    result = regime_module.build_regime(
        trend={"state": "SUY YẾU"},
        stress={"state": "RẤT CAO"},
        breadth={"state": "YẾU", "score": 30.0, "sufficient": True},
        dispersion={"state": "PHÂN HÓA CAO"},
        concentration={"state": "TẬP TRUNG CAO"},
    )
    assert result["regime"] == regime_module.STRESSED
    assert result["risk_level"] == regime_module.RISK_VERY_HIGH
    assert set(result["inputs"]) == {"trend", "stress", "breadth", "breadth_score", "dispersion", "concentration"}


def test_portfolio_layer_never_prescribes_a_fixed_allocation():
    for regime in [
        regime_module.FAVOURABLE, regime_module.WARNING, regime_module.TRANSITION,
        regime_module.UNDER_PRESSURE, regime_module.STRESSED, regime_module.UNKNOWN,
    ]:
        payload = guidance({"regime": regime, "risk_level": "TRUNG BÌNH", "risk_reasons": []})
        # Câu miễn trừ cố tình chứa cụm "khuyến nghị mua bán" ở dạng phủ định.
        advice = " ".join(
            str(value) for key, value in payload.items() if key != "disclaimer"
        )
        assert "%" not in advice
        assert "mua" not in advice.lower()
        assert "bán" not in advice.lower()
        assert "khuyến nghị" in payload["disclaimer"]


def test_portfolio_covers_the_five_required_dimensions():
    payload = guidance({"regime": regime_module.FAVOURABLE, "risk_level": "THẤP", "risk_reasons": []})
    for key in ("risk_budget", "caution", "leverage", "concentration", "equity_weight"):
        assert payload[key]
