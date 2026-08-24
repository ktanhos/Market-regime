"""Trend, Stress, Breadth, Dispersion, Risk concentration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.breadth import compute_breadth
from src.concentration import compute_concentration, concentration_metrics, risk_weights
from src.dispersion import compute_dispersion, cross_sectional_dispersion
from src.trend import TREND_UNKNOWN, calculate_roro, calculate_strength, classify_roro, trend_snapshot
from src.schema import standardize_ohlcv, to_close_panel
from src.stress import classify_stress, parkinson_volatility, stress_snapshot
from tests.conftest import synthetic_ohlcv


# --- Trend -------------------------------------------------------------------

def test_strength_matches_the_documented_weights():
    close = pd.Series(np.linspace(100, 200, 400))
    expected = (
        close.pct_change(63) * 0.4
        + close.pct_change(126) * 0.2
        + close.pct_change(189) * 0.2
        + close.pct_change(252) * 0.2
    ) * 100
    pd.testing.assert_series_equal(calculate_strength(close), expected.rename("strength"))


def test_roro_is_strength_minus_its_own_moving_average():
    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(4).normal(0, 0.01, 500))))
    frame = calculate_roro(close)
    strength = frame["strength"]
    expected = strength - strength.rolling(config.RORO_BASELINE_WINDOW, min_periods=config.RORO_BASELINE_WINDOW).mean()
    pd.testing.assert_series_equal(frame["roro"], expected.rename("roro"))


def test_roro_has_three_levels_with_a_neutral_band():
    assert classify_roro(5.0, 1.0) == "TÍCH CỰC"
    assert classify_roro(0.4, 1.0) == "TRUNG TÍNH"
    assert classify_roro(-5.0, 1.0) == "SUY YẾU"
    assert classify_roro(float("nan"), 1.0) == TREND_UNKNOWN


def test_trend_snapshot_on_short_series_says_unknown():
    snapshot = trend_snapshot(pd.Series([100.0, 101.0, 102.0]))
    assert snapshot["state"] == TREND_UNKNOWN


# --- Stress ------------------------------------------------------------------

def test_parkinson_uses_mean_squared_log_range():
    """Kiểm định trực tiếp công thức, tránh nhầm giữa variance và mean squared range."""
    rng = np.random.default_rng(9)
    high = pd.Series(100 + rng.random(60) * 5)
    low = high - rng.random(60) * 3
    window = config.PARKINSON_WINDOW
    manual = np.sqrt(
        np.mean(np.log(high[-window:] / low[-window:]) ** 2) / (4 * np.log(2)) * 252
    ) * 100
    assert parkinson_volatility(high, low).iloc[-1] == pytest.approx(manual)


def test_stress_bands_are_percentiles_not_absolute_levels():
    assert classify_stress(5.0) == "THẤP"
    assert classify_stress(40.0) == "BÌNH THƯỜNG"
    assert classify_stress(70.0) == "CAO"
    assert classify_stress(95.0) == "RẤT CAO"
    assert classify_stress(float("nan")) == "CHƯA ĐỦ DỮ LIỆU"


def test_stress_snapshot_returns_proxy_label_not_vix():
    snapshot = stress_snapshot(standardize_ohlcv(synthetic_ohlcv(400, seed=5)))
    assert "VIX" not in snapshot["label"].upper()
    assert snapshot["state"] in {"THẤP", "BÌNH THƯỜNG", "CAO", "RẤT CAO", "CHƯA ĐỦ DỮ LIỆU"}


# --- Breadth -----------------------------------------------------------------

def _panel_frames(count: int = 30, periods: int = 300, seed: int = 12):
    return {
        f"S{i:02d}": standardize_ohlcv(
            synthetic_ohlcv(periods, seed=seed + i, drift=0.0006 if i % 2 == 0 else -0.0006)
        )
        for i in range(count)
    }


def test_breadth_reports_valid_over_total_and_never_assumes_full_coverage():
    frames = _panel_frames(30)
    frames["S05"] = frames["S05"].tail(25)          # thiếu lịch sử dài
    universe = sorted(list(frames) + ["MISSING1", "MISSING2"])
    del frames["S07"]                                # thiếu hẳn tệp dữ liệu

    result = compute_breadth(frames, universe)
    assert result["universe_size"] == 32
    assert result["loaded_symbols"] == 29
    assert set(result["missing_symbols"]) == {"MISSING1", "MISSING2", "S07"}
    # MA200 chỉ tính trên các mã đủ 200 phiên: 29 mã tải được trừ S05.
    assert result["components"]["ma200"]["valid"] == 28
    assert result["components"]["ma20"]["valid"] == 29
    assert result["valid_symbols"] == 28
    assert 0 <= result["score"] <= 100


def test_breadth_covers_all_required_components():
    result = compute_breadth(_panel_frames(25), [f"S{i:02d}" for i in range(25)])
    assert set(result["components"]) == {"ma20", "ma50", "ma200", "ret1", "ret5", "ret20"}


def test_breadth_with_no_data_is_not_a_crash():
    result = compute_breadth({}, ["AAA", "BBB"])
    assert result["state"] == "CHƯA CÓ DỮ LIỆU"
    assert result["data_state"] == "no_data"
    assert result["sufficient"] is False
    assert result["missing_symbols"] == ["AAA", "BBB"]


def test_breadth_distinguishes_no_data_from_short_history():
    """Chưa có tệp và có tệp nhưng thiếu lịch sử là hai tình huống khác nhau."""
    universe = [f"S{i:02d}" for i in range(30)]
    short = {
        symbol: standardize_ohlcv(synthetic_ohlcv(30, seed=i))
        for i, symbol in enumerate(universe)
    }
    result = compute_breadth(short, universe)
    assert result["data_state"] == "insufficient"
    assert result["state"] == "DỮ LIỆU CHƯA ĐỦ"
    assert result["loaded_symbols"] == 30       # có tệp
    assert result["min_valid_symbols"] == 0     # nhưng không mã nào đủ MA200


def test_breadth_flags_symbols_whose_data_lags_the_rest():
    universe = [f"S{i:02d}" for i in range(30)]
    frames = {
        symbol: standardize_ohlcv(synthetic_ohlcv(300, seed=i))
        for i, symbol in enumerate(universe)
    }
    frames["S07"] = frames["S07"].iloc[:-20]
    result = compute_breadth(frames, universe)
    assert result["data_state"] == "stale"
    assert result["state"] == "DỮ LIỆU KHÔNG ĐỒNG BỘ"
    assert result["stale_symbols"] == ["S07"]
    assert result["max_gap_sessions"] == 20
    # Vẫn tính được số liệu, chỉ kèm cảnh báo.
    assert result["sufficient"] is True
    assert 0 <= result["score"] <= 100


def test_breadth_all_above_ma_gives_hundred_percent():
    dates = pd.bdate_range("2024-01-01", periods=260)
    rising = pd.DataFrame(
        {"date": dates, "open": 1.0, "high": 1.0, "low": 1.0,
         "close": np.linspace(100, 300, 260), "volume": 1}
    )
    frames = {f"S{i}": rising.copy() for i in range(5)}
    result = compute_breadth(frames, list(frames))
    assert result["components"]["ma200"]["pct"] == pytest.approx(100.0)
    assert result["components"]["ret20"]["pct"] == pytest.approx(100.0)


# --- Dispersion --------------------------------------------------------------

def test_dispersion_is_cross_sectional_std_of_returns():
    panel = to_close_panel(_panel_frames(15, seed=40))
    window = 20
    series = cross_sectional_dispersion(panel, window)
    manual = panel.pct_change(window).std(axis=1, ddof=1) * 100
    assert series.dropna().iloc[-1] == pytest.approx(manual.dropna().iloc[-1])


def test_dispersion_reports_all_windows_and_flags_its_historical_basis():
    result = compute_dispersion(to_close_panel(_panel_frames(20, seed=60)))
    assert set(result["windows"]) == {1, 5, 20}
    assert result["primary_window"] == 20
    assert 0 <= result["percentile"] <= 100
    assert "rổ VN30 hiện tại" in result["historical_basis"]


def test_dispersion_of_identical_stocks_is_zero():
    dates = pd.bdate_range("2024-01-01", periods=120)
    close = np.linspace(100, 150, 120)
    frames = {
        f"S{i}": pd.DataFrame({"date": dates, "open": close, "high": close, "low": close,
                               "close": close, "volume": 1})
        for i in range(12)
    }
    result = compute_dispersion(to_close_panel(frames))
    assert result["value"] == pytest.approx(0.0, abs=1e-9)


def test_dispersion_empty_panel():
    result = compute_dispersion(pd.DataFrame())
    assert result["state"] == "CHƯA ĐỦ DỮ LIỆU"


# --- Risk concentration ------------------------------------------------------

def test_risk_weights_sum_to_one():
    weights = risk_weights(pd.Series({"A": 10.0, "B": 20.0, "C": 30.0, "D": np.nan}))
    assert weights.sum() == pytest.approx(1.0)
    assert "D" not in weights.index


def test_effective_names_equals_inverse_hhi():
    weights = pd.Series([0.25] * 4, index=list("ABCD"))
    metrics = concentration_metrics(weights)
    assert metrics["hhi"] == pytest.approx(0.25)
    assert metrics["effective_names"] == pytest.approx(4.0)
    assert metrics["top_shares"][5] == pytest.approx(100.0)


def test_concentration_is_labelled_as_a_proxy():
    result = compute_concentration(to_close_panel(_panel_frames(20, seed=90)))
    assert "proxy" in result["proxy_note"].lower()
    assert result["contributors"] == 20
    assert 0 < result["top_shares"][5] <= 100
    assert result["effective_names"] <= result["contributors"]


def test_concentration_empty_panel():
    result = compute_concentration(pd.DataFrame())
    assert result["state"] == "CHƯA ĐỦ DỮ LIỆU"
    assert result["table"].empty
