"""Interpretation Layer: dịch chỉ tiêu sang ngôn ngữ phổ thông, không tính lại gì."""

from __future__ import annotations

from src import config, features, narrative, regime as regime_module, storage
from src import universe as universe_module
from src.schema import standardize_ohlcv

from tests.conftest import synthetic_ohlcv

UNIVERSE = [f"S{i:02d}" for i in range(30)]


def _seed_full_dataset(drift: float = 0.0006):
    universe_module.save_universe(UNIVERSE, source="kiểm thử", as_of="2026-08-21")
    storage.write_frame(
        standardize_ohlcv(synthetic_ohlcv(700, seed=1, drift=drift)),
        storage.index_path(config.VNINDEX_DATASET),
    )
    for i, symbol in enumerate(UNIVERSE):
        storage.write_frame(
            standardize_ohlcv(synthetic_ohlcv(300, seed=100 + i, drift=drift)),
            storage.stock_path(symbol),
        )
    return features.rebuild(UNIVERSE)


def _full_state(temp_store):
    snapshot = features.build_snapshot(UNIVERSE)
    assert snapshot["ready"]
    snapshot["regime"] = regime_module.build_regime(
        snapshot["trend"], snapshot["stress"], snapshot["breadth"],
        snapshot["dispersion"], snapshot["concentration"],
    )
    return snapshot


def test_build_narrative_covers_every_factor(temp_store):
    _seed_full_dataset()
    state = _full_state(temp_store)
    story = narrative.build_narrative(state)
    for key in ("regime", "trend", "stress", "breadth", "dispersion", "concentration"):
        assert key in story

    for key in ("trend", "stress", "breadth", "dispersion", "concentration"):
        card = story[key]
        assert card["title"]
        assert card["verdict"]
        assert card["plain"]
        assert card["why"]
        assert card["change"]
        assert card["watch"]

    assert story["regime"]["headline"]
    assert story["regime"]["summary"]
    assert story["regime"]["watch"]


def test_narrative_never_crashes_on_missing_data(temp_store):
    """Chưa có dữ liệu VN30 nào vẫn phải diễn giải được, không được sập."""
    universe_module.save_universe(UNIVERSE, source="kiểm thử", as_of="2026-08-21")
    storage.write_frame(
        standardize_ohlcv(synthetic_ohlcv(300, seed=1)), storage.index_path(config.VNINDEX_DATASET)
    )
    features.rebuild(UNIVERSE)
    state = _full_state(temp_store)
    story = narrative.build_narrative(state)
    assert "Chưa có dữ liệu" in story["breadth"]["plain"] or story["breadth"]["plain"]


def test_previous_snapshot_is_persisted_across_rebuilds(temp_store):
    """Lần rebuild thứ hai phải giữ lại vài trường của lần trước để so sánh."""
    first = _seed_full_dataset(drift=0.0006)
    assert first.get("previous") is None

    for i, symbol in enumerate(UNIVERSE):
        storage.write_frame(
            standardize_ohlcv(synthetic_ohlcv(305, seed=100 + i, drift=-0.0006)),
            storage.stock_path(symbol),
        )
    second = features.rebuild(UNIVERSE)
    assert second["previous"] is not None
    assert second["previous"]["breadth_score"] == first["breadth_score"]

    state = _full_state(temp_store)
    assert state["vn30_previous"] == second["previous"]

    story = narrative.build_narrative(state)
    assert "lần cập nhật" in story["breadth"]["change"]


def test_change_wording_reflects_direction_not_just_a_number(temp_store):
    _seed_full_dataset()
    state = _full_state(temp_store)
    card = narrative.trend_story(state["trend"])
    assert card["change"] in (
        "Chưa đủ dữ liệu để so sánh."
    ) or any(word in card["change"] for word in ("tăng", "giảm", "không đổi"))
