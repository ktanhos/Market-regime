"""Chạy thật app.py bằng Streamlit AppTest.

Kiểm tra hai điều quan trọng nhất:
1. Mở dashboard không gây lỗi.
2. Mở dashboard KHÔNG gọi API.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")


@pytest.fixture()
def no_network(monkeypatch):
    """Bất kỳ lời gọi mạng nào trong lúc render đều làm test thất bại."""
    import requests

    import src.vnstock_data as client

    def forbidden(*args, **kwargs):
        raise AssertionError("Dashboard đã gọi mạng khi chỉ mở trang")

    monkeypatch.setattr(requests, "request", forbidden)
    monkeypatch.setattr(requests, "get", forbidden)
    monkeypatch.setattr(requests, "post", forbidden)
    monkeypatch.setattr(client, "_single_call", forbidden)
    monkeypatch.setattr(client, "fetch_vn30_constituents", forbidden)


def test_dashboard_renders_without_exception(no_network):
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]


def test_dashboard_opens_without_calling_the_api(no_network):
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    assert not app.exception


def test_update_button_exists_and_is_not_triggered_on_open(no_network):
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    labels = [b.label for b in app.sidebar.button]
    assert "Cập nhật dữ liệu" in labels
    assert "Kiểm tra kết nối API" in labels


def test_no_deprecated_streamlit_width_argument():
    source = Path(APP).read_text(encoding="utf-8")
    assert "use_container_width" not in source
    assert 'width="stretch"' in source


def test_dashboard_renders_the_full_layout_with_complete_data(no_network, temp_store):
    """Đủ dữ liệu chỉ số và cổ phiếu thì toàn bộ thẻ phải hiện ra."""
    from src import config, storage, universe as universe_module
    from src.schema import standardize_ohlcv
    from src.updater import rebuild_features
    from tests.conftest import synthetic_ohlcv

    symbols = [f"S{i:02d}" for i in range(30)]
    universe_module.save_universe(symbols, source="ảnh chụp kiểm thử", as_of="2026-08-21")
    storage.write_frame(
        standardize_ohlcv(synthetic_ohlcv(600, seed=1)), storage.index_path(config.VNINDEX_DATASET)
    )
    storage.write_frame(
        standardize_ohlcv(synthetic_ohlcv(600, seed=2)), storage.index_path(config.VN30_INDEX_DATASET)
    )
    for i, symbol in enumerate(symbols):
        storage.write_frame(
            standardize_ohlcv(synthetic_ohlcv(320, seed=100 + i, drift=0.0005 if i % 3 else -0.0004)),
            storage.stock_path(symbol),
        )
    rebuild_features(symbols)

    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]

    headers = [h.value for h in app.subheader]
    for expected in ("Nhóm VN30 hiện tại", "Diễn biến", "Quản trị danh mục", "Chất lượng dữ liệu"):
        assert expected in headers

    body = " ".join(m.value for m in app.markdown)
    assert "TRẠNG THÁI THỊ TRƯỜNG VIỆT NAM" in body
    assert "Market Regime" in body
    assert "mã hợp lệ" in body
