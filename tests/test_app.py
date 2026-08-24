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
    monkeypatch.setattr(client, "fetch_index_members", forbidden)


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
    assert "Kiểm tra API" in labels


def test_no_deprecated_streamlit_width_argument():
    source = Path(APP).read_text(encoding="utf-8")
    assert "use_container_width" not in source
    assert 'width="stretch"' in source


def test_dashboard_renders_the_full_layout_with_complete_data(no_network, temp_store):
    """Đủ dữ liệu chỉ số và cổ phiếu thì toàn bộ thẻ phải hiện ra."""
    from src import config, features, storage, universe as universe_module
    from src.schema import standardize_ohlcv
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
    features.rebuild(symbols)

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


@pytest.mark.parametrize("available", [0, 15, 29, 30])
def test_dashboard_never_crashes_at_any_data_stage(no_network, temp_store, available):
    """0/30, 15/30, 29/30 và 30/30 mã đều phải render được."""
    from src import config, features, storage, universe as universe_module
    from src.schema import standardize_ohlcv
    from tests.conftest import synthetic_ohlcv

    symbols = [f"S{i:02d}" for i in range(30)]
    universe_module.save_universe(symbols, source="ảnh chụp kiểm thử", as_of="2026-08-23")
    storage.write_frame(
        standardize_ohlcv(synthetic_ohlcv(600, seed=1)), storage.index_path(config.VNINDEX_DATASET)
    )
    storage.write_frame(
        standardize_ohlcv(synthetic_ohlcv(600, seed=2)), storage.index_path(config.VN30_INDEX_DATASET)
    )
    for i, symbol in enumerate(symbols[:available]):
        storage.write_frame(
            standardize_ohlcv(synthetic_ohlcv(320, seed=100 + i)), storage.stock_path(symbol)
        )
    features.rebuild(symbols)

    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]

    body = " ".join(m.value for m in app.markdown)
    assert "TRẠNG THÁI THỊ TRƯỜNG VIỆT NAM" in body
    assert "Chất lượng dữ liệu" in [h.value for h in app.subheader]


def test_first_run_shows_an_invitation_not_a_wall_of_errors(no_network, temp_store):
    """Lần đầu chưa có giá cổ phiếu thì phải mời khởi tạo, không đổ 30 dòng lỗi."""
    from src import config, features, storage, universe as universe_module
    from src.schema import standardize_ohlcv
    from tests.conftest import synthetic_ohlcv

    symbols = [f"S{i:02d}" for i in range(30)]
    universe_module.save_universe(symbols, source="ảnh chụp kiểm thử", as_of="2026-08-23")
    storage.write_frame(
        standardize_ohlcv(synthetic_ohlcv(600, seed=1)), storage.index_path(config.VNINDEX_DATASET)
    )
    features.rebuild(symbols)

    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    assert not app.exception

    invitations = " ".join(item.value for item in app.info)
    assert "chưa được khởi tạo" in invitations
    assert "Cập nhật dữ liệu" in invitations
    # Bảng chi tiết 30 dòng "Không có tệp" không được bung ra ngay ở lần đầu.
    assert not any("Chi tiết từng tệp dữ liệu" in str(e.label) for e in app.expander)
