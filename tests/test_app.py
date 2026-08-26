"""Chạy thật app.py bằng Streamlit AppTest.

Kiểm tra hai điều quan trọng nhất:
1. Mở dashboard không gây lỗi.
2. Mở dashboard KHÔNG gọi API.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.proto.TextInput_pb2 import TextInput as TextInputProto
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
    assert "Kiểm tra kết nối API" in labels


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


def test_sidebar_shows_api_access_without_a_key(no_network, temp_store, monkeypatch):
    from src import config, features, storage, universe as universe_module
    from src.schema import standardize_ohlcv
    from tests.conftest import synthetic_ohlcv

    monkeypatch.delenv(config.VNSTOCK_API_KEY_ENV, raising=False)
    universe_module.save_universe([f"S{i:02d}" for i in range(30)], source="kiểm thử")
    storage.write_frame(
        standardize_ohlcv(synthetic_ohlcv(400, seed=1)), storage.index_path(config.VNINDEX_DATASET)
    )
    features.rebuild(universe_module.symbols())

    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]

    sidebar = " ".join(m.value for m in app.sidebar.markdown)
    assert "KẾT NỐI VNSTOCK" in sidebar
    assert "API key chưa được cấu hình" in sidebar
    assert "Giới hạn vận hành" in sidebar

    # Ô nhập phải hiện thẳng ở thanh bên, không nằm trong expander đóng sẵn.
    fields = [t.label for t in app.sidebar.text_input]
    assert "VNSTOCK API KEY" in fields
    # type="password" nằm trong proto, không phải thuộc tính .type của AppTest.
    field = app.sidebar.text_input(key="vnstock_api_key_input")
    assert field.proto.type == TextInputProto.Type.PASSWORD
    assert field.placeholder
    buttons = [b.label for b in app.sidebar.button]
    assert "Áp dụng API key" in buttons


def test_a_session_key_is_used_and_never_rendered(no_network, temp_store, monkeypatch):
    """Khóa nhập trong phiên phải được ưu tiên nhưng không bao giờ hiện ra trang."""
    from src import config, features, storage, universe as universe_module
    from src.schema import standardize_ohlcv
    from tests.conftest import synthetic_ohlcv

    secret = "SESSION-KEY-0123456789"
    monkeypatch.setenv(config.VNSTOCK_API_KEY_ENV, "ENV-KEY-0123456789")
    universe_module.save_universe([f"S{i:02d}" for i in range(30)], source="kiểm thử")
    storage.write_frame(
        standardize_ohlcv(synthetic_ohlcv(400, seed=1)), storage.index_path(config.VNINDEX_DATASET)
    )
    features.rebuild(universe_module.symbols())

    app = AppTest.from_file(APP, default_timeout=180)
    app.session_state["vnstock_api_key"] = secret
    app.run()
    assert not app.exception, [str(e) for e in app.exception]

    rendered = " ".join(
        [m.value for m in app.markdown]
        + [m.value for m in app.sidebar.markdown]
        + [c.value for c in app.caption]
    )
    assert secret not in rendered
    assert "API key đã được cấu hình" in rendered
    assert "khóa nhập trong phiên" in rendered      # nguồn khóa, không phải giá trị
    assert "Xóa API key phiên này" in [b.label for b in app.sidebar.button]


def test_the_api_key_field_is_not_hidden_behind_an_expander(no_network, temp_store, monkeypatch):
    """Ô nhập phải nhìn thấy ngay, và nằm phía trên hai nút hành động."""
    from src import config, features, storage, universe as universe_module
    from src.schema import standardize_ohlcv
    from tests.conftest import synthetic_ohlcv

    monkeypatch.delenv(config.VNSTOCK_API_KEY_ENV, raising=False)
    universe_module.save_universe([f"S{i:02d}" for i in range(30)], source="kiểm thử")
    storage.write_frame(
        standardize_ohlcv(synthetic_ohlcv(400, seed=1)), storage.index_path(config.VNINDEX_DATASET)
    )
    features.rebuild(universe_module.symbols())

    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]

    # Không có expander nào chứa ô nhập khóa.
    expander_labels = [str(e.label) for e in app.expander]
    assert not any("API" in label for label in expander_labels), expander_labels

    # Ô nhập tồn tại thật ở thanh bên và đứng trước hai nút hành động.
    labels = [b.label for b in app.sidebar.button]
    assert [t.label for t in app.sidebar.text_input] == ["VNSTOCK API KEY"]
    assert labels.index("Áp dụng API key") < labels.index("Kiểm tra kết nối API")
    assert labels.index("Áp dụng API key") < labels.index("Cập nhật dữ liệu")


def test_pipeline_speed_panel_shows_measured_phase_durations(no_network, temp_store):
    """Đo tốc độ pipeline phải tới được giao diện, không chỉ nằm trong log."""
    from src import config, features, storage, universe as universe_module
    from src.schema import standardize_ohlcv
    from tests.conftest import synthetic_ohlcv

    symbols = [f"S{i:02d}" for i in range(30)]
    universe_module.save_universe(symbols, source="kiểm thử", as_of="2026-08-21")
    storage.write_frame(
        standardize_ohlcv(synthetic_ohlcv(600, seed=1)), storage.index_path(config.VNINDEX_DATASET)
    )
    for i, symbol in enumerate(symbols):
        storage.write_frame(
            standardize_ohlcv(synthetic_ohlcv(320, seed=100 + i)), storage.stock_path(symbol)
        )
    features.rebuild(symbols)
    log = storage.read_json(config.UPDATE_LOG_FILE) or {}
    log["phase_seconds"] = {
        "universe": 0.4, "vnindex": 1.2, "vn30_index": 1.1, "stocks": 42.7,
        "features": 0.6, "github": 3.2,
    }
    storage.write_json(config.UPDATE_LOG_FILE, log)

    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    assert not app.exception, [str(e) for e in app.exception]

    labels = [str(e.label) for e in app.expander]
    assert any("Tốc độ" in label for label in labels)


def test_applying_a_key_through_the_widget_configures_the_run(no_network, temp_store, monkeypatch):
    """Bấm Áp dụng phải thật sự đưa khóa vào session và đổi trạng thái hiển thị."""
    from src import config, features, storage, universe as universe_module
    from src.schema import standardize_ohlcv
    from tests.conftest import synthetic_ohlcv

    monkeypatch.delenv(config.VNSTOCK_API_KEY_ENV, raising=False)
    universe_module.save_universe([f"S{i:02d}" for i in range(30)], source="kiểm thử")
    storage.write_frame(
        standardize_ohlcv(synthetic_ohlcv(400, seed=1)), storage.index_path(config.VNINDEX_DATASET)
    )
    features.rebuild(universe_module.symbols())

    app = AppTest.from_file(APP, default_timeout=180)
    app.run()
    assert "API key chưa được cấu hình" in " ".join(m.value for m in app.sidebar.markdown)

    secret = "TYPED-KEY-0123456789"
    app.sidebar.text_input(key="vnstock_api_key_input").set_value(secret).run()
    next(b for b in app.sidebar.button if b.label == "Áp dụng API key").click().run()

    assert app.session_state["vnstock_api_key"] == secret
    sidebar = " ".join(m.value for m in app.sidebar.markdown)
    assert "API key đã được cấu hình" in sidebar
    assert secret not in sidebar
