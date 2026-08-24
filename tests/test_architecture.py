"""Ràng buộc kiến trúc, kiểm tra bằng cách phân tích chính mã nguồn.

Các quy tắc này dễ bị phá vỡ khi thêm tính năng, nên chúng được kiểm tra tự động
thay vì chỉ ghi trong tài liệu.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Thứ tự tầng: mỗi tầng chỉ được phụ thuộc vào tầng thấp hơn hoặc cùng tầng.
LAYERS = {
    "logging_config": 0, "config": 0, "schema": 0,
    "storage": 1, "github_store": 1,
    "vnstock_data": 2, "universe": 2,
    "trend": 3, "stress": 3, "breadth": 3, "dispersion": 3, "concentration": 3,
    "features": 4, "quality": 4,
    "regime": 5,
    "portfolio_risk": 6,
    "updater": 7,
}


def module_imports(path: Path) -> set[str]:
    """Các module trong gói src mà tệp này import."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "src":
                found.update(alias.name for alias in node.names)
            elif module.startswith("src."):
                found.add(module.split(".", 1)[1].split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("src."):
                    found.add(alias.name.split(".", 1)[1].split(".")[0])
    return {name for name in found if name in LAYERS}


def top_level_packages(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.add((node.module or "").split(".")[0])
    return found


def test_only_vnstock_data_imports_vnstock():
    """Toàn bộ hiểu biết về vnstock nằm gọn trong một module."""
    offenders = []
    for path in [ROOT / "app.py", *sorted(SRC.glob("*.py"))]:
        if path.name == "vnstock_data.py":
            continue
        if "vnstock" in top_level_packages(path):
            offenders.append(path.name)
    assert offenders == [], f"Các tệp này import vnstock trực tiếp: {offenders}"


def test_app_does_not_import_streamlit_internals_of_the_data_layer():
    """Tầng dữ liệu không được biết đến Streamlit (trừ chỗ đọc secrets)."""
    offenders = []
    for path in sorted(SRC.glob("*.py")):
        if path.name == "github_store.py":  # chỉ đọc st.secrets, có fallback rõ ràng
            continue
        if "streamlit" in top_level_packages(path):
            offenders.append(path.name)
    assert offenders == [], f"Các module này phụ thuộc Streamlit: {offenders}"


def test_layer_dependencies_only_point_downwards():
    violations = []
    for path in sorted(SRC.glob("*.py")):
        name = path.stem
        if name not in LAYERS:
            continue
        for dependency in module_imports(path):
            if LAYERS[dependency] > LAYERS[name]:
                violations.append(f"{name} -> {dependency}")
    assert violations == [], f"Phụ thuộc ngược tầng: {violations}"


def test_no_import_cycles():
    graph = {p.stem: module_imports(p) for p in sorted(SRC.glob("*.py")) if p.stem in LAYERS}
    visiting: set[str] = set()
    done: set[str] = set()
    cycles: list[str] = []

    def walk(node: str, trail: list[str]) -> None:
        if node in visiting:
            cycles.append(" -> ".join(trail + [node]))
            return
        if node in done:
            return
        visiting.add(node)
        for child in sorted(graph.get(node, ())):
            walk(child, trail + [node])
        visiting.discard(node)
        done.add(node)

    for name in sorted(graph):
        walk(name, [])
    assert cycles == [], f"Phát hiện import vòng: {cycles}"


def test_every_src_module_imports_cleanly():
    for path in sorted(SRC.glob("*.py")):
        if path.stem == "__init__":
            continue
        importlib.import_module(f"src.{path.stem}")


def test_vnstock_data_exposes_the_documented_interface():
    module = importlib.import_module("src.vnstock_data")
    for name in ("fetch_index", "fetch_equity", "fetch_index_members", "fetch_history",
                 "connectivity_check", "expected_schema"):
        assert callable(getattr(module, name)), f"Thiếu {name} trong giao diện công khai"


def test_no_legacy_vnstock_entry_points_anywhere():
    """Không được quay lại các lối vào cũ đã bị thay thế."""
    banned = ("Vnstock()", "Vnstock().stock", "Vnstock().index", "Market(source=")
    offenders = []
    for path in [ROOT / "app.py", *sorted(SRC.glob("*.py")), *sorted((ROOT / "scripts").glob("*.py"))]:
        text = path.read_text(encoding="utf-8")
        for token in banned:
            # Bỏ qua các dòng chú thích nêu rõ "không dùng".
            for line in text.splitlines():
                stripped = line.strip()
                if token in stripped and not stripped.startswith(("#", "*", '"""')):
                    offenders.append(f"{path.name}: {stripped[:80]}")
    assert offenders == [], f"Còn lối vào vnstock cũ: {offenders}"


def test_no_hardcoded_data_paths():
    """Mọi đường dẫn đi qua pathlib trong src/config.py."""
    offenders = []
    for path in [ROOT / "app.py", *sorted(SRC.glob("*.py"))]:
        if path.stem == "config":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("#", "*")):
                continue
            if '"data/' in stripped or "'data/" in stripped:
                offenders.append(f"{path.name}: {stripped[:80]}")
    assert offenders == [], f"Đường dẫn dữ liệu bị hardcode: {offenders}"


@pytest.mark.parametrize("removed", [
    "cached_market", "cached_breadth", "refresh_data", "local_data", "data",
])
def test_dead_modules_are_gone(removed):
    assert not (SRC / f"{removed}.py").exists(), f"src/{removed}.py vẫn còn"


@pytest.mark.parametrize("removed", [
    "build_vn30_metrics", "backtest_vn30_structure", "refresh_base_data",
    "test_vnstock", "build_features", "validate_market_data",
])
def test_dead_scripts_are_gone(removed):
    assert not (ROOT / "scripts" / f"{removed}.py").exists(), f"scripts/{removed}.py vẫn còn"


def test_expected_file_structure_exists():
    for name in ("schema", "storage", "vnstock_data", "updater", "universe", "features",
                 "breadth", "dispersion", "concentration", "trend", "stress", "regime",
                 "portfolio_risk", "github_store"):
        assert (SRC / f"{name}.py").exists(), f"Thiếu src/{name}.py"


def test_full_history_threshold_matches_the_longest_indicator():
    """Ngưỡng 'đủ lịch sử' không được thấp hơn chỉ tiêu khắt khe nhất đang tính."""
    from src import config

    longest = max(config.BREADTH_MA_WINDOWS)
    assert config.MIN_SESSIONS_FOR_FULL_HISTORY >= longest
    # Cũng không được cao hơn nhiều, nếu không bảng chất lượng sẽ báo "thiếu lịch
    # sử" cho những mã mà breadth vẫn tính được đầy đủ.
    assert config.MIN_SESSIONS_FOR_FULL_HISTORY <= longest + 10


def test_returns_never_forward_fill_missing_sessions():
    """pandas mặc định pad NaN trước khi tính pct_change, làm sai lợi suất."""
    for name in ("dispersion", "concentration"):
        text = (SRC / f"{name}.py").read_text(encoding="utf-8")
        for line in text.splitlines():
            if "pct_change(" in line and not line.strip().startswith("#"):
                assert "fill_method=None" in line, f"{name}.py: {line.strip()}"
