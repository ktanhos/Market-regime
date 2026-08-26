"""Feature Layer: biến dữ liệu đã lưu thành các chỉ tiêu.

Tầng này đọc từ ``src.storage`` và **không bao giờ gọi mạng**. Nó cũng không
biết gì về Market Regime hay quản trị danh mục: hai tầng đó nằm phía trên và
nhận đầu vào từ đây.

    Market Data Layer (storage)
        ↓
    Feature Layer (module này)
        ↓
    Market Regime Layer (src.regime)
        ↓
    Portfolio Risk Layer (src.portfolio_risk)
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from src import breadth as breadth_module
from src import concentration as concentration_module
from src import config, storage
from src import dispersion as dispersion_module
from src import stress as stress_module
from src import trend as trend_module
from src import universe as universe_module
from src.logging_config import get_logger
from src.schema import DATE_COLUMN, to_close_panel

logger = get_logger(__name__)


def index_features(index_frame: pd.DataFrame) -> pd.DataFrame:
    """Chuỗi chỉ tiêu đầy đủ của VNINDEX: Trend và Stress theo từng phiên."""
    trend = trend_module.trend_frame(index_frame["close"])
    stress = stress_module.stress_frame(index_frame)
    return pd.concat(
        [
            index_frame.reset_index(drop=True),
            trend.reset_index(drop=True),
            stress.reset_index(drop=True),
        ],
        axis=1,
    )


def vn30_features(symbols: Sequence[str]) -> dict:
    """Breadth, dispersion và tập trung rủi ro của rổ VN30 hiện tại."""
    symbols = sorted({s.upper() for s in symbols})
    frames, missing = storage.load_stocks(symbols)
    panel = to_close_panel(frames)
    if missing:
        logger.info("Thiếu dữ liệu giá cho %d mã: %s", len(missing), ", ".join(missing))
    return {
        "panel": panel,
        "missing": missing,
        "breadth": breadth_module.compute_breadth(frames, symbols),
        "dispersion": dispersion_module.compute_dispersion(panel),
        "concentration": concentration_module.compute_concentration(panel),
    }


def build_snapshot(symbols: Sequence[str] | None = None) -> dict:
    """Toàn bộ chỉ tiêu mà các tầng trên cần, đọc từ đĩa. KHÔNG gọi API."""
    meta = universe_module.load_universe()
    symbols = list(symbols or meta["symbols"])

    index_frame = storage.load_index(config.VNINDEX_DATASET)
    if index_frame is None or index_frame.empty:
        logger.warning("Chưa có dữ liệu VNINDEX trong kho dữ liệu")
        return {
            "ready": False,
            "reason": "Chưa có dữ liệu VNINDEX trong kho dữ liệu.",
            "universe": meta,
        }

    vn30 = vn30_features(symbols)
    last_snapshot = storage.read_json(config.VN30_SNAPSHOT_FILE) or {}
    return {
        "ready": True,
        "index": index_frame,
        "vn30_index": storage.load_index(config.VN30_INDEX_DATASET),
        "as_of": pd.to_datetime(index_frame[DATE_COLUMN]).max(),
        "trend": trend_module.trend_snapshot(index_frame["close"]),
        "stress": stress_module.stress_snapshot(index_frame),
        "breadth": vn30["breadth"],
        "dispersion": vn30["dispersion"],
        "concentration": vn30["concentration"],
        "universe": meta,
        "missing_symbols": vn30["missing"],
        "update_log": storage.read_json(config.UPDATE_LOG_FILE) or {},
        # Ảnh chụp Breadth/Dispersion/Concentration của lần cập nhật TRƯỚC lần cập
        # nhật gần nhất (xem ``rebuild``). Dùng để mô tả "đã đổi gì so với lần cập
        # nhật trước" ở lớp diễn giải (``src.narrative``); không dùng để tính lại
        # bất kỳ chỉ tiêu nào.
        "vn30_previous": last_snapshot.get("previous") or {},
    }


def rebuild(symbols: Sequence[str] | None = None) -> dict:
    """Tính lại và ghi processed data. Được gọi ở cuối mỗi lần cập nhật."""
    symbols = list(symbols or universe_module.symbols())
    storage.ensure_dirs()

    index_frame = storage.load_index(config.VNINDEX_DATASET)
    if index_frame is None or index_frame.empty:
        raise RuntimeError(
            "Chưa có dữ liệu VNINDEX để tính chỉ tiêu. "
            "Chạy cập nhật dữ liệu trước khi tính lại features."
        )

    features = index_features(index_frame)
    features.to_parquet(config.VNINDEX_FEATURES_FILE, index=False)
    logger.info("Đã ghi %d dòng chỉ tiêu VNINDEX", len(features))

    vn30 = vn30_features(symbols)
    breadth = vn30["breadth"]
    dispersion = vn30["dispersion"]
    concentration = vn30["concentration"]

    # Ảnh chụp trước khi bị ghi đè, để lớp diễn giải so sánh "đã đổi gì so với
    # lần cập nhật trước". Chỉ giữ vài trường cần cho việc so sánh, không giữ
    # nguyên toàn bộ ảnh chụp cũ.
    existing = storage.read_json(config.VN30_SNAPSHOT_FILE) or {}
    previous_summary = (
        {
            "generated_at": existing.get("generated_at"),
            "as_of": existing.get("as_of"),
            "breadth_score": existing.get("breadth_score"),
            "breadth_state": existing.get("breadth_state"),
            "dispersion": {
                "value": (existing.get("dispersion") or {}).get("value"),
                "state": (existing.get("dispersion") or {}).get("state"),
            },
            "concentration": {
                "top_shares": (existing.get("concentration") or {}).get("top_shares", {}),
                "state": (existing.get("concentration") or {}).get("state"),
            },
        }
        if existing
        else None
    )

    snapshot = {
        "generated_at": storage.utc_now_iso(),
        "as_of": None
        if breadth.get("as_of") is None
        else pd.Timestamp(breadth["as_of"]).strftime("%Y-%m-%d"),
        "universe_size": breadth["universe_size"],
        "valid_symbols": breadth.get("valid_symbols", 0),
        "min_valid_symbols": breadth.get("min_valid_symbols", 0),
        "max_valid_symbols": breadth.get("max_valid_symbols", 0),
        "missing_symbols": breadth["missing_symbols"],
        "breadth_score": breadth["score"],
        "breadth_state": breadth["state"],
        "breadth_components": {
            key: {"pct": value["pct"], "valid": value["valid"], "window": value["window"]}
            for key, value in breadth["components"].items()
        },
        "dispersion": {
            "value": dispersion["value"],
            "percentile": dispersion["percentile"],
            "state": dispersion["state"],
            "windows": dispersion["windows"],
            "historical_basis": dispersion["historical_basis"],
        },
        "concentration": {
            "hhi": concentration["hhi"],
            "effective_names": concentration["effective_names"],
            "top_shares": concentration["top_shares"],
            "percentile": concentration["percentile"],
            "state": concentration["state"],
            "contributors": concentration["contributors"],
            "proxy_note": concentration["proxy_note"],
        },
        "previous": previous_summary,
    }
    storage.write_json(config.VN30_SNAPSHOT_FILE, snapshot)
    logger.info(
        "Đã ghi ảnh chụp VN30: %s/%s mã hợp lệ",
        snapshot["max_valid_symbols"],
        snapshot["universe_size"],
    )
    return snapshot
