"""Danh sách VN30 **tại thời điểm hiện tại**.

Nguyên tắc bắt buộc của dự án: 30 mã VN30 hôm nay KHÔNG phải là 30 mã đã thuộc
VN30 trong quá khứ. Vì vậy:

* Danh sách được lưu kèm ngày chụp (``as_of``) và nguồn.
* Không có hàm nào trả về "thành phần VN30 ngày X" trong quá khứ.
* Lịch sử giá của từng mã chỉ dùng để tính chỉ tiêu của chính mã đó
  (MA200, biến động 20 phiên...), không dùng để tái tạo rổ VN30 lịch sử.
"""

from __future__ import annotations

from datetime import date

from src import config, storage

# Danh sách dự phòng dùng khi chưa từng cập nhật và chưa gọi được API.
# Ảnh chụp ngày 2026-08-23 từ Listing(source="VCI").symbols_by_group("VN30").
# Đây vẫn chỉ là ảnh chụp: rổ VN30 được xem xét lại định kỳ nên danh sách này
# sẽ cũ đi. Nguồn đúng là data/reference/vn30_universe.json sau mỗi lần cập nhật.
FALLBACK_VN30 = [
    "ACB", "BID", "BSR", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG", "LPB",
    "MBB", "MCH", "MSN", "MWG", "SAB", "SHB", "SSB", "SSI", "STB", "TCB",
    "TCX", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VPL", "VRE",
]


def load_universe() -> dict:
    """Ảnh chụp VN30 đã lưu, hoặc danh sách dự phòng nếu chưa có."""
    payload = storage.read_json(config.UNIVERSE_FILE)
    if payload and payload.get("symbols"):
        symbols = sorted({str(s).upper() for s in payload["symbols"]})
        return {
            "symbols": symbols,
            "as_of": payload.get("as_of", ""),
            "source": payload.get("source", "không rõ"),
            "is_fallback": bool(payload.get("is_fallback", False)),
        }
    return {
        "symbols": sorted(FALLBACK_VN30),
        "as_of": "",
        "source": "danh sách dự phòng trong mã nguồn",
        "is_fallback": True,
    }


def symbols() -> list[str]:
    return load_universe()["symbols"]


def save_universe(symbol_list, source: str, as_of: str | None = None, is_fallback: bool = False) -> dict:
    payload = {
        "symbols": sorted({str(s).upper() for s in symbol_list}),
        "as_of": as_of or date.today().isoformat(),
        "source": source,
        "is_fallback": is_fallback,
        "note": (
            "Ảnh chụp thành phần VN30 tại ngày as_of. "
            "Không dùng danh sách này để suy ra thành phần VN30 trong quá khứ."
        ),
    }
    storage.write_json(config.UNIVERSE_FILE, payload)
    return payload
