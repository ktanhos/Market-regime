"""Portfolio Risk Layer: chuyển trạng thái thị trường thành thông tin rủi ro.

Không có khuyến nghị mua bán cổ phiếu cụ thể. Không có tỷ trọng cố định kiểu
"70% cổ phiếu, 30% tiền" vì repository chưa có mô hình được kiểm định để đưa ra
con số đó.
"""

from __future__ import annotations

from src import regime as regime_module

_GUIDANCE = {
    regime_module.FAVOURABLE: {
        "risk_budget": "Có thể duy trì mức rủi ro cao hơn trong giới hạn đã đặt ra từ trước.",
        "caution": "Bình thường",
        "leverage": "Đòn bẩy nếu dùng nên nằm trong hạn mức thông thường của danh mục.",
        "concentration": "Có thể chấp nhận mức tập trung danh mục thông thường.",
        "equity_weight": "Có cơ sở để duy trì tỷ trọng cổ phiếu ở vùng cao trong khoảng của riêng danh mục.",
    },
    regime_module.WARNING: {
        "risk_budget": "Giữ nguyên mức rủi ro hiện tại và tránh gia tăng thêm.",
        "caution": "Nâng cao",
        "leverage": "Hạn chế mở rộng đòn bẩy khi biến động đang tăng.",
        "concentration": "Giảm bớt mức tập trung vào một nhóm cổ phiếu hẹp.",
        "equity_weight": "Duy trì tỷ trọng cổ phiếu hiện tại, ưu tiên quan sát thay vì mở rộng.",
    },
    regime_module.TRANSITION: {
        "risk_budget": "Duy trì mức rủi ro vừa phải.",
        "caution": "Trung bình",
        "leverage": "Đòn bẩy nên ở mức thấp cho tới khi xu hướng rõ hơn.",
        "concentration": "Giữ danh mục ở mức phân tán vừa phải.",
        "equity_weight": "Duy trì tỷ trọng cổ phiếu ở vùng trung tính của khoảng đã đặt ra.",
    },
    regime_module.UNDER_PRESSURE: {
        "risk_budget": "Giảm mức rủi ro và kiểm soát chặt tỷ trọng.",
        "caution": "Cao",
        "leverage": "Ưu tiên đưa đòn bẩy về mức tối thiểu.",
        "concentration": "Giảm tập trung vào các mã có biến động cao nhất.",
        "equity_weight": "Khả năng duy trì tỷ trọng cổ phiếu ở mức cao là thấp.",
    },
    regime_module.STRESSED: {
        "risk_budget": "Ưu tiên bảo toàn vốn và giảm rủi ro.",
        "caution": "Rất cao",
        "leverage": "Không mở rộng đòn bẩy.",
        "concentration": "Ưu tiên phân tán và giảm các vị thế biến động mạnh.",
        "equity_weight": "Khả năng duy trì tỷ trọng cổ phiếu cao là rất hạn chế.",
    },
    regime_module.UNKNOWN: {
        "risk_budget": "Chưa đủ dữ liệu để đưa ra tham chiếu rủi ro.",
        "caution": "Chưa xác định",
        "leverage": "Chưa đủ dữ liệu.",
        "concentration": "Chưa đủ dữ liệu.",
        "equity_weight": "Chưa đủ dữ liệu.",
    },
}

DISCLAIMER = (
    "Đây là thông tin tham chiếu cho việc quản trị rủi ro danh mục, "
    "không phải khuyến nghị mua bán và không phải dự báo giá."
)


def guidance(regime_result: dict) -> dict:
    regime = regime_result.get("regime", regime_module.UNKNOWN)
    payload = dict(_GUIDANCE.get(regime, _GUIDANCE[regime_module.UNKNOWN]))
    payload["regime"] = regime
    payload["risk_level"] = regime_result.get("risk_level", regime_module.RISK_UNKNOWN)
    payload["notes"] = list(regime_result.get("risk_reasons", []))
    payload["disclaimer"] = DISCLAIMER
    return payload
