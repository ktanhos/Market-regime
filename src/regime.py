"""Market Regime: quy tắc tường minh, không dùng học máy.

Đầu vào là năm trạng thái đã tính riêng biệt:

* Trend  (RORO của VNINDEX)
* Stress (biến động Parkinson của VNINDEX so với chính nó)
* Breadth (rổ VN30 hiện tại)
* Dispersion (phân hóa lợi suất trong rổ VN30 hiện tại)
* Risk concentration (proxy tập trung rủi ro biến động)

Trend và Stress quyết định nhãn chế độ vì cả hai dựa trên lịch sử VNINDEX đầy đủ
và không vướng vấn đề thành phần rổ. Breadth tham gia ở các nhánh cần xác nhận
độ lan tỏa. Dispersion và Risk concentration KHÔNG đổi nhãn chế độ vì chúng chỉ
có bối cảnh mô tả trên rổ hiện tại; chúng chỉ điều chỉnh **mức độ rủi ro**.

Đây là mô tả trạng thái thị trường, không phải tín hiệu mua bán.
"""

from __future__ import annotations

import pandas as pd

from src import trend as trend_module
from src import stress as stress_module
from src.breadth import BREADTH_UNKNOWN
from src.concentration import CONCENTRATION_HIGH
from src.dispersion import DISPERSION_HIGH

FAVOURABLE = "THUẬN LỢI"
WARNING = "CẢNH BÁO"
TRANSITION = "CHUYỂN TIẾP"
UNDER_PRESSURE = "CHỊU ÁP LỰC"
STRESSED = "CĂNG THẲNG"
UNKNOWN = "CHƯA ĐỦ DỮ LIỆU"

RISK_LOW = "THẤP"
RISK_MEDIUM = "TRUNG BÌNH"
RISK_HIGH = "CAO"
RISK_VERY_HIGH = "RẤT CAO"
RISK_UNKNOWN = "CHƯA XÁC ĐỊNH"

BREADTH_STRONG = 55.0
BREADTH_WEAK = 45.0

_DESCRIPTIONS = {
    FAVOURABLE: (
        "Xu hướng VNINDEX tích cực, mức biến động chưa vượt ngưỡng thông thường "
        "và nhóm vốn hóa lớn đang đồng thuận."
    ),
    WARNING: (
        "Thị trường vẫn tăng nhưng rủi ro đã cao hơn: biến động tăng lên hoặc "
        "độ lan tỏa của nhóm VN30 đang mỏng dần."
    ),
    TRANSITION: (
        "Xu hướng đang thay đổi nhưng mức biến động chưa xác nhận. "
        "Nhóm cổ phiếu lớn chưa tạo ra sự đồng thuận rõ ràng."
    ),
    UNDER_PRESSURE: (
        "Xu hướng suy yếu đi cùng biến động tăng hoặc độ lan tỏa yếu."
    ),
    STRESSED: (
        "Xu hướng suy yếu và mức biến động đang ở vùng cao nhất so với chính "
        "thị trường này trong một năm qua."
    ),
    UNKNOWN: "Chưa đủ dữ liệu để mô tả trạng thái thị trường.",
}

_BASE_RISK = {
    FAVOURABLE: RISK_LOW,
    WARNING: RISK_MEDIUM,
    TRANSITION: RISK_MEDIUM,
    UNDER_PRESSURE: RISK_HIGH,
    STRESSED: RISK_VERY_HIGH,
    UNKNOWN: RISK_UNKNOWN,
}

_RISK_LADDER = [RISK_LOW, RISK_MEDIUM, RISK_HIGH, RISK_VERY_HIGH]


def _escalate(level: str, steps: int = 1) -> str:
    if level not in _RISK_LADDER:
        return level
    index = min(_RISK_LADDER.index(level) + steps, len(_RISK_LADDER) - 1)
    return _RISK_LADDER[index]


def classify_regime(
    trend_state: str,
    stress_state: str,
    breadth_score: float | None = None,
) -> str:
    """Bảng quyết định tường minh, đọc theo thứ tự từ trên xuống."""
    if trend_state in (None, trend_module.TREND_UNKNOWN) or stress_state in (
        None,
        stress_module.STRESS_UNKNOWN,
    ):
        return UNKNOWN

    breadth_known = breadth_score is not None and not pd.isna(breadth_score)
    breadth_weak = breadth_known and breadth_score < BREADTH_WEAK
    breadth_strong = breadth_known and breadth_score >= BREADTH_STRONG

    high_stress = stress_state in ("CAO", "RẤT CAO")
    extreme_stress = stress_state == "RẤT CAO"
    calm = stress_state in ("THẤP", "BÌNH THƯỜNG")

    if trend_state == trend_module.TREND_WEAK and extreme_stress:
        return STRESSED
    if trend_state == trend_module.TREND_WEAK and (high_stress or breadth_weak):
        return UNDER_PRESSURE
    if trend_state == trend_module.TREND_POSITIVE and calm and breadth_strong:
        return FAVOURABLE
    if trend_state == trend_module.TREND_POSITIVE and (high_stress or breadth_weak):
        return WARNING
    if trend_state == trend_module.TREND_NEUTRAL and extreme_stress:
        return UNDER_PRESSURE
    return TRANSITION


def risk_level(
    regime: str,
    dispersion_state: str | None = None,
    concentration_state: str | None = None,
    breadth_sufficient: bool = True,
) -> tuple[str, list[str]]:
    """Mức độ rủi ro kèm lý do. Phân hóa và tập trung rủi ro chỉ nâng mức, không đổi nhãn chế độ."""
    level = _BASE_RISK.get(regime, RISK_UNKNOWN)
    reasons: list[str] = []

    if level == RISK_UNKNOWN:
        return level, ["Chưa đủ dữ liệu để xác định mức rủi ro."]

    escalations = 0
    if dispersion_state == DISPERSION_HIGH:
        escalations += 1
        reasons.append("Phân hóa lợi suất trong rổ VN30 ở vùng cao so với một năm gần đây.")
    if concentration_state == CONCENTRATION_HIGH:
        escalations += 1
        reasons.append("Rủi ro biến động đang tập trung vào một nhóm nhỏ cổ phiếu.")
    if not breadth_sufficient:
        reasons.append("Không đủ số mã VN30 hợp lệ để đánh giá độ lan tỏa, mức rủi ro được nhìn thận trọng hơn.")
        escalations += 1

    if escalations >= 2:
        level = _escalate(level, 1)
    return level, reasons


def describe(regime: str) -> str:
    return _DESCRIPTIONS.get(regime, _DESCRIPTIONS[UNKNOWN])


def build_regime(
    trend: dict,
    stress: dict,
    breadth: dict,
    dispersion: dict,
    concentration: dict,
) -> dict:
    trend_state = trend.get("state", trend_module.TREND_UNKNOWN)
    stress_state = stress.get("state", stress_module.STRESS_UNKNOWN)
    breadth_score = breadth.get("score")
    breadth_state = breadth.get("state", BREADTH_UNKNOWN)

    regime = classify_regime(trend_state, stress_state, breadth_score)
    level, reasons = risk_level(
        regime,
        dispersion.get("state"),
        concentration.get("state"),
        bool(breadth.get("sufficient", False)),
    )
    return {
        "regime": regime,
        "description": describe(regime),
        "risk_level": level,
        "risk_reasons": reasons,
        "inputs": {
            "trend": trend_state,
            "stress": stress_state,
            "breadth": breadth_state,
            "breadth_score": breadth_score,
            "dispersion": dispersion.get("state"),
            "concentration": concentration.get("state"),
        },
    }
