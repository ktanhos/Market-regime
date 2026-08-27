"""Interpretation Layer: dịch chỉ tiêu định lượng sang ngôn ngữ phổ thông.

Tầng này KHÔNG tính lại bất kỳ chỉ tiêu nào và không thêm ngưỡng mới. Nó chỉ
đọc các dict đã được Feature Layer / Market Regime Layer / Portfolio Risk
Layer tính sẵn, rồi trả lời bốn câu hỏi cho người xem không có nền tảng thống
kê: đang ở đâu, vì sao, đã đổi gì so với lần cập nhật trước, và cần theo dõi
điều gì để biết khi nào đánh giá lại.

    Feature Layer -> Market Regime Layer -> Portfolio Risk Layer
        ↓
    Interpretation Layer (module này)
        ↓
    Streamlit UI (app.py)

``app.py`` là nơi duy nhất được import module này để vẽ; bản thân module
không phụ thuộc Streamlit nên có thể kiểm thử độc lập.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import breadth as breadth_module
from src import concentration as concentration_module
from src import dispersion as dispersion_module
from src import regime as regime_module
from src import stress as stress_module
from src import trend as trend_module
from src import config

# Số phiên dùng để mô tả thay đổi gần đây của các chuỗi có sẵn (Trend, Stress).
RECENT_SESSIONS = 5


def _num(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _series_change(series: pd.Series | None, periods: int = RECENT_SESSIONS) -> tuple[float | None, float | None]:
    """(giá trị hiện tại, giá trị cách đây ``periods`` phiên) từ một chuỗi có sẵn."""
    if series is None or len(series) <= periods:
        return None, None
    clean = series.dropna()
    if len(clean) <= periods:
        return None, None
    return _num(clean.iloc[-1]), _num(clean.iloc[-1 - periods])


def _direction_word(delta: float, unit: str = "") -> str:
    if delta > 0:
        return f"tăng {abs(delta):.1f}{unit}"
    if delta < 0:
        return f"giảm {abs(delta):.1f}{unit}"
    return "gần như không đổi"


# --- Xu hướng (Trend) ---------------------------------------------------------

_TREND_PLAIN = {
    trend_module.TREND_POSITIVE: "Giá đang có xu hướng đi lên rõ so với chính nó gần đây.",
    trend_module.TREND_NEUTRAL: "Giá đang đi ngang, chưa nghiêng hẳn về hướng nào.",
    trend_module.TREND_WEAK: "Giá đang có xu hướng đi xuống rõ so với chính nó gần đây.",
    trend_module.TREND_UNKNOWN: "Chưa đủ lịch sử giá để xác định xu hướng.",
}


def trend_story(trend: dict) -> dict:
    state = trend.get("state", trend_module.TREND_UNKNOWN)
    band = _num(trend.get("band"))
    series = trend.get("series")
    roro_series = series["roro"] if isinstance(series, pd.DataFrame) and "roro" in series else None
    cur, past = _series_change(roro_series)

    change = "Chưa đủ dữ liệu để so sánh."
    if cur is not None and past is not None:
        change = (
            f"So với {RECENT_SESSIONS} phiên trước, động lượng xu hướng "
            f"{_direction_word(cur - past)}."
        )

    watch = (
        "Theo dõi xem động lượng có vượt ra khỏi vùng trung tính "
        f"(±{band:.1f} điểm) hay không — vượt lên trên là xu hướng tích cực, "
        "vượt xuống dưới là xu hướng suy yếu."
        if band is not None
        else "Chưa đủ dữ liệu để nêu vùng cần theo dõi."
    )

    return {
        "title": "Xu hướng",
        "verdict": state,
        "plain": _TREND_PLAIN.get(state, _TREND_PLAIN[trend_module.TREND_UNKNOWN]),
        "why": (
            "Đo bằng cách gộp tốc độ tăng/giảm giá ở nhiều khung thời gian rồi so "
            "với mức trung bình 49 phiên gần nhất của chính chỉ tiêu này — nên đây "
            "là so sánh thị trường với chính nó, không phải một ngưỡng cố định."
        ),
        "change": change,
        "watch": watch,
    }


# --- Biến động (Stress) --------------------------------------------------------

_STRESS_PLAIN = {
    "THẤP": "Biến động giá đang thấp hơn phần lớn thời gian của một năm qua.",
    "BÌNH THƯỜNG": "Biến động giá đang ở vùng thường thấy, không có gì bất thường.",
    "CAO": "Biến động giá đang cao hơn phần lớn thời gian của một năm qua.",
    "RẤT CAO": "Biến động giá đang ở vùng cao nhất so với một năm qua — thị trường đang dao động mạnh.",
    stress_module.STRESS_UNKNOWN: "Chưa đủ lịch sử để xác định mức biến động.",
}


def stress_story(stress: dict) -> dict:
    state = stress.get("state", stress_module.STRESS_UNKNOWN)
    percentile = _num(stress.get("percentile"))
    series = stress.get("series")
    cur, past = _series_change(series["stress_index"] if isinstance(series, pd.DataFrame) and "stress_index" in series else None)

    change = "Chưa đủ dữ liệu để so sánh."
    if cur is not None and past is not None:
        change = (
            f"So với {RECENT_SESSIONS} phiên trước, mức biến động "
            f"{_direction_word(cur - past, ' điểm %')}."
        )

    watch = (
        f"Hiện đang ở phân vị {percentile:.0f}/100 so với một năm qua. "
        "Mức RẤT CAO (phân vị ≥ 85) là ngưỡng đáng chú ý nhất vì thường đi kèm "
        "biến động giá lớn theo cả hai chiều."
        if percentile is not None
        else "Chưa đủ dữ liệu để nêu vùng cần theo dõi."
    )

    return {
        "title": "Mức biến động",
        "verdict": state,
        "plain": _STRESS_PLAIN.get(state, _STRESS_PLAIN[stress_module.STRESS_UNKNOWN]),
        "why": (
            "Đo bằng biên độ dao động cao/thấp mỗi phiên trong 22 phiên gần nhất, "
            "rồi xếp hạng so với chính chuỗi biến động của thị trường này trong một "
            "năm qua. Đây là chỉ số biến động dạng ước lượng (proxy), không phải VIX."
        ),
        "change": change,
        "watch": watch,
    }


# --- Độ lan tỏa (Breadth) ------------------------------------------------------

_BREADTH_PLAIN = {
    "RẤT KHỎE": "Gần như toàn bộ cổ phiếu vốn hóa lớn đang cùng đi lên — mức tăng có sự đồng thuận rộng.",
    "KHỎE": "Phần lớn cổ phiếu vốn hóa lớn đang đi lên cùng nhau.",
    "CÂN BẰNG": "Số mã tăng và số mã yếu đi tương đối cân bằng.",
    "YẾU": "Chỉ một phần nhỏ cổ phiếu vốn hóa lớn còn giữ được đà tăng.",
    "RẤT YẾU": "Rất ít cổ phiếu vốn hóa lớn còn giữ được đà tăng — mức tăng/giảm của chỉ số không được nhiều mã ủng hộ.",
}


def breadth_story(breadth: dict, previous: dict | None = None) -> dict:
    data_state = breadth.get("data_state", breadth_module.DATA_NONE)
    state = breadth.get("state", breadth_module.BREADTH_NO_DATA)
    score = _num(breadth.get("score"))
    total = breadth.get("universe_size", 0)
    valid = breadth.get("valid_symbols", 0)

    if data_state == breadth_module.DATA_NONE:
        plain = "Chưa có dữ liệu giá của nhóm VN30 để đánh giá độ lan tỏa."
    elif data_state == breadth_module.DATA_INSUFFICIENT:
        plain = f"Mới có {valid}/{total} mã đủ lịch sử, chưa đủ để đánh giá độ lan tỏa đáng tin cậy."
    else:
        plain = _BREADTH_PLAIN.get(state, "Chưa đủ dữ liệu để mô tả độ lan tỏa.")

    change = "Chưa có lần cập nhật trước để so sánh."
    prev_score = _num((previous or {}).get("breadth_score"))
    if score is not None and prev_score is not None:
        change = (
            f"So với lần cập nhật dữ liệu trước, điểm độ lan tỏa "
            f"{_direction_word(score - prev_score, ' điểm')} (trên thang 100)."
        )

    watch = (
        f"Điểm hiện tại là {score:.0f}/100 trên {valid}/{total} mã đủ dữ liệu. "
        "Dưới 45 điểm là vùng yếu, từ 55 điểm trở lên là vùng khỏe — đây cũng là "
        "ranh giới được dùng để xác nhận Market Regime."
        if score is not None
        else "Chưa đủ dữ liệu để nêu vùng cần theo dõi."
    )

    return {
        "title": "Sức khỏe nhóm VN30",
        "verdict": state,
        "plain": plain,
        "why": (
            "Đo bằng tỷ lệ % số mã VN30 đang nằm trên các đường trung bình động "
            "của chính mã đó (20/50/200 phiên) và tỷ lệ % số mã tăng giá trong "
            "1/5/20 phiên gần nhất, rồi lấy trung bình các tỷ lệ này."
        ),
        "change": change,
        "watch": watch,
    }


# --- Phân hóa (Dispersion) ------------------------------------------------------

_DISPERSION_PLAIN = {
    dispersion_module.DISPERSION_LOW: "Các cổ phiếu trong rổ đang biến động khá giống nhau — ít mã đi ngược dòng.",
    dispersion_module.DISPERSION_NORMAL: "Mức khác biệt lợi suất giữa các cổ phiếu đang ở vùng thường thấy.",
    dispersion_module.DISPERSION_HIGH: "Các cổ phiếu trong rổ đang biến động rất khác nhau — có mã tăng mạnh trong khi mã khác giảm mạnh.",
    dispersion_module.DISPERSION_UNKNOWN: "Chưa đủ dữ liệu để đánh giá mức phân hóa.",
}


def dispersion_story(dispersion: dict, previous: dict | None = None) -> dict:
    state = dispersion.get("state", dispersion_module.DISPERSION_UNKNOWN)
    value = _num(dispersion.get("value"))

    change = "Chưa có lần cập nhật trước để so sánh."
    prev_value = _num((previous or {}).get("dispersion", {}).get("value") if previous else None)
    if value is not None and prev_value is not None:
        change = (
            f"So với lần cập nhật dữ liệu trước, mức phân hóa "
            f"{_direction_word(value - prev_value, ' điểm %')}."
        )

    return {
        "title": "Phân hóa trong rổ VN30",
        "verdict": state,
        "plain": _DISPERSION_PLAIN.get(state, _DISPERSION_PLAIN[dispersion_module.DISPERSION_UNKNOWN]),
        "why": (
            "Đo độ lệch chuẩn của lợi suất 20 phiên giữa các mã trong rổ VN30 hiện "
            "tại, so với phân vị của chính chuỗi này trong một năm gần nhất."
        ),
        "change": change,
        "watch": (
            "Phân hóa cao đồng nghĩa chọn sai cổ phiếu có thể lệch nhiều so với chỉ "
            "số chung — đây là lúc mức độ tập trung danh mục càng quan trọng."
        ),
    }


# --- Tập trung rủi ro (Concentration) -------------------------------------------

_CONCENTRATION_PLAIN = {
    concentration_module.CONCENTRATION_LOW: "Rủi ro biến động trải khá đều giữa các mã trong rổ.",
    concentration_module.CONCENTRATION_NORMAL: "Mức tập trung rủi ro đang ở vùng thường thấy.",
    concentration_module.CONCENTRATION_HIGH: "Rủi ro biến động đang dồn vào một nhóm nhỏ cổ phiếu trong rổ.",
    concentration_module.CONCENTRATION_UNKNOWN: "Chưa đủ dữ liệu để đánh giá mức tập trung.",
}


def concentration_story(concentration: dict, previous: dict | None = None) -> dict:
    state = concentration.get("state", concentration_module.CONCENTRATION_UNKNOWN)
    top5 = _num(concentration.get("top_shares", {}).get(5))

    change = "Chưa có lần cập nhật trước để so sánh."
    prev_top5 = None
    if previous:
        prev_top5 = _num((previous.get("concentration", {}) or {}).get("top_shares", {}).get("5"))
        if prev_top5 is None:
            prev_top5 = _num((previous.get("concentration", {}) or {}).get("top_shares", {}).get(5))
    if top5 is not None and prev_top5 is not None:
        change = (
            f"So với lần cập nhật dữ liệu trước, tỷ trọng rủi ro của 5 mã biến "
            f"động mạnh nhất {_direction_word(top5 - prev_top5, ' điểm %')}."
        )

    return {
        "title": "Tập trung rủi ro",
        "verdict": state,
        "plain": _CONCENTRATION_PLAIN.get(state, _CONCENTRATION_PLAIN[concentration_module.CONCENTRATION_UNKNOWN]),
        "why": (
            "Xếp hạng các mã theo biến động 20 phiên rồi tính tỷ trọng rủi ro biến "
            "động (không phải vốn hóa) của nhóm 5 và 10 mã biến động mạnh nhất. Đây "
            "là proxy mô tả rổ VN30, không phải rủi ro của một danh mục cụ thể."
        ),
        "change": change,
        "watch": (
            f"Top 5 mã đang chiếm {top5:.0f}% tỷ trọng rủi ro biến động của cả rổ."
            if top5 is not None
            else "Chưa đủ dữ liệu để nêu vùng cần theo dõi."
        ),
    }


# --- Market Regime tổng hợp -----------------------------------------------------

_REGIME_WATCH = {
    regime_module.FAVOURABLE: (
        "Trạng thái này thường thay đổi khi biến động tăng lên hoặc độ lan tỏa "
        "của nhóm VN30 mỏng dần — theo dõi hai chỉ tiêu Mức biến động và Sức khỏe "
        "VN30 ở trên."
    ),
    regime_module.WARNING: (
        "Theo dõi xem biến động có tiếp tục tăng hay độ lan tỏa có yếu thêm không — "
        "nếu cả hai cùng xấu đi, trạng thái có thể chuyển sang Chịu áp lực."
    ),
    regime_module.TRANSITION: (
        "Đây là trạng thái chưa rõ ràng. Theo dõi xu hướng có xác nhận rõ hướng "
        "hơn không, và biến động có tăng vọt hay không."
    ),
    regime_module.UNDER_PRESSURE: (
        "Theo dõi xem biến động có tiếp tục leo thang tới vùng RẤT CAO hay không — "
        "đó là điều kiện để trạng thái chuyển sang Căng thẳng."
    ),
    regime_module.STRESSED: (
        "Theo dõi xem biến động có bắt đầu hạ nhiệt và xu hướng có ngừng suy yếu "
        "hay không — đó là các dấu hiệu sớm nhất cho một sự cải thiện."
    ),
    regime_module.UNKNOWN: "Cần thêm dữ liệu lịch sử trước khi có thể theo dõi điều kiện chuyển trạng thái.",
}


def _regime_change(state: dict, current_regime: str) -> str:
    """Điều gì đã đổi so với lần cập nhật dữ liệu trước, không phải so với vài phiên gần đây.

    Dùng lại đúng bảng quyết định của Market Regime Layer (``regime_module.classify_regime``)
    trên các đầu vào của lần cập nhật TRƯỚC để biết trạng thái đó là gì — không thêm ngưỡng
    hay công thức mới, chỉ áp lại quy tắc đã có cho một bộ đầu vào khác.
    """
    previous = state.get("vn30_previous") or {}
    prev_trend = previous.get("trend_state")
    prev_stress = previous.get("stress_state")
    prev_breadth_score = _num(previous.get("breadth_score"))

    if not previous or not prev_trend or not prev_stress:
        return "Đây là lần đầu có đủ dữ liệu để mô tả, chưa có lần cập nhật trước để so sánh."

    prev_regime = regime_module.classify_regime(prev_trend, prev_stress, prev_breadth_score)

    parts = []
    if prev_regime != current_regime:
        parts.append(f"Market Regime đã chuyển từ {prev_regime} sang {current_regime}.")
    else:
        parts.append(f"Market Regime vẫn giữ nguyên ở {current_regime} so với lần cập nhật trước.")

    trend_state = state["trend"].get("state")
    if trend_state and trend_state != prev_trend:
        parts.append(f"Xu hướng đổi từ {prev_trend} sang {trend_state}.")

    stress_state = state["stress"].get("state")
    if stress_state and stress_state != prev_stress:
        parts.append(f"Mức biến động đổi từ {prev_stress} sang {stress_state}.")

    breadth_score = _num(state["breadth"].get("score"))
    if breadth_score is not None and prev_breadth_score is not None:
        delta = breadth_score - prev_breadth_score
        if abs(delta) >= 0.5:
            parts.append(
                f"Độ lan tỏa VN30 {_direction_word(delta, ' điểm')} so với lần cập nhật trước "
                f"({prev_breadth_score:.0f} → {breadth_score:.0f}/100)."
            )

    return " ".join(parts)


def regime_story(state: dict) -> dict:
    """Tóm tắt một khổ cho người không có nền tảng thống kê: đang ở đâu, vì sao."""
    regime = state["regime"]
    trend, stress, breadth = state["trend"], state["stress"], state["breadth"]

    trend_word = {
        trend_module.TREND_POSITIVE: "xu hướng tích cực",
        trend_module.TREND_NEUTRAL: "xu hướng trung tính",
        trend_module.TREND_WEAK: "xu hướng suy yếu",
    }.get(trend.get("state"), "xu hướng chưa rõ")

    stress_word = {
        "THẤP": "biến động thấp",
        "BÌNH THƯỜNG": "biến động bình thường",
        "CAO": "biến động cao",
        "RẤT CAO": "biến động rất cao",
    }.get(stress.get("state"), "biến động chưa rõ")

    breadth_score = _num(breadth.get("score"))
    if breadth.get("sufficient") and breadth_score is not None:
        breadth_word = f"độ lan tỏa {breadth.get('state', '').lower()} ({breadth_score:.0f}/100 điểm)"
    else:
        breadth_word = "chưa đủ dữ liệu về độ lan tỏa"

    summary = f"Thị trường hiện có {trend_word}, {stress_word}, và {breadth_word}."

    return {
        "headline": regime["description"],
        "summary": summary,
        "change": _regime_change(state, regime["regime"]),
        "watch": _REGIME_WATCH.get(regime["regime"], _REGIME_WATCH[regime_module.UNKNOWN]),
    }


def build_narrative(state: dict) -> dict:
    """Toàn bộ lớp diễn giải cho một snapshot đã sẵn sàng (``state['ready']``)."""
    previous = state.get("vn30_previous") or {}
    return {
        "regime": regime_story(state),
        "trend": trend_story(state["trend"]),
        "stress": stress_story(state["stress"]),
        "breadth": breadth_story(state["breadth"], previous),
        "dispersion": dispersion_story(state["dispersion"], previous),
        "concentration": concentration_story(state["concentration"], previous),
    }
