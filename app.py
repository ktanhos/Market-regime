"""Trạng thái thị trường Việt Nam — giao diện Streamlit.

Tầng này CHỈ vẽ. Nó không import vnstock, không gọi API và không tính chỉ tiêu.

    Streamlit UI (tệp này)
        ↓  src.features        Feature Layer
        ↓  src.regime          Market Regime Layer
        ↓  src.portfolio_risk  Portfolio Risk Layer

API chỉ được gọi khi người dùng bấm nút trong thanh bên, và lời gọi đó đi qua
``src.updater`` chứ không phải qua tệp này.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import concentration as concentration_module
from src import config, features, quality, storage
from src import dispersion as dispersion_module
from src import portfolio_risk, regime as regime_module
from src import universe as universe_module
from src.github_store import sync_files, token_status
from src.logging_config import get_logger
from src.updater import (
    PHASE_FEATURES,
    PHASE_INDEX,
    PHASE_LABELS,
    PHASE_STOCKS,
    PHASE_SYNC,
    PHASE_UNIVERSE,
    run_update,
)
from src.vnstock_data import connectivity_check, expected_schema, vnstock_version

logger = get_logger(__name__)

st.set_page_config(
    page_title="Trạng thái thị trường Việt Nam",
    page_icon="📊",
    layout="wide",
)

STYLE = """
<style>
:root{
  --ink:#0f172a; --muted:#64748b; --line:#e5e8ee; --surface:#ffffff; --canvas:#f6f7f9;
  --good:#0f7b52; --bad:#c0392f; --warn:#b7791f; --none:#6b7280;
  --good-bg:#e8f4ee; --bad-bg:#fbeceb; --warn-bg:#fdf5e5; --none-bg:#f1f2f4;
}
.stApp{background:var(--canvas);}
.block-container{max-width:1280px;padding-top:1.4rem;padding-bottom:3.5rem;}
h1,h2,h3{color:var(--ink);letter-spacing:-.015em;}
.hero{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:1.7rem 1.9rem;margin-bottom:1rem;}
.hero h1{margin:0 0 .4rem;font-size:1.8rem;line-height:1.25;}
.hero .q{color:var(--muted);font-size:1.02rem;margin-bottom:1.3rem;}
.hero-grid{display:flex;flex-wrap:wrap;gap:2.6rem;}
.hero-item .label{font-size:.71rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);}
.hero-item .value{font-size:1.4rem;font-weight:750;margin-top:.28rem;}
.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:1.1rem 1.2rem;height:100%;}
.card .label{font-size:.71rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);}
.card .value{font-size:1.3rem;font-weight:750;margin:.3rem 0 .4rem;line-height:1.25;}
.card .note{color:var(--muted);font-size:.85rem;line-height:1.6;}
.card .sub{color:var(--ink);font-size:.89rem;line-height:1.55;margin-top:.45rem;}
.regime{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:1.5rem 1.8rem;margin-bottom:1rem;}
.regime .kicker{font-size:.71rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);}
.regime .name{font-size:1.75rem;font-weight:800;margin:.2rem 0 .55rem;}
.regime .body{color:var(--ink);font-size:1rem;line-height:1.75;max-width:74ch;}
.pill{display:inline-block;padding:.24rem .75rem;border-radius:999px;font-size:.77rem;font-weight:700;}
.pill.good{background:var(--good-bg);color:var(--good);}
.pill.bad{background:var(--bad-bg);color:var(--bad);}
.pill.warn{background:var(--warn-bg);color:var(--warn);}
.pill.none{background:var(--none-bg);color:var(--none);}
.good{color:var(--good);} .bad{color:var(--bad);} .warn{color:var(--warn);} .none{color:var(--none);}
.bar{height:7px;border-radius:999px;background:#edf0f4;overflow:hidden;margin:.5rem 0 .35rem;}
.bar > span{display:block;height:100%;border-radius:999px;}
.rowline{display:flex;justify-content:space-between;gap:1rem;font-size:.87rem;padding:.3rem 0;border-bottom:1px dashed var(--line);}
.rowline:last-child{border-bottom:none;}
.rowline .k{color:var(--muted);} .rowline .v{font-weight:650;color:var(--ink);text-align:right;}
.foot{color:var(--muted);font-size:.81rem;line-height:1.65;margin-top:.5rem;}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

GOOD, BAD, WARN, NONE = "good", "bad", "warn", "none"

_TONE: dict[str, str] = {
    regime_module.FAVOURABLE: GOOD,
    regime_module.WARNING: WARN,
    regime_module.TRANSITION: WARN,
    regime_module.UNDER_PRESSURE: BAD,
    regime_module.STRESSED: BAD,
    regime_module.UNKNOWN: NONE,
    "TÍCH CỰC": GOOD, "TRUNG TÍNH": WARN, "SUY YẾU": BAD,
    "THẤP": GOOD, "BÌNH THƯỜNG": GOOD, "CAO": WARN, "RẤT CAO": BAD,
    "RẤT KHỎE": GOOD, "KHỎE": GOOD, "CÂN BẰNG": WARN, "YẾU": BAD, "RẤT YẾU": BAD,
    dispersion_module.DISPERSION_LOW: GOOD,
    dispersion_module.DISPERSION_NORMAL: WARN,
    dispersion_module.DISPERSION_HIGH: BAD,
    concentration_module.CONCENTRATION_LOW: GOOD,
    concentration_module.CONCENTRATION_NORMAL: WARN,
    concentration_module.CONCENTRATION_HIGH: BAD,
    regime_module.RISK_LOW: GOOD,
    regime_module.RISK_MEDIUM: WARN,
    regime_module.RISK_HIGH: BAD,
    regime_module.RISK_VERY_HIGH: BAD,
    regime_module.RISK_UNKNOWN: NONE,
    "CHƯA ĐỦ DỮ LIỆU": NONE,
    "CHƯA XÁC ĐỊNH": NONE,
}

_HEX = {
    GOOD: config.COLOR_GOOD,
    BAD: config.COLOR_BAD,
    WARN: config.COLOR_WARN,
    NONE: config.COLOR_MUTED,
}


# --- Trình bày ---------------------------------------------------------------

def tone(state: object) -> str:
    """Màu theo ý nghĩa, không bao giờ theo trang trí."""
    return _TONE.get(str(state).strip(), NONE)


def fmt(value: object, digits: int = 1, suffix: str = "") -> str:
    """Số liệu thiếu hiện dấu gạch ngang thay vì nan."""
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number:,.{digits}f}{suffix}"


def fmt_date(value: object) -> str:
    if value is None:
        return "—"
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)
    return "—" if pd.isna(stamp) else stamp.strftime("%d/%m/%Y")


def card(label: str, value: str, note: str = "", state: object = None, extra: str = "") -> str:
    css = tone(value if state is None else state)
    return (
        f"<div class='card'><div class='label'>{label}</div>"
        f"<div class='value {css}'>{value}</div>"
        f"<div class='note'>{note}</div>{extra}</div>"
    )


def row(key: str, value: str, css: str = "") -> str:
    return f"<div class='rowline'><span class='k'>{key}</span><span class='v {css}'>{value}</span></div>"


def bar(pct: object, label: str, suffix: str = "") -> str:
    if pct is None or (isinstance(pct, float) and not np.isfinite(pct)) or pd.isna(pct):
        return row(f"{label} {suffix}", "Thiếu dữ liệu", NONE)
    css = GOOD if pct >= 55 else (WARN if pct >= 45 else BAD)
    width = max(0.0, min(100.0, float(pct)))
    return (
        row(f"{label} {suffix}", f"{width:.0f}%", css)
        + f"<div class='bar'><span style='width:{width:.1f}%;background:{_HEX[css]}'></span></div>"
    )


# --- Tải trạng thái (chỉ đọc đĩa) --------------------------------------------

@st.cache_data(show_spinner=False)
def load_market_state() -> dict:
    """Gọi lần lượt ba tầng: Feature → Market Regime → Portfolio Risk."""
    snapshot = features.build_snapshot()
    if not snapshot.get("ready"):
        return snapshot
    regime = regime_module.build_regime(
        snapshot["trend"], snapshot["stress"], snapshot["breadth"],
        snapshot["dispersion"], snapshot["concentration"],
    )
    snapshot["regime"] = regime
    snapshot["portfolio"] = portfolio_risk.guidance(regime)
    return snapshot


@st.cache_data(show_spinner=False)
def load_coverage(symbols: tuple[str, ...]) -> dict:
    return quality.coverage_summary(list(symbols))


# =============================================================================
# Thanh bên
# =============================================================================

def render_sidebar() -> None:
    sb = st.sidebar
    sb.markdown("### DỮ LIỆU THỊ TRƯỜNG")

    meta = universe_module.load_universe()
    summary = load_coverage(tuple(meta["symbols"]))
    sb.markdown(
        row("Ngày dữ liệu mới nhất", fmt_date(summary["index_last_date"]))
        + row("Số phiên VNINDEX", f"{summary['index_sessions']:,}")
        + row(
            "Cổ phiếu VN30",
            f"{summary['stock_symbols_available']}/{summary['stock_symbols_expected']}",
            GOOD if summary["stock_symbols_available"] == summary["stock_symbols_expected"] else WARN,
        )
        + row("Nguồn dữ liệu", f"vnstock {vnstock_version()} · {config.PRIMARY_SOURCE}")
        + row("Cập nhật lần cuối", summary["last_update"][:16].replace("T", " ") or "—"),
        unsafe_allow_html=True,
    )

    if summary["universe_is_fallback"]:
        sb.warning(
            "Danh sách VN30 đang dùng bản dự phòng trong mã nguồn"
            f"{' (' + summary['universe_as_of'] + ')' if summary['universe_as_of'] else ''}. "
            "Bấm Cập nhật dữ liệu để lấy danh sách hiện hành từ API."
        )

    sb.divider()
    token = token_status()
    if not token["configured"]:
        sb.info(token["hint"])

    if sb.button("Kiểm tra API", width="stretch"):
        render_api_check(sb)

    if sb.button("Cập nhật dữ liệu", type="primary", width="stretch"):
        run_update_flow(sb)

    sb.caption(
        "Mở dashboard không gọi API. Dữ liệu chỉ được lấy khi bấm nút cập nhật. "
        "Các mã được gọi tuần tự, có nghỉ giữa các lượt."
    )


def render_api_check(sb) -> None:
    """Chẩn đoán API: VNINDEX, FPT và danh sách VN30, không cần chạy 30 mã."""
    with sb.status("Đang kiểm tra API...", expanded=True) as status:
        probes = connectivity_check()
        for probe in probes:
            mark = "✅" if probe.ok else "❌"
            st.write(f"{mark} **{probe.name}** — {'SUCCESS' if probe.ok else 'FAILED'}")
            if probe.ok:
                st.caption(
                    f"{probe.source} · {probe.rows} dòng"
                    + (f" · {probe.first_date} → {probe.last_date}" if probe.last_date else "")
                )
                st.caption("Schema: " + ", ".join(probe.schema))
            else:
                st.caption(probe.error or probe.detail)
        ok = all(p.ok for p in probes)
        status.update(
            label="API hoạt động bình thường" if ok else "API đang có vấn đề",
            state="complete" if ok else "error",
        )
    st.session_state["api_probes"] = [p.as_dict() for p in probes]
    if not all(p.ok for p in probes):
        sb.warning("Các phép thử cơ bản chưa đạt nên chưa nên chạy cập nhật toàn bộ 30 mã.")
    sb.caption("Schema kỳ vọng: " + ", ".join(expected_schema()))


def run_update_flow(sb) -> None:
    """Chạy pipeline cập nhật với năm thanh tiến trình theo giai đoạn."""
    order = [PHASE_INDEX, PHASE_UNIVERSE, PHASE_STOCKS, PHASE_FEATURES, PHASE_SYNC]
    bars = {phase: sb.progress(0.0, text=PHASE_LABELS[phase]) for phase in order}
    line = sb.empty()

    def progress(phase: str, ratio: float, message: str) -> None:
        line.caption(message)
        if phase in bars:
            bars[phase].progress(min(1.0, max(0.0, float(ratio))), text=PHASE_LABELS[phase])

    try:
        report = run_update(progress=progress)
    except Exception as exc:  # dashboard không được phép sập vì lỗi cập nhật
        logger.exception("Cập nhật thất bại")
        line.empty()
        sb.error(f"Không hoàn tất cập nhật: {type(exc).__name__}: {exc}")
        return

    if report.rate_limited:
        sb.warning(report.aborted_reason)

    # --- Đồng bộ GitHub -------------------------------------------------------
    try:
        result = sync_files(
            storage.data_files(),
            repo=config.GITHUB_REPO,
            branch=config.GITHUB_BRANCH,
            message=f"Cập nhật dữ liệu thị trường đến {fmt_date(pd.Timestamp.today())}",
            root=config.ROOT,
        )
        bars[PHASE_SYNC].progress(1.0, text=PHASE_LABELS[PHASE_SYNC])
        if result.committed:
            sb.success(f"{result.message} Commit {result.commit_sha}.")
        else:
            sb.info(result.message)
    except Exception as exc:
        logger.exception("Đồng bộ GitHub thất bại")
        bars[PHASE_SYNC].progress(1.0, text=PHASE_LABELS[PHASE_SYNC])
        sb.warning(f"Dữ liệu đã lưu cục bộ nhưng chưa đồng bộ lên GitHub: {exc}")

    line.empty()
    latest = storage.last_stored_date(storage.index_path(config.VNINDEX_DATASET))
    headline = f"{report.success_count}/{report.total_count} nguồn · dữ liệu đến ngày {fmt_date(latest)}"
    if report.completed:
        sb.success(f"Cập nhật thành công — {headline}")
    elif report.success_count:
        sb.warning(f"Cập nhật một phần — {headline}")
    else:
        sb.error(f"Cập nhật thất bại — {headline}")

    if report.failures:
        with sb.expander(f"{report.failure_count} nguồn chưa cập nhật được", expanded=not report.success_count):
            st.dataframe(quality.failure_table(report.as_dict()), width="stretch", hide_index=True)

    st.cache_data.clear()
    if report.success_count:
        st.rerun()


# =============================================================================
# Trang chính
# =============================================================================

def render_hero(state: dict) -> None:
    regime = state["regime"]
    st.markdown(
        "<div class='hero'>"
        "<h1>TRẠNG THÁI THỊ TRƯỜNG VIỆT NAM</h1>"
        "<div class='q'>Thị trường đang ở trạng thái nào và điều gì đang tạo nên trạng thái đó?</div>"
        "<div class='hero-grid'>"
        f"<div class='hero-item'><div class='label'>Market Regime</div>"
        f"<div class='value {tone(regime['regime'])}'>{regime['regime']}</div></div>"
        f"<div class='hero-item'><div class='label'>Mức độ rủi ro</div>"
        f"<div class='value {tone(regime['risk_level'])}'>{regime['risk_level']}</div></div>"
        f"<div class='hero-item'><div class='label'>Ngày dữ liệu</div>"
        f"<div class='value'>{fmt_date(state['as_of'])}</div></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_regime(state: dict) -> None:
    regime = state["regime"]
    portfolio = state["portfolio"]
    notes = "".join(f"<li>{note}</li>" for note in regime["risk_reasons"])
    st.markdown(
        "<div class='regime'>"
        "<div class='kicker'>Market Regime</div>"
        f"<div class='name {tone(regime['regime'])}'>{regime['regime']}</div>"
        f"<div class='body'>{regime['description']}</div>"
        f"<div style='margin-top:1rem'><span class='pill {tone(regime['risk_level'])}'>"
        f"Mức độ rủi ro: {regime['risk_level']}</span></div>"
        + (f"<ul class='foot'>{notes}</ul>" if notes else "")
        + f"<div class='body' style='margin-top:1rem'><strong>Quản trị danh mục:</strong> "
        f"{portfolio['risk_budget']}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_cards(state: dict) -> None:
    trend, stress, breadth = state["trend"], state["stress"], state["breadth"]
    dispersion, concentration = state["dispersion"], state["concentration"]

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        card(
            "Xu hướng",
            trend["state"],
            "RORO đo chênh lệch giữa sức mạnh động lượng đa khung và trung bình 49 phiên của chính nó.",
            extra=row("RORO", fmt(trend.get("roro"), 2))
            + row("Vùng trung tính", "±" + fmt(trend.get("band"), 2))
            + row("Strength", fmt(trend.get("strength"), 2, "%")),
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        card(
            "Mức biến động",
            stress["state"],
            "Market Stress đo bằng biến động Parkinson của VNINDEX so với chính nó. Đây là proxy, không phải VIX.",
            extra=row("Parkinson 22 phiên", fmt(stress.get("parkinson_vol"), 1, "%"))
            + row("Phân vị 252 phiên", fmt(stress.get("percentile"), 0) + "/100")
            + row("Z-score", fmt(stress.get("zscore"), 2)),
        ),
        unsafe_allow_html=True,
    )
    components = breadth.get("components", {})
    total = breadth["universe_size"]
    c3.markdown(
        card(
            "Sức khỏe VN30",
            breadth["state"],
            f"Breadth score {fmt(breadth.get('score'), 1)}/100 · "
            f"{breadth.get('valid_symbols', 0)}/{total} mã hợp lệ.",
            extra="".join(
                row(f"Trên MA{w}", fmt(components.get(f'ma{w}', {}).get('pct'), 0, "%"))
                for w in config.BREADTH_MA_WINDOWS
            ),
        ),
        unsafe_allow_html=True,
    )

    c4, c5 = st.columns(2)
    windows = dispersion.get("windows", {})
    c4.markdown(
        card(
            "Phân hóa",
            dispersion["state"],
            dispersion.get("historical_basis", ""),
            extra=row("Dispersion 20 phiên", fmt(dispersion.get("value"), 2, "%"))
            + row("Dispersion 5 phiên", fmt(windows.get(5, {}).get("value"), 2, "%"))
            + row("Dispersion 1 phiên", fmt(windows.get(1, {}).get("value"), 2, "%"))
            + row("Phân vị", fmt(dispersion.get("percentile"), 0) + "/100"),
        ),
        unsafe_allow_html=True,
    )
    top = concentration.get("top_shares", {})
    c5.markdown(
        card(
            "Tập trung rủi ro",
            concentration["state"],
            concentration.get("proxy_note", ""),
            extra=row("Top 5 chiếm", fmt(top.get(5), 1, "%"))
            + row("Top 10 chiếm", fmt(top.get(10), 1, "%"))
            + row(
                "Số mã đóng góp hiệu dụng",
                f"{fmt(concentration.get('effective_names'), 1)}/{concentration.get('contributors', 0)}",
            )
            + row("Herfindahl", fmt(concentration.get("hhi"), 4)),
        ),
        unsafe_allow_html=True,
    )


def render_vn30(state: dict) -> None:
    breadth, concentration = state["breadth"], state["concentration"]
    st.subheader("Nhóm VN30 hiện tại")

    if not breadth.get("sufficient"):
        st.warning(
            f"Chỉ có {breadth.get('max_valid_symbols', 0)}/{breadth['universe_size']} mã VN30 có dữ liệu giá. "
            "Bấm Cập nhật dữ liệu trong thanh bên để tải giá cổ phiếu thành phần."
        )
        if breadth["missing_symbols"]:
            st.caption("Mã chưa có dữ liệu: " + ", ".join(breadth["missing_symbols"]))
        return

    total = breadth["universe_size"]
    rows = ""
    for window in config.BREADTH_MA_WINDOWS:
        item = breadth["components"].get(f"ma{window}", {})
        rows += bar(item.get("pct"), f"Trên MA{window}", f"({item.get('valid', 0)}/{total} mã)")
    for window in config.BREADTH_RETURN_WINDOWS:
        item = breadth["components"].get(f"ret{window}", {})
        label = "Tăng trong phiên" if window == 1 else f"Tăng trong {window} phiên"
        rows += bar(item.get("pct"), label, f"({item.get('valid', 0)}/{total} mã)")

    left, right = st.columns([1.05, 1])
    left.markdown(
        f"<div class='card'><div class='label'>Độ lan tỏa</div>"
        f"<div class='value {tone(breadth['state'])}'>{breadth['state']} · "
        f"{fmt(breadth['score'], 0)}/100</div>{rows}</div>",
        unsafe_allow_html=True,
    )
    right.plotly_chart(_concentration_chart(concentration), width="stretch")

    with st.expander("Chi tiết từng mã VN30"):
        st.dataframe(breadth["table"], width="stretch", hide_index=True)
        st.dataframe(concentration["table"], width="stretch", hide_index=True)


# --- Biểu đồ -----------------------------------------------------------------

def _base_layout(figure: go.Figure, title: str, percent: bool = False) -> go.Figure:
    figure.update_layout(
        height=260,
        margin=dict(l=8, r=8, t=30, b=8),
        title=dict(text=title, font=dict(size=13, color="#475467")),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=False, linecolor="#e5e8ee")
    figure.update_yaxes(gridcolor="#f0f2f5", zeroline=False, ticksuffix="%" if percent else "")
    return figure


def _line_chart(x, y, title: str, color: str, hline: float | None = None, percent: bool = False) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=x, y=y, mode="lines", name=title, line=dict(color=color, width=2)))
    if hline is not None:
        figure.add_hline(y=hline, line_dash="dot", line_color="#98a2b3", line_width=1)
    return _base_layout(figure, title, percent)


def _breadth_chart(breadth: dict, total: int) -> go.Figure:
    labels: list[str] = []
    values: list[object] = []
    colors: list[str] = []
    for window in config.BREADTH_MA_WINDOWS:
        item = breadth["components"].get(f"ma{window}", {})
        labels.append(f"Trên MA{window}")
        values.append(item.get("pct"))
    for window in config.BREADTH_RETURN_WINDOWS:
        item = breadth["components"].get(f"ret{window}", {})
        labels.append("Tăng 1 phiên" if window == 1 else f"Tăng {window} phiên")
        values.append(item.get("pct"))
    clean = [0.0 if v is None or pd.isna(v) else float(v) for v in values]
    for value, raw in zip(clean, values):
        if raw is None or pd.isna(raw):
            colors.append(_HEX[NONE])
        else:
            colors.append(_HEX[GOOD] if value >= 55 else (_HEX[WARN] if value >= 45 else _HEX[BAD]))

    figure = go.Figure(
        go.Bar(x=labels, y=clean, marker_color=colors, text=[f"{v:.0f}%" for v in clean],
               textposition="outside", cliponaxis=False)
    )
    figure.add_hline(y=50, line_dash="dot", line_color="#98a2b3", line_width=1)
    figure = _base_layout(figure, f"Độ lan tỏa VN30 ({total} mã trong rổ)", percent=True)
    figure.update_yaxes(range=[0, 112])
    return figure


def _concentration_chart(concentration: dict) -> go.Figure:
    table = concentration.get("table")
    if table is None or table.empty:
        return _base_layout(go.Figure(), "Tập trung rủi ro biến động", percent=True)
    top = table.head(10).iloc[::-1]
    figure = go.Figure(
        go.Bar(
            x=top["Tỷ trọng rủi ro %"], y=top["Mã"], orientation="h",
            marker_color=config.COLOR_WARN,
            text=[f"{v:.1f}%" for v in top["Tỷ trọng rủi ro %"]],
            textposition="outside", cliponaxis=False,
        )
    )
    figure = _base_layout(figure, "Top 10 mã theo tỷ trọng rủi ro biến động", percent=True)
    figure.update_layout(height=340)
    figure.update_yaxes(gridcolor="white")
    return figure


def render_charts(state: dict) -> None:
    st.subheader("Diễn biến")

    index = state["index"].reset_index(drop=True)
    frames = [index[["date", "close"]]]
    trend_series = state["trend"].get("series")
    stress_series = state["stress"].get("series")
    if trend_series is not None and len(trend_series) == len(index):
        frames.append(trend_series.reset_index(drop=True)[["roro"]])
    if stress_series is not None and len(stress_series) == len(index):
        frames.append(stress_series.reset_index(drop=True)[["parkinson_vol"]])
    chart = pd.concat(frames, axis=1).tail(500)

    c1, c2 = st.columns(2)
    c1.plotly_chart(
        _line_chart(chart["date"], chart["close"], "VNINDEX", config.COLOR_MUTED), width="stretch"
    )
    if "roro" in chart.columns:
        c2.plotly_chart(
            _line_chart(chart["date"], chart["roro"], "RORO — xu hướng tương đối",
                        config.COLOR_GOOD, hline=0.0),
            width="stretch",
        )
    else:
        c2.info("Chưa đủ dữ liệu để vẽ chuỗi RORO.")

    c3, c4 = st.columns(2)
    if "parkinson_vol" in chart.columns:
        c3.plotly_chart(
            _line_chart(chart["date"], chart["parkinson_vol"], "Market Stress — biến động Parkinson 22 phiên",
                        config.COLOR_WARN, percent=True),
            width="stretch",
        )
    else:
        c3.info("Chưa đủ dữ liệu để vẽ chuỗi biến động.")

    breadth = state["breadth"]
    if breadth.get("sufficient"):
        c4.plotly_chart(_breadth_chart(breadth, breadth["universe_size"]), width="stretch")
    else:
        c4.info("Chưa đủ dữ liệu cổ phiếu VN30 để vẽ độ lan tỏa.")

    series = state["dispersion"].get("series")
    if series is not None and len(series) > 5:
        st.plotly_chart(
            _line_chart(series.index, series.values,
                        "Phân hóa 20 phiên trong rổ VN30 hiện tại", config.COLOR_BAD, percent=True),
            width="stretch",
        )
        st.caption(state["dispersion"].get("historical_basis", ""))


def render_portfolio(state: dict) -> None:
    portfolio = state["portfolio"]
    st.subheader("Quản trị danh mục")
    items = [
        ("Mức rủi ro tham chiếu", portfolio["risk_budget"]),
        ("Mức độ thận trọng", portfolio["caution"]),
        ("Kiểm soát đòn bẩy", portfolio["leverage"]),
        ("Mức độ tập trung danh mục", portfolio["concentration"]),
        ("Khả năng duy trì tỷ trọng cổ phiếu", portfolio["equity_weight"]),
    ]
    for column, (label, text) in zip(st.columns(len(items)), items):
        column.markdown(
            f"<div class='card'><div class='label'>{label}</div><div class='sub'>{text}</div></div>",
            unsafe_allow_html=True,
        )
    st.caption(portfolio["disclaimer"])


def render_quality(state: dict) -> None:
    st.subheader("Chất lượng dữ liệu")
    meta = state["universe"]
    summary = load_coverage(tuple(meta["symbols"]))
    log = state.get("update_log") or {}
    indices = summary["indices"]
    available = summary["stock_symbols_available"]
    expected = summary["stock_symbols_expected"]
    missing = summary["stock_symbols_missing"]

    complete = bool(log.get("completed")) and not missing
    detail = (
        row("Ngày dữ liệu", fmt_date(summary["index_last_date"]))
        + row("VNINDEX", f"{indices['VNINDEX']['sessions']:,} phiên · đến {fmt_date(indices['VNINDEX']['last_date'])}")
        + row("VN30", f"{indices['VN30']['sessions']:,} phiên · đến {fmt_date(indices['VN30']['last_date'])}")
        + row("Số mã hợp lệ", f"{available}/{expected}", GOOD if available == expected else WARN)
        + row("Số mã thiếu", str(len(missing)), GOOD if not missing else BAD)
        + row("Nguồn dữ liệu", f"vnstock {log.get('vnstock_version') or vnstock_version()} · {config.PRIMARY_SOURCE}")
        + row("Lần cập nhật cuối", (log.get("finished_at") or "—")[:19].replace("T", " "))
        + row(
            "Kết quả lần cuối",
            f"{log.get('success_count', 0)}/{log.get('total_count', 0)} nguồn" if log else "chưa chạy",
            GOOD if complete else (NONE if not log else WARN),
        )
    )
    st.markdown(
        f"<div class='card'><div class='label'>Tình trạng kho dữ liệu</div>"
        f"<div class='value {GOOD if complete else (NONE if not log else WARN)}'>"
        f"{'Đầy đủ' if complete else ('Chưa cập nhật' if not log else 'Thiếu dữ liệu')}</div>{detail}"
        f"<div class='foot'>Danh sách VN30 chụp ngày {summary['universe_as_of'] or '—'} từ "
        f"{summary['universe_source'] or '—'}"
        f"{' — đang dùng bản dự phòng trong mã nguồn' if summary['universe_is_fallback'] else ''}.</div></div>",
        unsafe_allow_html=True,
    )

    if missing:
        st.warning(f"{len(missing)} mã chưa có dữ liệu giá: " + ", ".join(missing))

    failures = quality.failure_table(log)
    if len(failures):
        st.error(f"{len(failures)} nguồn chưa cập nhật được trong lần chạy gần nhất.")
        st.dataframe(failures, width="stretch", hide_index=True)

    probes = st.session_state.get("api_probes")
    if probes:
        with st.expander("Kết quả kiểm tra API gần nhất"):
            st.dataframe(pd.DataFrame(probes), width="stretch", hide_index=True)

    with st.expander("Chi tiết từng tệp dữ liệu"):
        st.dataframe(quality.dataset_rows(meta["symbols"]), width="stretch", hide_index=True)
        if log.get("datasets"):
            st.caption("Số dòng trước và sau khi hợp nhất của lần cập nhật gần nhất")
            labels = {
                "name": "Mã", "source": "Nguồn", "requested_start": "Yêu cầu từ",
                "requested_end": "Yêu cầu đến", "rows_before_merge": "Dòng trước",
                "rows_after_merge": "Dòng sau", "rows_added": "Dòng thêm", "last_date": "Đến ngày",
            }
            table = pd.DataFrame(log["datasets"]).reindex(columns=list(labels))
            st.dataframe(table.rename(columns=labels), width="stretch", hide_index=True)


def render_empty() -> None:
    st.markdown(
        "<div class='hero'><h1>TRẠNG THÁI THỊ TRƯỜNG VIỆT NAM</h1>"
        "<div class='q'>Thị trường đang ở trạng thái nào và điều gì đang tạo nên trạng thái đó?</div></div>",
        unsafe_allow_html=True,
    )
    st.warning("Chưa có dữ liệu VNINDEX trong kho dữ liệu.")
    st.info("Mở thanh bên, bấm Kiểm tra API rồi bấm Cập nhật dữ liệu để tạo bộ dữ liệu đầu tiên.")


def main() -> None:
    render_sidebar()
    state = load_market_state()
    if not state.get("ready"):
        render_empty()
        return

    render_hero(state)
    render_regime(state)
    render_cards(state)
    render_vn30(state)
    render_charts(state)
    render_portfolio(state)
    render_quality(state)

    st.caption(
        "Ứng dụng mô tả trạng thái thị trường để hỗ trợ quản trị rủi ro danh mục. "
        "Không dự báo giá và không phải khuyến nghị mua bán."
    )


main()
