"""Trạng thái thị trường Việt Nam.

Ứng dụng mô tả trạng thái hiện tại của thị trường và chuyển trạng thái đó thành
thông tin tham chiếu cho quản trị rủi ro danh mục.

Đây không phải công cụ dự báo giá và không phải hệ thống tín hiệu mua bán.

Nguyên tắc vận hành: mở dashboard KHÔNG gọi API. API chỉ được gọi khi người dùng
chủ động bấm nút trong thanh bên.
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
from src import config, quality, storage
from src import dispersion as dispersion_module
from src import regime as regime_module
from src import universe as universe_module
from src.github_store import sync_files, token_status
from src.updater import PHASE_FEATURES, PHASE_INDEX, PHASE_STOCKS, PHASE_UNIVERSE, build_market_state, run_update
from src.vnstock_data import connectivity_check, vnstock_version

st.set_page_config(
    page_title="Trạng thái thị trường Việt Nam",
    page_icon="📊",
    layout="wide",
)

STYLE = """
<style>
:root{
  --ink:#101828; --muted:#667085; --line:#e6e9ef; --surface:#ffffff; --canvas:#f7f8fa;
  --good:#0f7b52; --bad:#c0392f; --warn:#b7791f; --none:#6b7280;
  --good-bg:#e9f5ef; --bad-bg:#fbecea; --warn-bg:#fdf4e3; --none-bg:#f1f2f4;
}
.stApp{background:var(--canvas);}
.block-container{max-width:1240px;padding-top:1.4rem;padding-bottom:3rem;}
h1,h2,h3{color:var(--ink);letter-spacing:-.01em;}
.hero{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:1.6rem 1.8rem;margin-bottom:1.1rem;}
.hero h1{margin:0 0 .35rem;font-size:1.85rem;}
.hero .q{color:var(--muted);font-size:1rem;margin-bottom:1.1rem;}
.hero-grid{display:flex;flex-wrap:wrap;gap:2.4rem;}
.hero-item .label{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);}
.hero-item .value{font-size:1.35rem;font-weight:700;margin-top:.25rem;}
.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:1.15rem 1.25rem;height:100%;}
.card .label{font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);}
.card .value{font-size:1.35rem;font-weight:700;margin:.3rem 0 .35rem;}
.card .note{color:var(--muted);font-size:.86rem;line-height:1.55;}
.card .sub{color:var(--ink);font-size:.9rem;margin-top:.5rem;}
.regime{background:var(--surface);border:1px solid var(--line);border-radius:18px;padding:1.5rem 1.7rem;margin-bottom:1.1rem;}
.regime .name{font-size:1.7rem;font-weight:800;margin:.15rem 0 .5rem;}
.regime .body{color:var(--ink);font-size:1rem;line-height:1.7;max-width:70ch;}
.pill{display:inline-block;padding:.22rem .7rem;border-radius:999px;font-size:.78rem;font-weight:700;}
.good{color:var(--good);} .bad{color:var(--bad);} .warn{color:var(--warn);} .none{color:var(--none);}
.pill.good{background:var(--good-bg);color:var(--good);}
.pill.bad{background:var(--bad-bg);color:var(--bad);}
.pill.warn{background:var(--warn-bg);color:var(--warn);}
.pill.none{background:var(--none-bg);color:var(--none);}
.bar{height:8px;border-radius:999px;background:#eef0f4;overflow:hidden;margin:.55rem 0 .3rem;}
.bar > span{display:block;height:100%;border-radius:999px;}
.rowline{display:flex;justify-content:space-between;font-size:.88rem;padding:.32rem 0;border-bottom:1px dashed var(--line);}
.rowline:last-child{border-bottom:none;}
.rowline .k{color:var(--muted);} .rowline .v{font-weight:600;color:var(--ink);}
.foot{color:var(--muted);font-size:.82rem;line-height:1.6;margin-top:.4rem;}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

GOOD, BAD, WARN, NONE = "good", "bad", "warn", "none"

_TONE = {
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

_HEX = {GOOD: config.COLOR_GOOD, BAD: config.COLOR_BAD, WARN: config.COLOR_WARN, NONE: config.COLOR_MUTED}


def tone(state) -> str:
    return _TONE.get(str(state).strip(), NONE)


def fmt(value, digits: int = 1, suffix: str = "") -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)) or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}{suffix}"


def fmt_date(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        return pd.Timestamp(value).strftime("%d/%m/%Y")
    except Exception:
        return str(value)


def card(label: str, value: str, note: str = "", state=None, extra: str = "") -> str:
    css = tone(value if state is None else state)
    return (
        f"<div class='card'><div class='label'>{label}</div>"
        f"<div class='value {css}'>{value}</div>"
        f"<div class='note'>{note}</div>{extra}</div>"
    )


def bar(pct, label: str, valid: str = "") -> str:
    if pct is None or pd.isna(pct):
        return f"<div class='rowline'><span class='k'>{label}</span><span class='v none'>Thiếu dữ liệu</span></div>"
    css = GOOD if pct >= 55 else (WARN if pct >= 45 else BAD)
    width = max(0.0, min(100.0, float(pct)))
    return (
        f"<div class='rowline'><span class='k'>{label} {valid}</span>"
        f"<span class='v {css}'>{width:.0f}%</span></div>"
        f"<div class='bar'><span style='width:{width:.1f}%;background:{_HEX[css]}'></span></div>"
    )


# =============================================================================
# Thanh bên: trạng thái dữ liệu, kiểm tra API, cập nhật dữ liệu
# =============================================================================

@st.cache_data(show_spinner=False)
def load_state():
    return build_market_state()


@st.cache_data(show_spinner=False)
def load_coverage(symbols: tuple[str, ...]):
    return quality.coverage_summary(list(symbols))


def render_sidebar() -> None:
    sb = st.sidebar
    sb.markdown("### Dữ liệu")

    meta = universe_module.load_universe()
    summary = load_coverage(tuple(meta["symbols"]))
    sb.markdown(
        f"<div class='rowline'><span class='k'>Ngày mới nhất</span>"
        f"<span class='v'>{fmt_date(summary['index_last_date'])}</span></div>"
        f"<div class='rowline'><span class='k'>Số phiên VNINDEX</span>"
        f"<span class='v'>{summary['index_sessions']:,}</span></div>"
        f"<div class='rowline'><span class='k'>Cổ phiếu VN30</span>"
        f"<span class='v'>{summary['stock_symbols_available']}/{summary['stock_symbols_expected']}</span></div>"
        f"<div class='rowline'><span class='k'>Nguồn dữ liệu</span>"
        f"<span class='v'>vnstock {vnstock_version()} · {config.PRIMARY_SOURCE}</span></div>"
        f"<div class='rowline'><span class='k'>Cập nhật gần nhất</span>"
        f"<span class='v'>{summary['last_update'][:16].replace('T', ' ') or '—'}</span></div>",
        unsafe_allow_html=True,
    )

    sb.divider()
    token = token_status()
    if not token["configured"]:
        sb.info(token["hint"])

    if sb.button("Kiểm tra kết nối API", width="stretch"):
        with sb.status("Đang thử VNINDEX và FPT...", expanded=True) as status:
            results = connectivity_check()
            for row in results:
                if row["ok"]:
                    st.write(
                        f"✅ {row['symbol']} · {row['source']} · {row['rows']} phiên · đến {row['last_date']}"
                    )
                else:
                    st.write(f"❌ {row['symbol']} · {row['message']}")
            ok = all(r["ok"] for r in results)
            status.update(
                label="Kết nối API bình thường" if ok else "Không kết nối được nguồn dữ liệu",
                state="complete" if ok else "error",
            )
            st.session_state["api_ok"] = ok
        if not st.session_state.get("api_ok", False):
            sb.warning("Hai phép thử cơ bản chưa chạy được nên chưa nên cập nhật toàn bộ 30 mã.")

    if sb.button("Cập nhật dữ liệu", type="primary", width="stretch"):
        run_update_flow(sb)

    sb.caption(
        "Mở dashboard không gọi API. Dữ liệu chỉ được lấy về khi bấm nút cập nhật. "
        "Các mã được gọi tuần tự, có nghỉ giữa các lượt."
    )


def run_update_flow(sb) -> None:
    bars = {
        PHASE_UNIVERSE: sb.progress(0.0, text="Danh sách VN30"),
        PHASE_INDEX: sb.progress(0.0, text="Chỉ số VNINDEX / VN30"),
        PHASE_STOCKS: sb.progress(0.0, text="Cổ phiếu VN30"),
        PHASE_FEATURES: sb.progress(0.0, text="Tính toán"),
    }
    sync_bar = sb.progress(0.0, text="Đồng bộ dữ liệu")
    line = sb.empty()

    def progress(phase: str, ratio: float, message: str) -> None:
        line.caption(message)
        if phase == PHASE_UNIVERSE:
            bars[PHASE_UNIVERSE].progress(1.0, text="Danh sách VN30")
            return
        if phase == PHASE_INDEX:
            bars[PHASE_INDEX].progress(min(1.0, max(0.05, ratio * 6)), text="Chỉ số VNINDEX / VN30")
            return
        if phase == PHASE_STOCKS:
            bars[PHASE_INDEX].progress(1.0, text="Chỉ số VNINDEX / VN30")
            bars[PHASE_STOCKS].progress(min(1.0, max(0.02, ratio)), text="Cổ phiếu VN30")
            return
        bars[PHASE_STOCKS].progress(1.0, text="Cổ phiếu VN30")
        bars[PHASE_FEATURES].progress(min(1.0, ratio), text="Tính toán")

    try:
        report = run_update(progress=progress)
    except Exception as exc:  # dashboard không được phép sập vì lỗi cập nhật
        line.empty()
        sb.error(f"Không hoàn tất cập nhật: {type(exc).__name__}: {exc}")
        return

    for key in bars:
        bars[key].progress(1.0)

    if report.rate_limited:
        sb.warning(report.aborted_reason)

    try:
        result = sync_files(
            storage.data_files(),
            repo=config.GITHUB_REPO,
            branch=config.GITHUB_BRANCH,
            message=f"Cập nhật dữ liệu thị trường đến {fmt_date(pd.Timestamp.today())}",
            root=config.ROOT,
        )
        sync_bar.progress(1.0, text="Đồng bộ dữ liệu")
        if result.committed:
            sb.success(f"{result.message} Commit {result.commit_sha}.")
        else:
            sb.info(result.message)
    except Exception as exc:
        sync_bar.progress(1.0, text="Đồng bộ dữ liệu")
        sb.warning(f"Dữ liệu đã lưu cục bộ nhưng chưa đồng bộ lên GitHub: {exc}")

    line.empty()
    if report.success_count:
        sb.success(
            f"Cập nhật thành công {report.success_count}/{report.total_count} nguồn. "
            f"Dữ liệu đến ngày {fmt_date(storage.last_stored_date(storage.index_path(config.VNINDEX_DATASET)))}."
        )
    else:
        sb.error(f"Cập nhật thất bại {report.success_count}/{report.total_count} nguồn.")

    if report.failures:
        with sb.expander(f"{len(report.failures)} nguồn chưa cập nhật được", expanded=not report.success_count):
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
        "<div class='q'>Thị trường hiện đang ở trạng thái nào và điều gì đang tạo nên trạng thái đó?</div>"
        "<div class='hero-grid'>"
        f"<div class='hero-item'><div class='label'>Trạng thái hiện tại</div>"
        f"<div class='value {tone(regime['regime'])}'>{regime['regime']}</div></div>"
        f"<div class='hero-item'><div class='label'>Mức độ rủi ro</div>"
        f"<div class='value {tone(regime['risk_level'])}'>{regime['risk_level']}</div></div>"
        f"<div class='hero-item'><div class='label'>Dữ liệu đến ngày</div>"
        f"<div class='value'>{fmt_date(state['as_of'])}</div></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_regime(state: dict) -> None:
    regime = state["regime"]
    portfolio = state["portfolio"]
    notes = "".join(f"<li>{n}</li>" for n in regime["risk_reasons"])
    notes_html = f"<ul class='foot'>{notes}</ul>" if notes else ""
    st.markdown(
        "<div class='regime'>"
        "<div class='label' style='font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)'>Market Regime</div>"
        f"<div class='name {tone(regime['regime'])}'>{regime['regime']}</div>"
        f"<div class='body'>{regime['description']}</div>"
        f"<div style='margin-top:.9rem'><span class='pill {tone(regime['risk_level'])}'>"
        f"Mức độ rủi ro: {regime['risk_level']}</span></div>"
        f"{notes_html}"
        f"<div class='body' style='margin-top:.9rem'><strong>Hành động quản trị:</strong> {portfolio['risk_budget']}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_state_cards(state: dict) -> None:
    trend, stress, breadth = state["trend"], state["stress"], state["breadth"]
    dispersion, concentration = state["dispersion"], state["concentration"]

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        card(
            "Trạng thái xu hướng",
            trend["state"],
            f"RORO {fmt(trend.get('roro'), 2)} · vùng trung tính ±{fmt(trend.get('band'), 2)}. "
            "RORO là chênh lệch giữa sức mạnh động lượng đa khung và trung bình 49 phiên của chính nó.",
        ),
        unsafe_allow_html=True,
    )
    c2.markdown(
        card(
            "Trạng thái biến động",
            stress["state"],
            f"Parkinson 22 phiên {fmt(stress.get('parkinson_vol'), 1, '%')} · "
            f"phân vị {fmt(stress.get('percentile'), 0)}/100 trong 252 phiên gần nhất. "
            "Đây là chỉ số căng thẳng biến động dạng proxy, không phải VIX.",
        ),
        unsafe_allow_html=True,
    )
    valid = f"{breadth.get('valid_symbols', 0)}/{breadth['universe_size']} mã hợp lệ"
    c3.markdown(
        card(
            "Sức khỏe VN30",
            breadth["state"],
            f"Điểm breadth {fmt(breadth.get('score'), 1)}/100 · {valid}.",
        ),
        unsafe_allow_html=True,
    )

    c4, c5 = st.columns(2)
    c4.markdown(
        card(
            "Mức độ phân hóa",
            dispersion["state"],
            f"Độ lệch chuẩn lợi suất 20 phiên giữa các mã: {fmt(dispersion.get('value'), 2, '%')} · "
            f"phân vị {fmt(dispersion.get('percentile'), 0)}/100 trong {dispersion.get('context_sessions', 0)} phiên quan sát. "
            f"{dispersion.get('historical_basis', '')}",
        ),
        unsafe_allow_html=True,
    )
    top5 = concentration.get("top_shares", {}).get(5)
    c5.markdown(
        card(
            "Mức tập trung rủi ro",
            concentration["state"],
            f"Top 5 mã chiếm {fmt(top5, 1, '%')} tổng rủi ro biến động · "
            f"số mã đóng góp hiệu dụng {fmt(concentration.get('effective_names'), 1)}. "
            f"{concentration.get('proxy_note', '')}",
        ),
        unsafe_allow_html=True,
    )


def render_vn30(state: dict) -> None:
    breadth, concentration, dispersion = state["breadth"], state["concentration"], state["dispersion"]
    st.subheader("Nhóm VN30 hiện tại")

    if not breadth.get("sufficient"):
        st.warning(
            f"Chỉ có {breadth.get('max_valid_symbols', 0)}/{breadth['universe_size']} mã VN30 có dữ liệu. "
            "Bấm Cập nhật dữ liệu trong thanh bên để tải giá cổ phiếu thành phần."
        )
        if breadth["missing_symbols"]:
            st.caption("Mã chưa có dữ liệu: " + ", ".join(breadth["missing_symbols"]))
        return

    left, right = st.columns([1.05, 1])
    rows = ""
    for window in config.BREADTH_MA_WINDOWS:
        item = breadth["components"].get(f"ma{window}", {})
        rows += bar(item.get("pct"), f"Trên MA{window}", f"({item.get('valid', 0)}/{breadth['universe_size']} mã)")
    for window in config.BREADTH_RETURN_WINDOWS:
        item = breadth["components"].get(f"ret{window}", {})
        label = "Tăng trong phiên" if window == 1 else f"Tăng trong {window} phiên"
        rows += bar(item.get("pct"), label, f"({item.get('valid', 0)}/{breadth['universe_size']} mã)")
    left.markdown(
        f"<div class='card'><div class='label'>Độ lan tỏa</div>"
        f"<div class='value {tone(breadth['state'])}'>{breadth['state']} · {fmt(breadth['score'], 0)}/100</div>{rows}</div>",
        unsafe_allow_html=True,
    )

    top_shares = concentration.get("top_shares", {})
    detail = (
        f"<div class='rowline'><span class='k'>Phân hóa 1 phiên</span><span class='v'>"
        f"{fmt(dispersion['windows'].get(1, {}).get('value'), 2, '%')}</span></div>"
        f"<div class='rowline'><span class='k'>Phân hóa 5 phiên</span><span class='v'>"
        f"{fmt(dispersion['windows'].get(5, {}).get('value'), 2, '%')}</span></div>"
        f"<div class='rowline'><span class='k'>Phân hóa 20 phiên</span><span class='v'>"
        f"{fmt(dispersion.get('value'), 2, '%')}</span></div>"
        f"<div class='rowline'><span class='k'>Top 5 mã chiếm</span><span class='v'>{fmt(top_shares.get(5), 1, '%')} rủi ro biến động</span></div>"
        f"<div class='rowline'><span class='k'>Top 10 mã chiếm</span><span class='v'>{fmt(top_shares.get(10), 1, '%')} rủi ro biến động</span></div>"
        f"<div class='rowline'><span class='k'>Số mã đóng góp hiệu dụng</span><span class='v'>"
        f"{fmt(concentration.get('effective_names'), 1)}/{concentration.get('contributors', 0)}</span></div>"
        f"<div class='rowline'><span class='k'>Herfindahl</span><span class='v'>{fmt(concentration.get('hhi'), 4)}</span></div>"
    )
    right.markdown(
        f"<div class='card'><div class='label'>Phân hóa và tập trung rủi ro</div>"
        f"<div class='value {tone(dispersion['state'])}'>{dispersion['state']}</div>{detail}"
        f"<div class='foot'>Không hiển thị chuỗi lịch sử của rổ VN30 vì repository chưa có dữ liệu "
        f"thành phần VN30 theo từng thời điểm trong quá khứ.</div></div>",
        unsafe_allow_html=True,
    )

    with st.expander("Chi tiết từng mã VN30"):
        st.dataframe(breadth["table"], width="stretch", hide_index=True)
        st.dataframe(concentration["table"], width="stretch", hide_index=True)


def _line_chart(x, y, name: str, color: str, hline: float | None = None, percent: bool = False) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=2))
    )
    if hline is not None:
        figure.add_hline(y=hline, line_dash="dot", line_color="#98a2b3", line_width=1)
    figure.update_layout(
        height=260,
        margin=dict(l=8, r=8, t=28, b=8),
        title=dict(text=name, font=dict(size=13, color="#475467")),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=False, linecolor="#e6e9ef")
    figure.update_yaxes(gridcolor="#f0f2f5", zeroline=False, ticksuffix="%" if percent else "")
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
        _line_chart(chart["date"], chart["close"], "VNINDEX", config.COLOR_MUTED),
        width="stretch",
    )
    if "roro" in chart.columns:
        c2.plotly_chart(
            _line_chart(chart["date"], chart["roro"], "RORO (xu hướng tương đối)", config.COLOR_GOOD, hline=0.0),
            width="stretch",
        )
    else:
        c2.info("Chưa đủ dữ liệu để vẽ chuỗi RORO.")

    c3, c4 = st.columns(2)
    if "parkinson_vol" in chart.columns:
        c3.plotly_chart(
            _line_chart(
                chart["date"], chart["parkinson_vol"],
                "Biến động Parkinson 22 phiên", config.COLOR_WARN, percent=True,
            ),
            width="stretch",
        )
    else:
        c3.info("Chưa đủ dữ liệu để vẽ chuỗi biến động.")

    dispersion_series = state["dispersion"].get("series")
    if dispersion_series is not None and len(dispersion_series) > 5:
        c4.plotly_chart(
            _line_chart(
                dispersion_series.index,
                dispersion_series.values,
                "Phân hóa 20 phiên trong rổ VN30 hiện tại",
                config.COLOR_BAD,
                percent=True,
            ),
            width="stretch",
        )
    else:
        c4.info("Chưa đủ dữ liệu cổ phiếu VN30 để vẽ chuỗi phân hóa.")


def render_portfolio(state: dict) -> None:
    portfolio = state["portfolio"]
    st.subheader("Quản trị danh mục")
    items = [
        ("Mức rủi ro tham chiếu", portfolio["risk_budget"]),
        ("Mức độ thận trọng", portfolio["caution"]),
        ("Mức đòn bẩy tham chiếu", portfolio["leverage"]),
        ("Mức độ tập trung danh mục", portfolio["concentration"]),
        ("Khả năng duy trì tỷ trọng cổ phiếu", portfolio["equity_weight"]),
    ]
    columns = st.columns(len(items))
    for column, (label, text) in zip(columns, items):
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

    columns = st.columns(4)
    columns[0].metric("Ngày dữ liệu mới nhất", fmt_date(summary["index_last_date"]))
    columns[1].metric("Số phiên VNINDEX", f"{summary['index_sessions']:,}")
    columns[2].metric(
        "Cổ phiếu VN30 có dữ liệu",
        f"{summary['stock_symbols_available']}/{summary['stock_symbols_expected']}",
    )
    columns[3].metric(
        "Nguồn cập nhật gần nhất",
        f"{log.get('success_count', 0)}/{log.get('total_count', 0)}" if log else "—",
    )

    st.markdown(
        f"<div class='foot'>Khoảng dữ liệu VNINDEX: {fmt_date(summary['index_first_date'])} → "
        f"{fmt_date(summary['index_last_date'])}. "
        f"Danh sách VN30 chụp ngày {summary['universe_as_of'] or '—'} từ {summary['universe_source'] or '—'}"
        f"{' (danh sách dự phòng trong mã nguồn)' if summary['universe_is_fallback'] else ''}. "
        f"Thời điểm cập nhật gần nhất: {(log.get('finished_at') or '—')[:19].replace('T', ' ')}. "
        f"Thư viện: vnstock {log.get('vnstock_version') or vnstock_version()}.</div>",
        unsafe_allow_html=True,
    )

    failures = quality.failure_table(log)
    if len(failures):
        st.error(f"{len(failures)} nguồn chưa cập nhật được trong lần chạy gần nhất.")
        st.dataframe(failures, width="stretch", hide_index=True)

    with st.expander("Chi tiết từng tệp dữ liệu"):
        st.dataframe(quality.dataset_rows(meta["symbols"]), width="stretch", hide_index=True)
        if log.get("datasets"):
            st.caption("Số dòng trước và sau khi hợp nhất của lần cập nhật gần nhất")
            labels = {
                "name": "Mã", "source": "Nguồn", "requested_start": "Yêu cầu từ",
                "requested_end": "Yêu cầu đến", "rows_before_merge": "Dòng trước",
                "rows_after_merge": "Dòng sau", "rows_added": "Dòng thêm", "last_date": "Đến ngày",
            }
            # reindex thay vì chỉ mục cột trực tiếp, để nhật ký cũ thiếu trường
            # không làm hỏng cả trang.
            table = pd.DataFrame(log["datasets"]).reindex(columns=list(labels))
            st.dataframe(table.rename(columns=labels), width="stretch", hide_index=True)


def main() -> None:
    render_sidebar()
    state = load_state()

    if not state.get("ready"):
        st.markdown(
            "<div class='hero'><h1>TRẠNG THÁI THỊ TRƯỜNG VIỆT NAM</h1>"
            "<div class='q'>Thị trường hiện đang ở trạng thái nào và điều gì đang tạo nên trạng thái đó?</div></div>",
            unsafe_allow_html=True,
        )
        st.warning(state.get("reason", "Chưa có dữ liệu."))
        st.info("Mở thanh bên, bấm Kiểm tra kết nối API rồi bấm Cập nhật dữ liệu để tạo bộ dữ liệu đầu tiên.")
        return

    render_hero(state)
    render_regime(state)
    render_state_cards(state)
    render_vn30(state)
    render_charts(state)
    render_portfolio(state)
    render_quality(state)

    st.caption(
        "Ứng dụng mô tả trạng thái thị trường để hỗ trợ quản trị rủi ro danh mục. "
        "Không dự báo giá, không phải khuyến nghị mua bán."
    )


main()
