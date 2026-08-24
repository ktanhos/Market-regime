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

from src import breadth as breadth_module
from src import concentration as concentration_module
from src import config, features, quality, storage
from src import dispersion as dispersion_module
from src import portfolio_risk, regime as regime_module
from src import universe as universe_module
from src.credentials import ApiCredentials, resolve_vnstock_api_key
from src.github_store import sync_files, token_status
from src.logging_config import get_logger
from src.updater import (
    PHASE_ORDER,
    PHASE_LABELS,
    PHASE_SYNC,
    SYNC_FAILED,
    SYNC_SKIPPED,
    SYNC_SUCCESS,
    SYNC_VIA_CI,
    record_sync,
    run_update,
)
from src.vnstock_data import (
    api_access,
    connectivity_check,
    describe_api_access,
    expected_schema,
    make_limiter,
    vnstock_version,
)

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
.hero{background:var(--surface);border:1px solid var(--line);border-radius:18px;
      padding:1.7rem 1.9rem;margin-bottom:1rem;}
.hero h1{margin:0 0 .4rem;font-size:1.8rem;line-height:1.25;}
.hero .q{color:var(--muted);font-size:1.02rem;margin-bottom:1.3rem;}
.hero-grid{display:flex;flex-wrap:wrap;gap:2.6rem;}
.hero-item .label{font-size:.71rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);}
.hero-item .value{font-size:1.4rem;font-weight:750;margin-top:.28rem;}
.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;
      padding:1.1rem 1.2rem;height:100%;}
.card .label{font-size:.71rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);}
.card .value{font-size:1.3rem;font-weight:750;margin:.3rem 0 .4rem;line-height:1.25;}
.card .note{color:var(--muted);font-size:.85rem;line-height:1.6;}
.card .sub{color:var(--ink);font-size:.89rem;line-height:1.55;margin-top:.45rem;}
.regime{background:var(--surface);border:1px solid var(--line);border-radius:18px;
        padding:1.5rem 1.8rem;margin-bottom:1rem;}
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
.rowline{display:flex;justify-content:space-between;gap:1rem;font-size:.87rem;
         padding:.3rem 0;border-bottom:1px dashed var(--line);}
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
    breadth_module.BREADTH_NO_DATA: NONE,
    breadth_module.BREADTH_INSUFFICIENT: NONE,
    breadth_module.BREADTH_STALE: WARN,
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


def tone(state: object) -> str:
    """Màu theo ý nghĩa, không bao giờ theo trang trí."""
    return _TONE.get(str(state).strip(), NONE)


def fmt(value: object, digits: int = 1, suffix: str = "") -> str:
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


@st.cache_data(show_spinner=False)
def load_market_state() -> dict:
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


SESSION_KEY_FIELD = "vnstock_api_key"


def _secrets_api_key() -> str | None:
    try:
        value = st.secrets.get(config.VNSTOCK_API_KEY_ENV, "")
    except (FileNotFoundError, KeyError, AttributeError):
        return None
    return str(value).strip() or None


def current_credentials() -> ApiCredentials:
    return resolve_vnstock_api_key(
        session_key=st.session_state.get(SESSION_KEY_FIELD),
        secrets_key=_secrets_api_key(),
    )


def render_api_key_controls(sb, credentials: ApiCredentials) -> None:
    label = (
        f"Thay đổi khóa API ({credentials.source_label})"
        if credentials.configured
        else "Nhập khóa API"
    )
    with sb.expander(label, expanded=False):
        entered = st.text_input(
            "VNSTOCK API KEY",
            type="password",
            key="vnstock_api_key_input",
            placeholder="Dán API key tại đây",
            help=(
                "Khóa nhập tại đây chỉ tồn tại trong phiên làm việc này. "
                "Không được ghi vào tệp, dữ liệu hay GitHub. Nếu đã cấu hình "
                "Streamlit Secrets thì không cần nhập lại."
            ),
        )
        columns = st.columns(2)
        if columns[0].button("Áp dụng", type="primary", width="stretch"):
            cleaned = (entered or "").strip()
            if not cleaned:
                st.warning("Chưa nhập khóa.")
            elif len(cleaned) < 10:
                st.error("Khóa quá ngắn, vnstock có thể từ chối.")
            else:
                st.session_state[SESSION_KEY_FIELD] = cleaned
                st.session_state.pop("vnstock_api_key_input", None)
                st.rerun()
        if columns[1].button("Xóa khóa phiên", width="stretch"):
            st.session_state.pop(SESSION_KEY_FIELD, None)
            st.session_state.pop("vnstock_api_key_input", None)
            st.rerun()

        active = current_credentials()
        if active.configured:
            st.success(f"API key đã được cấu hình · {active.source_label}")
        else:
            st.caption("Chưa có khóa. Ứng dụng vẫn chạy với hạn mức gói khách.")


def render_api_access(sb, credentials: ApiCredentials) -> None:
    access = describe_api_access(credentials)
    if credentials.configured:
        status_text, status_tone = "API key đã được cấu hình", GOOD
        quota = f"{access.observed_limit} lượt/phút"
    else:
        status_text, status_tone = "Chưa cấu hình API key", WARN
        quota = f"Gói khách · {access.observed_limit} lượt/phút"

    sb.markdown(
        "<div class='label' style='margin-top:.5rem'>KẾT NỐI DỮ LIỆU</div>"
        + row("Nguồn dữ liệu", f"VNStock {vnstock_version()}")
        + row("API access", status_text, status_tone)
        + row("Hạn mức API", quota)
        + row("Giới hạn vận hành", f"{access.effective_limit} lượt/phút")
        + (row("Nguồn khóa", credentials.source_label) if credentials.configured else ""),
        unsafe_allow_html=True,
    )
    render_api_key_controls(sb, credentials)


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
    credentials = current_credentials()
    render_api_access(sb, credentials)

    sb.divider()
    token = token_status()
    if not token["configured"]:
        sb.info(token["hint"])

    if sb.button("Kiểm tra API", width="stretch"):
        render_api_check(sb, credentials)

    if sb.button("Cập nhật dữ liệu", type="primary", width="stretch"):
        run_update_flow(sb, credentials)

    sb.caption(
        "Mở dashboard không gọi API. Dữ liệu chỉ được lấy khi bấm nút cập nhật. "
        "Các mã được gọi tuần tự, có giới hạn tốc độ tự động."
    )


def render_api_check(sb, credentials: ApiCredentials | None = None) -> None:
    credentials = credentials or current_credentials()
    with sb.status("Đang kiểm tra API...", expanded=True) as status:
        with api_access(credentials) as access:
            st.caption(
                f"Gói {access.tier} · hạn mức {access.observed_limit}, "
                f"vận hành {access.effective_limit} lượt/phút"
            )
            probes = connectivity_check(limiter=make_limiter(access=access))
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


def run_update_flow(sb, credentials: ApiCredentials | None = None) -> None:
    credentials = credentials or current_credentials()
    bars = {phase: sb.progress(0.0, text=PHASE_LABELS[phase]) for phase in PHASE_ORDER}
    line = sb.empty()

    def progress(phase: str, ratio: float, message: str) -> None:
        line.caption(f"{PHASE_LABELS.get(phase, phase)} · {message}")
        if phase in bars:
            bars[phase].progress(min(1.0, max(0.0, float(ratio))), text=PHASE_LABELS[phase])

    try:
        report = run_update(progress=progress, credentials=credentials)
    except Exception as exc:
        logger.exception("Cập nhật thất bại")
        line.empty()
        sb.error(f"Không hoàn tất cập nhật: {type(exc).__name__}: {exc}")
        return

    access = report.api_access or {}
    if access.get("effective_limit"):
        line.caption(
            f"Gói {access.get('tier', '-')} · vận hành {access['effective_limit']} lượt/phút"
        )
    if report.first_run:
        sb.info("Đây là lần khởi tạo dữ liệu VN30 đầu tiên.")
    if report.rate_limited:
        sb.warning(report.aborted_reason)

    line.caption(f"{PHASE_LABELS[PHASE_SYNC]} · Đang đồng bộ")
    files = storage.data_files()
    try:
        result = sync_files(
            files,
            repo=config.GITHUB_REPO,
            branch=config.GITHUB_BRANCH,
            message=f"Cập nhật dữ liệu thị trường đến {fmt_date(pd.Timestamp.today())}",
            root=config.ROOT,
        )
        if result.committed:
            record_sync(report, SYNC_SUCCESS, result.files, result.message)
        else:
            record_sync(report, SYNC_SKIPPED, result.files, result.message)
    except Exception as exc:
        logger.exception("Đồng bộ GitHub thất bại")
        record_sync(report, SYNC_FAILED, 0, f"{type(exc).__name__}: {exc}")
    bars[PHASE_SYNC].progress(1.0, text=PHASE_LABELS[PHASE_SYNC])
    line.empty()

    render_update_result(sb, report)

    load_market_state.clear()
    load_coverage.clear()
    if report.success_count:
        st.rerun()


def render_update_result(sb, report) -> None:
    latest = storage.last_stored_date(storage.index_path(config.VNINDEX_DATASET))

    if report.completed:
        sb.success("Cập nhật thành công")
    elif report.data_complete:
        sb.warning("Dữ liệu đã lấy thành công nhưng chưa đồng bộ GitHub")
    elif report.success_count:
        sb.warning("Cập nhật chưa đầy đủ")
    else:
        sb.error("Cập nhật thất bại")

    index_ok = report.index_success == report.index_total and report.index_total > 0
    stock_ok = report.stock_success == report.stock_total and report.stock_total > 0
    sync_tone = {
        SYNC_SUCCESS: GOOD, SYNC_VIA_CI: GOOD, SYNC_SKIPPED: WARN, SYNC_FAILED: BAD,
    }.get(report.sync_status, NONE)
    sync_text = {
        SYNC_SUCCESS: f"{report.sync_files} tệp trong một commit",
        SYNC_VIA_CI: f"{report.sync_files} tệp, do workflow commit",
        SYNC_SKIPPED: "không có thay đổi để đồng bộ",
        SYNC_FAILED: "thất bại",
    }.get(report.sync_status, "chưa chạy")

    sb.markdown(
        row("Chỉ số", f"{report.index_success}/{report.index_total} thành công",
            GOOD if index_ok else BAD)
        + row("Cổ phiếu VN30", f"{report.stock_success}/{report.stock_total} mã thành công",
              GOOD if stock_ok else (WARN if report.stock_success else BAD))
        + row("Tệp dữ liệu", f"{report.files_written}/{report.files_expected}",
              GOOD if report.files_written == report.files_expected else WARN)
        + row("Đồng bộ GitHub", sync_text, sync_tone)
        + row("Dữ liệu đến ngày", fmt_date(latest)),
        unsafe_allow_html=True,
    )

    if report.sync_status == SYNC_FAILED:
        sb.error(f"GitHub: {report.sync_message}")
    if report.stock_missing:
        sb.warning(f"{len(report.stock_missing)} mã chưa có tệp: " + ", ".join(report.stock_missing))
    if report.failures:
        with sb.expander(f"{report.failure_count} nguồn chưa cập nhật được", expanded=not report.success_count):
            st.dataframe(quality.failure_table(report.as_dict()), width="stretch", hide_index=True)


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
