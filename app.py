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
import time
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
from src import config, features, narrative, quality, storage
from src import dispersion as dispersion_module
from src import portfolio_risk, regime as regime_module
from src import universe as universe_module
from src.credentials import MIN_KEY_LENGTH, ApiCredentials, resolve_vnstock_api_key
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
  --ink:#0f172a; --muted:#64748b; --faint:#94a3b8; --line:#e7e9ee; --surface:#ffffff; --canvas:#f7f8fa;
  --good:#0f7b52; --bad:#c0392f; --warn:#b7791f; --none:#6b7280; --accent:#3454d1;
  --good-bg:#e8f4ee; --bad-bg:#fbeceb; --warn-bg:#fdf5e5; --none-bg:#f1f2f4; --accent-bg:#eaeefb;
  --shadow:0 1px 2px rgba(15,23,42,.04), 0 8px 24px -16px rgba(15,23,42,.12);
}
.stApp{background:var(--canvas);}
.block-container{max-width:1280px;padding-top:1.2rem;padding-bottom:3.5rem;}
h1,h2,h3{color:var(--ink);letter-spacing:-.015em;}
h3{font-weight:750;}

/* Hero: kết luận trong vài giây, không thuật ngữ. */
.hero{background:var(--surface);border:1px solid var(--line);border-radius:20px;
      padding:1.8rem 2rem;margin-bottom:.9rem;box-shadow:var(--shadow);}
.hero .kicker{font-size:.71rem;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);
              font-weight:700;margin-bottom:.35rem;}
.hero h1{margin:0 0 .5rem;font-size:1.85rem;line-height:1.25;}
.hero .q{color:var(--muted);font-size:1.02rem;margin-bottom:1.4rem;}
.hero-grid{display:flex;flex-wrap:wrap;gap:2.6rem;}
.hero-item .label{font-size:.71rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);}
.hero-item .value{font-size:1.4rem;font-weight:750;margin-top:.28rem;}

.card{background:var(--surface);border:1px solid var(--line);border-radius:16px;
      padding:1.15rem 1.25rem;height:100%;box-shadow:var(--shadow);}
.card .label{font-size:.71rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:650;}
.card .value{font-size:1.32rem;font-weight:750;margin:.32rem 0 .4rem;line-height:1.25;}
.card .note{color:var(--muted);font-size:.85rem;line-height:1.6;}
.card .sub{color:var(--ink);font-size:.92rem;line-height:1.6;margin-top:.4rem;}
.card .plain{color:var(--ink);font-size:.93rem;line-height:1.6;margin-top:.35rem;}

/* Khối trạng thái tổng: bên trong đã có "vì sao" bằng lời văn thường. */
.regime{background:var(--surface);border:1px solid var(--line);border-radius:20px;
        padding:1.6rem 1.9rem;margin-bottom:.9rem;box-shadow:var(--shadow);}
.regime .kicker{font-size:.71rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:650;}
.regime .name{font-size:1.8rem;font-weight:800;margin:.22rem 0 .6rem;}
.regime .body{color:var(--ink);font-size:1.02rem;line-height:1.75;max-width:76ch;}
.regime .summary{color:var(--muted);font-size:.95rem;line-height:1.7;max-width:76ch;margin-top:.5rem;}
.watchbox{background:var(--accent-bg);border-radius:12px;padding:.85rem 1rem;margin-top:1.1rem;
          font-size:.9rem;line-height:1.6;color:var(--ink);max-width:76ch;}
.watchbox b{color:var(--accent);}

.pill{display:inline-block;padding:.24rem .75rem;border-radius:999px;font-size:.77rem;font-weight:700;}
.pill.good{background:var(--good-bg);color:var(--good);}
.pill.bad{background:var(--bad-bg);color:var(--bad);}
.pill.warn{background:var(--warn-bg);color:var(--warn);}
.pill.none{background:var(--none-bg);color:var(--none);}
.good{color:var(--good);} .bad{color:var(--bad);} .warn{color:var(--warn);} .none{color:var(--none);}

.bar{height:7px;border-radius:999px;background:#edf0f4;overflow:hidden;margin:.5rem 0 .35rem;}
.bar > span{display:block;height:100%;border-radius:999px;}
.rowline{display:flex;justify-content:space-between;gap:1rem;font-size:.87rem;
         padding:.32rem 0;border-bottom:1px dashed var(--line);}
.rowline:last-child{border-bottom:none;}
.rowline .k{color:var(--muted);} .rowline .v{font-weight:650;color:var(--ink);text-align:right;}
.foot{color:var(--muted);font-size:.81rem;line-height:1.65;margin-top:.5rem;}

/* Progressive disclosure: expander của lớp diễn giải kỹ thuật không được nổi
   bật hơn nội dung phổ thông phía trên nó. */
div[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;background:var(--surface);}
div[data-testid="stExpander"] summary{font-size:.85rem;color:var(--muted);font-weight:600;}
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
    snapshot["story"] = narrative.build_narrative(snapshot)
    return snapshot


@st.cache_data(show_spinner=False)
def load_coverage(symbols: tuple[str, ...]) -> dict:
    return quality.coverage_summary(list(symbols))


@st.cache_data(show_spinner=False)
def load_dataset_rows(symbols: tuple[str, ...], log: dict) -> pd.DataFrame:
    """Bảng chi tiết từng tệp dữ liệu, chỉ tính lại khi log cập nhật đổi.

    Nội dung này nằm trong một ``st.expander``, nhưng Streamlit vẫn thực thi
    toàn bộ thân hàm render dù expander đang đóng — nếu không cache thì mỗi
    lần người dùng tương tác với bất kỳ ô nào trên trang (kể cả gõ vào ô khóa
    API) đều đọc lại 30+ tệp Parquet từ đĩa một cách không cần thiết.
    """
    return quality.dataset_rows(list(symbols), log)


# =============================================================================
# Khóa API
# =============================================================================

SESSION_KEY_FIELD = "vnstock_api_key"


def _secrets_api_key() -> str | None:
    """Đọc Streamlit Secrets. Chỉ tầng giao diện được phép chạm vào st.secrets."""
    try:
        value = st.secrets.get(config.VNSTOCK_API_KEY_ENV, "")
    except (FileNotFoundError, KeyError, AttributeError):
        return None
    return str(value).strip() or None


def current_credentials() -> ApiCredentials:
    """Khóa sẽ được dùng: phiên → Streamlit Secrets → biến môi trường."""
    return resolve_vnstock_api_key(
        session_key=st.session_state.get(SESSION_KEY_FIELD),
        secrets_key=_secrets_api_key(),
    )


def render_vnstock_connection(sb, credentials: ApiCredentials) -> None:
    """Khối KẾT NỐI VNSTOCK, hiển thị thẳng trong thanh bên.

    Ô nhập nằm trực tiếp ở thanh bên chứ không nằm trong expander: người dùng
    phải thấy ngay là có chỗ nhập khóa mà không cần bấm mở gì cả.

    Khóa nhập ở đây chỉ nằm trong ``st.session_state``. Nó không được ghi vào
    tệp, vào dữ liệu, vào nhật ký hay lên GitHub, và không bao giờ được hiển
    thị lại.
    """
    access = describe_api_access(credentials)

    if credentials.configured:
        status_text, status_tone = "API key đã được cấu hình", GOOD
        quota_text = "Tự động xác định theo gói API"
    else:
        status_text, status_tone = "API key chưa được cấu hình", WARN
        quota_text = f"Gói khách · {access.observed_limit} lượt/phút"

    sb.markdown(
        "<div class='label' style='margin-top:.35rem'>KẾT NỐI VNSTOCK</div>"
        + row("Trạng thái", status_text, status_tone)
        + row("Hạn mức API", quota_text)
        + row("Giới hạn vận hành", f"{access.effective_limit} lượt/phút")
        + (row("Nguồn khóa", credentials.source_label) if credentials.configured else ""),
        unsafe_allow_html=True,
    )

    entered = sb.text_input(
        "VNSTOCK API KEY",
        type="password",
        key="vnstock_api_key_input",
        placeholder="Dán API key tại đây",
        help=(
            "Khóa nhập tại đây chỉ tồn tại trong phiên làm việc này. Không được ghi "
            "vào tệp, dữ liệu hay GitHub. Nếu đã cấu hình Streamlit Secrets thì không "
            "cần nhập lại."
        ),
    )

    if sb.button("Áp dụng API key", width="stretch"):
        cleaned = (entered or "").strip()
        if not cleaned:
            sb.warning("Chưa nhập khóa.")
        elif len(cleaned) < MIN_KEY_LENGTH:
            sb.error(f"Khóa quá ngắn, cần tối thiểu {MIN_KEY_LENGTH} ký tự.")
        else:
            st.session_state[SESSION_KEY_FIELD] = cleaned
            st.rerun()

    if st.session_state.get(SESSION_KEY_FIELD):
        if sb.button("Xóa API key phiên này", width="stretch"):
            st.session_state.pop(SESSION_KEY_FIELD, None)
            st.rerun()
        sb.caption(f"Đang dùng khóa nhập trong phiên: {credentials.masked}")
    elif not credentials.configured:
        sb.caption("Không có khóa vẫn chạy được, chỉ chậm hơn vì hạn mức thấp hơn.")


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
    credentials = current_credentials()
    render_vnstock_connection(sb, credentials)

    sb.divider()
    token = token_status()
    if not token["configured"]:
        sb.info(token["hint"])

    if sb.button("Kiểm tra kết nối API", width="stretch"):
        render_api_check(sb, credentials)

    if sb.button("Cập nhật dữ liệu", type="primary", width="stretch"):
        run_update_flow(sb, credentials)

    sb.caption(
        "Mở dashboard không gọi API. Dữ liệu chỉ được lấy khi bấm nút cập nhật. "
        "Các mã được gọi tuần tự, có nghỉ giữa các lượt."
    )


def render_api_check(sb, credentials: ApiCredentials | None = None) -> None:
    """Chẩn đoán API: VNINDEX, FPT và danh sách VN30, không cần chạy 30 mã."""
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
    """Chạy pipeline cập nhật với thanh tiến trình theo từng giai đoạn."""
    credentials = credentials or current_credentials()
    bars = {phase: sb.progress(0.0, text=PHASE_LABELS[phase]) for phase in PHASE_ORDER}
    line = sb.empty()

    def progress(phase: str, ratio: float, message: str) -> None:
        line.caption(f"{PHASE_LABELS.get(phase, phase)} · {message}")
        if phase in bars:
            bars[phase].progress(min(1.0, max(0.0, float(ratio))), text=PHASE_LABELS[phase])

    try:
        report = run_update(progress=progress, credentials=credentials)
    except Exception as exc:  # dashboard không được phép sập vì lỗi cập nhật
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

    # --- Đồng bộ GitHub: chỉ đẩy tệp thực sự thay đổi ------------------------
    line.caption(f"{PHASE_LABELS[PHASE_SYNC]} · Đang đồng bộ")
    files = storage.data_files()
    sync_started = time.monotonic()
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
    report.phase_seconds[PHASE_SYNC] = round(time.monotonic() - sync_started, 2)
    storage.write_json(config.UPDATE_LOG_FILE, report.as_dict())
    bars[PHASE_SYNC].progress(1.0, text=PHASE_LABELS[PHASE_SYNC])
    line.empty()

    render_update_result(sb, report)

    # Chỉ xóa cache dữ liệu để dashboard đọc Parquet mới. Không đụng tới
    # session_state, nếu không người dùng mất khóa vừa nhập trong cùng phiên.
    load_market_state.clear()
    load_coverage.clear()
    if report.success_count:
        st.rerun()


def render_update_result(sb, report) -> None:
    """Kết quả cập nhật: tách riêng chỉ số và cổ phiếu, nêu rõ trạng thái đồng bộ."""
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
        with sb.expander(f"{report.failure_count} nguồn chưa cập nhật được",
                         expanded=not report.success_count):
            st.dataframe(quality.failure_table(report.as_dict()), width="stretch", hide_index=True)


# =============================================================================
# Trang chính
# =============================================================================

def render_hero(state: dict) -> None:
    """Lớp đầu tiên: hiểu kết luận trong vài giây, chưa có thuật ngữ kỹ thuật."""
    regime = state["regime"]
    story = state["story"]["regime"]
    st.markdown(
        "<div class='hero'>"
        "<div class='kicker'>Cập nhật trạng thái thị trường</div>"
        "<h1>TRẠNG THÁI THỊ TRƯỜNG VIỆT NAM</h1>"
        f"<div class='q'>{story['headline']}</div>"
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
    """Ba khối thông tin MỚI, không lặp lại kết luận đã nêu ở Hero:

    vì sao thị trường ở trạng thái này -> điều gì đã đổi so với lần cập nhật
    trước -> cần theo dõi gì để biết khi nào đánh giá lại. Tên chế độ, mô tả
    một câu và mức độ rủi ro đã xuất hiện ở Hero nên không lặp lại ở đây.
    """
    regime = state["regime"]
    portfolio = state["portfolio"]
    story = state["story"]["regime"]
    notes = "".join(f"<li>{note}</li>" for note in regime["risk_reasons"])
    st.markdown(
        "<div class='regime'>"
        "<div class='kicker'>Vì sao thị trường ở trạng thái này</div>"
        f"<div class='body'>{story['summary']}</div>"
        + (f"<ul class='foot'>{notes}</ul>" if notes else "")
        + "<div class='kicker' style='margin-top:1.2rem'>Điều gì đã thay đổi so với lần cập nhật trước</div>"
        f"<div class='body'>{story.get('change', 'Đây là lần đầu có đủ dữ liệu để mô tả, chưa có lần cập nhật trước để so sánh.')}</div>"
        + f"<div class='body' style='margin-top:1rem'><strong>Quản trị danh mục:</strong> "
        f"{portfolio['risk_budget']}</div>"
        f"<div class='watchbox'><b>Cần theo dõi gì:</b> {story['watch']}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def factor_card(column, title: str, verdict: object, plain: str, story: dict, technical: str) -> None:
    """Một chỉ tiêu, ba lớp: nhận định ngay → vì sao & cần theo dõi → số liệu kỹ thuật.

    Đây là Progressive Disclosure cho từng chỉ tiêu: người xem không có nền
    tảng thống kê chỉ cần đọc phần thẻ để hiểu kết luận; ai muốn hiểu sâu hơn
    mới cần mở phần mở rộng bên dưới.
    """
    column.markdown(
        f"<div class='card'><div class='label'>{title}</div>"
        f"<div class='value {tone(verdict)}'>{verdict}</div>"
        f"<div class='plain'>{plain}</div></div>",
        unsafe_allow_html=True,
    )
    with column.expander("Vì sao & cần theo dõi gì"):
        st.caption(story["why"])
        st.markdown(f"**Thay đổi:** {story.get('change', 'Chưa có lần cập nhật trước để so sánh.')}")
        st.markdown(f"**Cần theo dõi:** {story['watch']}")
        st.markdown(f"<div class='foot'>Số liệu kỹ thuật</div>{technical}", unsafe_allow_html=True)


def render_cards(state: dict) -> None:
    trend, stress, breadth = state["trend"], state["stress"], state["breadth"]
    dispersion, concentration = state["dispersion"], state["concentration"]
    story = state["story"]

    c1, c2, c3 = st.columns(3)
    factor_card(
        c1, "Xu hướng", trend["state"], story["trend"]["plain"], story["trend"],
        technical=row("RORO", fmt(trend.get("roro"), 2))
        + row("Vùng trung tính", "±" + fmt(trend.get("band"), 2))
        + row("Strength", fmt(trend.get("strength"), 2, "%")),
    )
    factor_card(
        c2, "Mức biến động", stress["state"], story["stress"]["plain"], story["stress"],
        technical=row("Parkinson 22 phiên", fmt(stress.get("parkinson_vol"), 1, "%"))
        + row("Phân vị 252 phiên", fmt(stress.get("percentile"), 0) + "/100")
        + row("Z-score", fmt(stress.get("zscore"), 2)),
    )
    components = breadth.get("components", {})
    total = breadth["universe_size"]
    factor_card(
        c3, "Sức khỏe VN30", breadth["state"], story["breadth"]["plain"], story["breadth"],
        technical=row("Breadth score", fmt(breadth.get("score"), 1) + "/100")
        + row("Mã hợp lệ", f"{breadth.get('valid_symbols', 0)}/{total} mã hợp lệ")
        + "".join(
            row(f"Trên MA{w}", fmt(components.get(f'ma{w}', {}).get('pct'), 0, "%"))
            for w in config.BREADTH_MA_WINDOWS
        ),
    )

    c4, c5 = st.columns(2)
    windows = dispersion.get("windows", {})
    factor_card(
        c4, "Phân hóa", dispersion["state"], story["dispersion"]["plain"], story["dispersion"],
        technical=row("Dispersion 20 phiên", fmt(dispersion.get("value"), 2, "%"))
        + row("Dispersion 5 phiên", fmt(windows.get(5, {}).get("value"), 2, "%"))
        + row("Dispersion 1 phiên", fmt(windows.get(1, {}).get("value"), 2, "%"))
        + row("Phân vị", fmt(dispersion.get("percentile"), 0) + "/100"),
    )
    top = concentration.get("top_shares", {})
    factor_card(
        c5, "Tập trung rủi ro", concentration["state"], story["concentration"]["plain"], story["concentration"],
        technical=row("Top 5 chiếm", fmt(top.get(5), 1, "%"))
        + row("Top 10 chiếm", fmt(top.get(10), 1, "%"))
        + row(
            "Số mã đóng góp hiệu dụng",
            f"{fmt(concentration.get('effective_names'), 1)}/{concentration.get('contributors', 0)}",
        )
        + row("Herfindahl", fmt(concentration.get("hhi"), 4)),
    )


def render_vn30(state: dict) -> None:
    breadth, concentration = state["breadth"], state["concentration"]
    st.subheader("Nhóm VN30 hiện tại")

    data_state = breadth.get("data_state", breadth_module.DATA_NONE)
    total = breadth["universe_size"]

    if data_state == breadth_module.DATA_NONE:
        st.info(
            "**Dữ liệu VN30 chưa được khởi tạo.**\n\n"
            "Ứng dụng đã có dữ liệu VNINDEX nhưng chưa có dữ liệu giá của các cổ phiếu VN30. "
            "Bấm **Cập nhật dữ liệu** trong thanh bên để tải dữ liệu lần đầu."
        )
        return

    if data_state == breadth_module.DATA_INSUFFICIENT:
        st.warning(
            f"Đang có {breadth.get('loaded_symbols', 0)}/{total} mã, trong đó "
            f"{breadth.get('min_valid_symbols', 0)} mã đủ lịch sử cho MA200. "
            "Bấm Cập nhật dữ liệu để tải nốt phần còn thiếu."
        )
        if breadth["missing_symbols"]:
            st.caption("Mã chưa có tệp: " + ", ".join(breadth["missing_symbols"]))
        return

    if data_state == breadth_module.DATA_STALE:
        st.warning(
            f"Ngày dữ liệu giữa các mã chênh nhau tới {breadth.get('max_gap_sessions', 0)} phiên. "
            "Số liệu dưới đây có thể không phản ánh cùng một thời điểm: "
            + ", ".join(breadth.get("stale_symbols", []))
        )

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
        c4.info("Chưa có đủ dữ liệu cổ phiếu VN30 để vẽ độ lan tỏa.")

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
    statuses = summary["statuses"]
    expected = summary["stock_symbols_expected"]
    complete = statuses[quality.STATUS_COMPLETE]

    if summary["never_updated"]:
        headline, headline_tone = "Chưa khởi tạo", NONE
    elif complete == expected and summary["files_written"] == summary["files_expected"]:
        headline, headline_tone = "Đầy đủ", GOOD
    else:
        headline, headline_tone = "Chưa đầy đủ", WARN

    sync_label = {
        SYNC_SUCCESS: f"thành công · {summary['sync_files']} tệp",
        SYNC_VIA_CI: f"workflow đã commit · {summary['sync_files']} tệp",
        SYNC_SKIPPED: "không có thay đổi",
        SYNC_FAILED: "thất bại",
    }.get(summary["sync_status"], "chưa chạy")
    sync_tone = {
        SYNC_SUCCESS: GOOD, SYNC_VIA_CI: GOOD, SYNC_SKIPPED: WARN, SYNC_FAILED: BAD,
    }.get(summary["sync_status"], NONE)

    detail = (
        row("Ngày dữ liệu", fmt_date(summary["index_last_date"]))
        + row(
            "VNINDEX",
            f"{indices['VNINDEX']['sessions']:,} phiên · đến {fmt_date(indices['VNINDEX']['last_date'])}",
            GOOD if indices["VNINDEX"]["sessions"] else BAD,
        )
        + row(
            "VN30 index",
            f"{indices['VN30']['sessions']:,} phiên · đến {fmt_date(indices['VN30']['last_date'])}",
            GOOD if indices["VN30"]["sessions"] else BAD,
        )
        + row("Cổ phiếu VN30 đầy đủ", f"{complete}/{expected}",
              GOOD if complete == expected else WARN)
        + row("Thiếu lịch sử", str(statuses[quality.STATUS_SHORT]),
              GOOD if not statuses[quality.STATUS_SHORT] else WARN)
        + row("Không có tệp", str(statuses[quality.STATUS_MISSING]),
              GOOD if not statuses[quality.STATUS_MISSING] else BAD)
        + row("Lỗi cập nhật", str(statuses[quality.STATUS_ERROR]),
              GOOD if not statuses[quality.STATUS_ERROR] else BAD)
        + row("Tệp dữ liệu", f"{summary['files_written']}/{summary['files_expected']}",
              GOOD if summary["files_written"] == summary["files_expected"] else WARN)
        + row("Nguồn dữ liệu",
              f"vnstock {log.get('vnstock_version') or vnstock_version()} · {config.PRIMARY_SOURCE}")
        + row("Lần cập nhật cuối", (log.get("finished_at") or "—")[:19].replace("T", " "))
        + row("Đồng bộ GitHub", sync_label, sync_tone)
    )
    st.markdown(
        f"<div class='card'><div class='label'>Tình trạng kho dữ liệu</div>"
        f"<div class='value {headline_tone}'>{headline}</div>{detail}"
        f"<div class='foot'>Danh sách VN30 chụp ngày {summary['universe_as_of'] or '—'} từ "
        f"{summary['universe_source'] or '—'}"
        f"{' — đang dùng bản dự phòng trong mã nguồn' if summary['universe_is_fallback'] else ''}."
        f"</div></div>",
        unsafe_allow_html=True,
    )

    if summary["legacy_files"]:
        st.warning(
            "Còn tệp ở bố cục cũ data/raw/: " + ", ".join(summary["legacy_files"])
            + ". Giá cổ phiếu chỉ được lưu ở data/raw/stocks/."
        )

    if summary["never_updated"]:
        st.info("Chưa có lần cập nhật nào. Bấm Cập nhật dữ liệu trong thanh bên để khởi tạo.")
    elif statuses[quality.STATUS_MISSING]:
        st.warning(
            f"{statuses[quality.STATUS_MISSING]} mã chưa có tệp giá: "
            + ", ".join(summary["stock_symbols_missing"])
        )

    failures = quality.failure_table(log)
    if len(failures):
        st.error(f"{len(failures)} nguồn chưa cập nhật được trong lần chạy gần nhất.")
        st.dataframe(failures, width="stretch", hide_index=True)

    probes = st.session_state.get("api_probes")
    if probes:
        with st.expander("Kết quả kiểm tra API gần nhất"):
            st.dataframe(pd.DataFrame(probes), width="stretch", hide_index=True)

    phase_seconds = log.get("phase_seconds") or {}
    if phase_seconds:
        with st.expander("Tốc độ lần cập nhật gần nhất"):
            st.caption(
                "Đo trực tiếp từng bước của lần cập nhật gần nhất, dùng để xác định "
                "bước nào thực sự chiếm thời gian trước khi tối ưu bước đó."
            )
            total_seconds = sum(v for v in phase_seconds.values() if isinstance(v, (int, float)))
            table = pd.DataFrame(
                [
                    {
                        "Giai đoạn": PHASE_LABELS.get(name, name),
                        "Thời gian (giây)": round(seconds, 1),
                        "Tỷ trọng": f"{seconds / total_seconds * 100:.0f}%" if total_seconds else "—",
                    }
                    for name, seconds in phase_seconds.items()
                ]
            )
            st.dataframe(table, width="stretch", hide_index=True)
            st.caption(
                "Bước VN30 Stocks bị giới hạn bởi hạn mức lượt gọi/phút của nguồn dữ "
                "liệu, không phải bởi tốc độ xử lý cục bộ — tăng song song không giúp "
                "vượt qua giới hạn này. Bước GitHub chỉ tải lên những tệp thực sự đổi "
                "nội dung so với lần commit trước."
            )

    # Bảng chi tiết chỉ có ý nghĩa khi đã có dữ liệu; ở lần đầu nó chỉ là 30 dòng lỗi.
    if not summary["never_updated"] or summary["stock_symbols_available"]:
        with st.expander("Chi tiết từng tệp dữ liệu"):
            st.dataframe(load_dataset_rows(tuple(meta["symbols"]), log), width="stretch", hide_index=True)
            if log.get("datasets"):
                st.caption("Số dòng trước và sau khi hợp nhất của lần cập nhật gần nhất")
                labels = {
                    "name": "Mã", "source": "Nguồn", "mode": "Chế độ",
                    "requested_start": "Yêu cầu từ", "requested_end": "Yêu cầu đến",
                    "rows_before_merge": "Dòng trước", "rows_after_merge": "Dòng sau",
                    "rows_added": "Dòng thêm", "last_date": "Đến ngày",
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
