"""Xác định API key, áp dụng vào client, và bảo đảm khóa không rò ra ngoài."""

from __future__ import annotations

import json
import os

import pytest

from src import config, storage
from src.credentials import (
    SOURCE_ENV,
    SOURCE_NONE,
    SOURCE_SECRETS,
    SOURCE_SESSION,
    ApiCredentials,
    contains_secret,
    mask_secret,
    resolve_vnstock_api_key,
)
from src.updater import record_sync, run_update
from src.vnstock_data import RateLimiter, api_access, describe_api_access, make_limiter

SESSION = "SESSION-KEY-0123456789"
SECRETS = "SECRETS-KEY-0123456789"
ENV = "ENV-KEY-0123456789"


# --- Thứ tự ưu tiên -----------------------------------------------------------

def test_session_key_wins_over_everything():
    creds = resolve_vnstock_api_key(SESSION, SECRETS, {config.VNSTOCK_API_KEY_ENV: ENV})
    assert creds.key == SESSION
    assert creds.source == SOURCE_SESSION


def test_secrets_win_over_environment():
    creds = resolve_vnstock_api_key(None, SECRETS, {config.VNSTOCK_API_KEY_ENV: ENV})
    assert creds.key == SECRETS
    assert creds.source == SOURCE_SECRETS


def test_environment_is_the_last_resort():
    creds = resolve_vnstock_api_key(None, None, {config.VNSTOCK_API_KEY_ENV: ENV})
    assert creds.key == ENV
    assert creds.source == SOURCE_ENV


def test_no_key_anywhere_is_a_valid_state():
    creds = resolve_vnstock_api_key(None, None, {})
    assert creds.key is None
    assert creds.source == SOURCE_NONE
    assert creds.configured is False


def test_blank_and_whitespace_values_are_ignored():
    creds = resolve_vnstock_api_key("   ", "", {config.VNSTOCK_API_KEY_ENV: ENV})
    assert creds.source == SOURCE_ENV
    assert resolve_vnstock_api_key("  " + SESSION + " ", None, {}).key == SESSION


# --- Không rò khóa ------------------------------------------------------------

def test_repr_never_exposes_the_key():
    creds = ApiCredentials(key=SESSION, source=SOURCE_SESSION)
    assert SESSION not in repr(creds)
    assert SESSION not in str(creds)
    assert SESSION not in f"{creds}"


def test_mask_keeps_at_most_four_characters():
    masked = mask_secret("ABCDEFGHIJKLMNOP")
    assert masked.endswith("MNOP")
    assert "ABCDEFGHIJKL" not in masked
    assert mask_secret(None) == "—"


def test_api_access_metadata_carries_no_key():
    with api_access(ApiCredentials(key=SESSION, source=SOURCE_SESSION)) as access:
        payload = access.as_dict()
    assert SESSION not in json.dumps(payload)
    assert set(payload) == {
        "tier", "observed_limit", "effective_limit", "configured", "verified", "source", "note",
    }


# --- Áp dụng vào client -------------------------------------------------------

@pytest.fixture()
def clean_env(monkeypatch):
    monkeypatch.delenv(config.VNSTOCK_API_KEY_ENV, raising=False)
    return monkeypatch


def test_key_is_actually_handed_to_the_client(clean_env):
    """Không chỉ phát hiện có chuỗi khóa, mà client phải thật sự cầm đúng khóa đó."""
    from vnai.beam.auth import authenticator

    with api_access(ApiCredentials(key=SESSION, source=SOURCE_SESSION)) as access:
        assert access.verified is True
        assert authenticator.get_api_key() == SESSION
        assert os.environ[config.VNSTOCK_API_KEY_ENV] == SESSION


def test_environment_is_restored_so_a_session_key_cannot_leak(clean_env):
    """Streamlit Cloud dùng chung tiến trình; khóa một phiên không được sang phiên khác."""
    with api_access(ApiCredentials(key=SESSION, source=SOURCE_SESSION)):
        pass
    assert config.VNSTOCK_API_KEY_ENV not in os.environ

    clean_env.setenv(config.VNSTOCK_API_KEY_ENV, ENV)
    with api_access(ApiCredentials(key=SESSION, source=SOURCE_SESSION)):
        assert os.environ[config.VNSTOCK_API_KEY_ENV] == SESSION
    assert os.environ[config.VNSTOCK_API_KEY_ENV] == ENV


def test_a_key_too_short_is_refused_and_falls_back_to_guest(clean_env):
    with api_access(ApiCredentials(key="short", source=SOURCE_SESSION)) as access:
        assert access.configured is True
        assert access.verified is False
        assert access.effective_limit == config.requests_per_minute(False)
        assert "quá ngắn" in access.note


def test_an_unverified_key_never_raises_the_limit(clean_env, monkeypatch):
    """Client không nhận khóa thì tuyệt đối không được giả định hạn mức cao."""
    import src.vnstock_data as module

    class Refusing:
        def get_api_key(self):
            return None

        def get_tier(self, force_refresh=False):
            return "guest"

        def get_limits(self, tier=None):
            return {"min": 20}

    monkeypatch.setattr(module, "_authenticator", lambda: Refusing())
    with api_access(ApiCredentials(key=SESSION, source=SOURCE_SESSION)) as access:
        assert access.verified is False
        assert access.observed_limit == config.RATE_LIMIT_FALLBACK_GUEST
        assert make_limiter(access=access).max_calls == config.requests_per_minute(False)


# --- Hạn mức ------------------------------------------------------------------

def test_no_key_runs_at_the_guest_budget(clean_env):
    with api_access() as access:
        assert access.tier == "guest"
        assert access.observed_limit == 20
        assert access.effective_limit == config.effective_rate_limit(20)
        assert make_limiter(access=access).max_calls == access.effective_limit


def test_a_verified_key_runs_at_the_community_budget(clean_env):
    with api_access(ApiCredentials(key=ENV, source=SOURCE_ENV)) as access:
        assert access.tier == "free"
        assert access.observed_limit == 60
        assert access.effective_limit == config.effective_rate_limit(60)
        assert make_limiter(access=access).max_calls == access.effective_limit


def test_the_budget_comes_from_the_clients_own_tier_table(clean_env, monkeypatch):
    """Gói tài trợ phải tự động được nhận, không cần sửa config."""
    import src.vnstock_data as module

    class Sponsor:
        def get_api_key(self):
            return ENV

        def get_tier(self, force_refresh=False):
            return "golden"

        def get_limits(self, tier=None):
            return {"min": 500}

    monkeypatch.setattr(module, "_authenticator", lambda: Sponsor())
    with api_access(ApiCredentials(key=ENV, source=SOURCE_ENV)) as access:
        assert access.tier == "golden"
        assert access.observed_limit == 500
        assert access.effective_limit == config.effective_rate_limit(500)


def test_safety_margin_lives_in_exactly_one_place():
    assert config.effective_rate_limit(20) < 20
    assert config.effective_rate_limit(60) < 60
    ratio_20 = config.effective_rate_limit(20) / 20
    ratio_60 = config.effective_rate_limit(60) / 60
    assert ratio_20 == pytest.approx(ratio_60, abs=0.05)


def test_limiter_downgrades_when_the_source_still_refuses():
    """Bị chặn dù đã điều tiết nghĩa là ước lượng gói đã sai."""
    limiter = RateLimiter(max_calls=45, sleep=lambda s: None, clock=lambda: 0.0)
    assert limiter.downgrade() == config.requests_per_minute(False)
    # Không bao giờ tự nâng lên lại.
    assert limiter.downgrade(999) == config.requests_per_minute(False)


def test_describe_api_access_changes_nothing(clean_env):
    describe_api_access(ApiCredentials(key=SESSION, source=SOURCE_SESSION))
    assert config.VNSTOCK_API_KEY_ENV not in os.environ


# --- Không lọt vào dữ liệu ----------------------------------------------------

def _fetcher():
    import numpy as np
    import pandas as pd

    from src.schema import standardize_ohlcv
    from src.vnstock_data import FetchResult

    def fetcher(symbol, start, end=None, asset_type="stock", sleep=None, **kwargs):
        rng = np.random.default_rng(abs(hash(symbol)) % 997)
        index = pd.bdate_range("2025-06-19", periods=260)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, 260)))
        frame = pd.DataFrame(
            {"time": index, "open": close, "high": close * 1.01,
             "low": close * 0.99, "close": close, "volume": 1}
        )
        return FetchResult(symbol=symbol, frame=standardize_ohlcv(frame), source="VCI", attempts=1)

    return fetcher


def test_the_key_never_reaches_the_update_log(temp_store, clean_env):
    symbols = [f"S{i:02d}" for i in range(25)]
    creds = ApiCredentials(key=SESSION, source=SOURCE_SESSION)

    report = run_update(
        fetcher=_fetcher(), universe_fetcher=lambda: list(symbols),
        sleep=lambda s: None, credentials=creds,
    )
    record_sync(report, "success", report.files_written, "ok")

    written = config.UPDATE_LOG_FILE.read_text(encoding="utf-8")
    assert SESSION not in written
    assert contains_secret(json.loads(written), creds) is False
    # Nhưng siêu dữ liệu về gói truy cập thì vẫn phải có.
    assert report.api_access["source"] == SOURCE_SESSION
    assert report.api_access["configured"] is True


def test_the_key_never_reaches_any_data_file(temp_store, clean_env):
    symbols = [f"S{i:02d}" for i in range(25)]
    creds = ApiCredentials(key=SESSION, source=SOURCE_SESSION)
    run_update(
        fetcher=_fetcher(), universe_fetcher=lambda: list(symbols),
        sleep=lambda s: None, credentials=creds,
    )

    for path in storage.data_files():
        assert SESSION.encode() not in path.read_bytes(), path


def test_the_key_never_reaches_a_fetch_error_message(clean_env):
    from src.vnstock_data import TRANSIENT, FetchError, fetch_history

    def caller(symbol, source, start, end):
        raise ConnectionError("connection reset")

    with api_access(ApiCredentials(key=SESSION, source=SOURCE_SESSION)):
        with pytest.raises(FetchError) as info:
            fetch_history("FPT", start="2024-01-01", caller=caller, sleep=lambda s: None)
    assert info.value.kind == TRANSIENT
    assert SESSION not in str(info.value)
    assert SESSION not in json.dumps(info.value.as_dict())


def test_the_pipeline_runs_without_any_key(temp_store, clean_env):
    symbols = [f"S{i:02d}" for i in range(25)]
    report = run_update(
        fetcher=_fetcher(), universe_fetcher=lambda: list(symbols), sleep=lambda s: None,
    )
    assert report.stock_success == 25
    assert report.api_access["configured"] is False
    assert report.api_access["effective_limit"] == config.requests_per_minute(False)


def test_exactly_one_limiter_is_used_for_a_whole_run(temp_store, clean_env, monkeypatch):
    import src.vnstock_data as module

    created: list = []
    original = module.RateLimiter

    class Counting(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(module, "RateLimiter", Counting)
    run_update(
        fetcher=_fetcher(), universe_fetcher=lambda: [f"S{i:02d}" for i in range(25)],
        sleep=lambda s: None,
    )
    assert len(created) == 1, f"Tạo {len(created)} bộ điều tiết, phải đúng 1"


def test_the_key_cannot_reach_github_sync(temp_store, clean_env, monkeypatch):
    """Mọi thứ đi lên GitHub đều phải sạch khóa: nội dung tệp và thông điệp commit."""
    from src import github_store

    symbols = [f"S{i:02d}" for i in range(25)]
    creds = ApiCredentials(key=SESSION, source=SOURCE_SESSION)
    run_update(
        fetcher=_fetcher(), universe_fetcher=lambda: list(symbols),
        sleep=lambda s: None, credentials=creds,
    )

    sent: list[str] = []

    def fake_request(method, url, token, **kwargs):
        sent.append(json.dumps(kwargs.get("json", {})))
        if url.endswith("/git/ref/heads/main"):
            return {"object": {"sha": "head"}}
        if "/git/commits/head" in url:
            return {"tree": {"sha": "base"}}
        if url.endswith("/git/blobs"):
            return {"sha": "blob"}
        if url.endswith("/git/trees"):
            return {"sha": "tree"}
        if url.endswith("/git/commits"):
            return {"sha": "abc1234", "html_url": "https://example/c"}
        return {}

    monkeypatch.setattr(github_store, "_request", fake_request)
    result = github_store.sync_files(
        storage.data_files(), "o/r", "main",
        f"Cập nhật dữ liệu ({config.VNSTOCK_API_KEY_ENV} không được nhắc tới)",
        config.ROOT, token="fake",
    )

    assert result.committed is True
    payload = " ".join(sent)
    assert SESSION not in payload
    # Blob được mã hóa base64 nên kiểm tra cả dạng đã mã hóa.
    import base64

    assert base64.b64encode(SESSION.encode()).decode() not in payload
