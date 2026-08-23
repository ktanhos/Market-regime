"""Đồng bộ GitHub: một lần cập nhật là một commit, token không nằm trong mã nguồn."""

from __future__ import annotations

from pathlib import Path

import pytest

from src import github_store


def test_no_token_disables_sync_without_breaking_the_app(tmp_path):
    target = tmp_path / "a.parquet"
    target.write_bytes(b"x")
    result = github_store.sync_files([target], "o/r", "main", "msg", tmp_path, token="")
    assert result.committed is False
    assert "GITHUB_TOKEN" in result.message


def test_token_is_never_hardcoded():
    source = Path(github_store.__file__).read_text(encoding="utf-8")
    assert "ghp_" not in source and "github_pat_" not in source
    assert "st.secrets" in source


def test_token_status_explains_the_fallback(monkeypatch):
    monkeypatch.setattr(github_store, "resolve_token", lambda: "")
    status = github_store.token_status()
    assert status["configured"] is False
    assert "Dashboard vẫn hoạt động" in status["hint"]


def test_sync_creates_exactly_one_commit_for_many_files(tmp_path, monkeypatch):
    files = []
    for i in range(31):
        path = tmp_path / f"f{i}.parquet"
        path.write_bytes(f"payload-{i}".encode())
        files.append(path)

    calls = {"blobs": 0, "trees": 0, "commits": 0, "refs": 0}

    def fake_request(method, url, token, **kwargs):
        if url.endswith("/git/ref/heads/main"):
            return {"object": {"sha": "head"}}
        if "/git/commits/head" in url:
            return {"tree": {"sha": "base-tree"}}
        if url.endswith("/git/blobs"):
            calls["blobs"] += 1
            return {"sha": f"blob{calls['blobs']}"}
        if url.endswith("/git/trees"):
            calls["trees"] += 1
            assert kwargs["json"]["base_tree"] == "base-tree"
            assert len(kwargs["json"]["tree"]) == 31
            return {"sha": "new-tree"}
        if url.endswith("/git/commits"):
            calls["commits"] += 1
            assert kwargs["json"]["parents"] == ["head"]
            return {"sha": "abcdef1234", "html_url": "https://example/commit"}
        if "/git/refs/heads/main" in url:
            calls["refs"] += 1
            return {}
        raise AssertionError(f"lời gọi ngoài dự kiến: {url}")

    monkeypatch.setattr(github_store, "_request", fake_request)
    result = github_store.sync_files(files, "o/r", "main", "một commit", tmp_path, token="fake")

    assert result.committed is True
    assert calls == {"blobs": 31, "trees": 1, "commits": 1, "refs": 1}
    assert result.files == 31


def test_unchanged_data_does_not_create_an_empty_commit(tmp_path, monkeypatch):
    path = tmp_path / "a.parquet"
    path.write_bytes(b"x")

    def fake_request(method, url, token, **kwargs):
        if url.endswith("/git/ref/heads/main"):
            return {"object": {"sha": "head"}}
        if "/git/commits/head" in url:
            return {"tree": {"sha": "same-tree"}}
        if url.endswith("/git/blobs"):
            return {"sha": "blob"}
        if url.endswith("/git/trees"):
            return {"sha": "same-tree"}
        raise AssertionError("không được tạo commit khi dữ liệu không đổi")

    monkeypatch.setattr(github_store, "_request", fake_request)
    result = github_store.sync_files([path], "o/r", "main", "msg", tmp_path, token="fake")
    assert result.committed is False
    assert "không thay đổi" in result.message


def test_permission_error_is_explained(tmp_path, monkeypatch):
    path = tmp_path / "a.parquet"
    path.write_bytes(b"x")

    class Response:
        status_code = 403
        text = "Resource not accessible"

    monkeypatch.setattr(github_store.requests, "request", lambda *a, **k: Response())
    with pytest.raises(github_store.GitHubSyncError) as info:
        github_store.sync_files([path], "o/r", "main", "msg", tmp_path, token="fake")
    assert "token" in str(info.value).lower()
