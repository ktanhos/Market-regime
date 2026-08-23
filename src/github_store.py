"""Đồng bộ dữ liệu lên GitHub bằng đúng MỘT commit cho mỗi lần cập nhật.

Kiến trúc cũ dùng ``contents`` API nên mỗi tệp là một commit riêng: 31 tệp là 31
commit, và mỗi commit lại kích hoạt Streamlit Cloud triển khai lại giữa chừng.

Ở đây dùng Git Data API::

    GET  /git/ref/heads/<branch>      -> commit hiện tại
    POST /git/blobs                   -> tải nội dung từng tệp
    POST /git/trees                   -> gộp thành một cây, base_tree là cây cũ
    POST /git/commits                 -> tạo một commit duy nhất
    PATCH /git/refs/heads/<branch>    -> đẩy nhánh về commit mới

Token KHÔNG bao giờ nằm trong mã nguồn. Nó chỉ đến từ ``st.secrets`` hoặc biến
môi trường. Thiếu token thì chỉ tính năng đồng bộ bị tắt, dashboard vẫn chạy.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import requests

API = "https://api.github.com"
TIMEOUT = 60
TOKEN_KEY = "GITHUB_TOKEN"


class GitHubSyncError(RuntimeError):
    pass


@dataclass
class SyncResult:
    committed: bool
    commit_sha: str = ""
    commit_url: str = ""
    files: int = 0
    message: str = ""


def resolve_token() -> str:
    """Lấy token từ Streamlit secrets, sau đó tới biến môi trường."""
    try:
        import streamlit as st

        value = st.secrets.get(TOKEN_KEY, "")
        if value:
            return str(value).strip()
    except Exception:
        # Không chạy trong Streamlit, hoặc chưa cấu hình secrets.
        pass
    return os.environ.get(TOKEN_KEY, "").strip()


def token_status() -> dict:
    token = resolve_token()
    return {
        "configured": bool(token),
        "hint": (
            "Đã cấu hình token đồng bộ GitHub."
            if token
            else f'Chưa cấu hình {TOKEN_KEY}. Thêm vào Streamlit Secrets dạng {TOKEN_KEY} = "..." để bật đồng bộ. '
            "Dashboard vẫn hoạt động bình thường khi chưa có token."
        ),
    }


def _headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _request(method: str, url: str, token: str, **kwargs):
    response = requests.request(method, url, headers=_headers(token), timeout=TIMEOUT, **kwargs)
    if response.status_code >= 400:
        detail = response.text[:400]
        if response.status_code in (401, 403):
            raise GitHubSyncError(
                f"GitHub từ chối token ({response.status_code}). Kiểm tra quyền ghi của token. {detail}"
            )
        raise GitHubSyncError(f"GitHub trả về {response.status_code}: {detail}")
    return response.json()


def sync_files(
    files: Sequence[Path],
    repo: str,
    branch: str,
    message: str,
    root: Path,
    token: str | None = None,
) -> SyncResult:
    """Đẩy toàn bộ danh sách tệp lên GitHub trong một commit."""
    token = token if token is not None else resolve_token()
    if not token:
        return SyncResult(False, message="Chưa cấu hình GITHUB_TOKEN nên bỏ qua bước đồng bộ.")

    files = [Path(f) for f in files if Path(f).exists()]
    if not files:
        return SyncResult(False, message="Không có tệp dữ liệu nào để đồng bộ.")

    ref = _request("GET", f"{API}/repos/{repo}/git/ref/heads/{branch}", token)
    head_sha = ref["object"]["sha"]
    head_commit = _request("GET", f"{API}/repos/{repo}/git/commits/{head_sha}", token)
    base_tree = head_commit["tree"]["sha"]

    tree_entries = []
    for path in files:
        blob = _request(
            "POST",
            f"{API}/repos/{repo}/git/blobs",
            token,
            json={
                "content": base64.b64encode(path.read_bytes()).decode("ascii"),
                "encoding": "base64",
            },
        )
        tree_entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "mode": "100644",
                "type": "blob",
                "sha": blob["sha"],
            }
        )

    tree = _request(
        "POST",
        f"{API}/repos/{repo}/git/trees",
        token,
        json={"base_tree": base_tree, "tree": tree_entries},
    )
    if tree["sha"] == base_tree:
        return SyncResult(False, files=len(files), message="Dữ liệu không thay đổi nên không tạo commit mới.")

    commit = _request(
        "POST",
        f"{API}/repos/{repo}/git/commits",
        token,
        json={"message": message, "tree": tree["sha"], "parents": [head_sha]},
    )
    _request(
        "PATCH",
        f"{API}/repos/{repo}/git/refs/heads/{branch}",
        token,
        json={"sha": commit["sha"], "force": False},
    )
    return SyncResult(
        committed=True,
        commit_sha=commit["sha"][:7],
        commit_url=commit.get("html_url", ""),
        files=len(files),
        message=f"Đã đồng bộ {len(files)} tệp trong một commit.",
    )
