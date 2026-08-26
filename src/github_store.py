"""Đồng bộ dữ liệu lên GitHub bằng đúng MỘT commit cho mỗi lần cập nhật.

Kiến trúc cũ dùng ``contents`` API nên mỗi tệp là một commit riêng: 31 tệp là 31
commit, và mỗi commit lại kích hoạt Streamlit Cloud triển khai lại giữa chừng.

Ở đây dùng Git Data API::

    GET  /git/ref/heads/<branch>              -> commit hiện tại
    GET  /git/trees/<base_tree>?recursive=1    -> sha của từng tệp đã có trên GitHub
    POST /git/blobs                            -> chỉ tải nội dung tệp THỰC SỰ đổi
    POST /git/trees                            -> gộp thành một cây, base_tree là cây cũ
    POST /git/commits                          -> tạo một commit duy nhất
    PATCH /git/refs/heads/<branch>             -> đẩy nhánh về commit mới

Bỏ qua bước tải blob cho tệp không đổi: Git dùng SHA-1 nội dung, nên có thể tự
tính "sha sẽ có" của một tệp cục bộ và so với sha đang có trên GitHub mà không
cần gọi mạng. Ở một lượt cập nhật tăng dần, phần lớn cổ phiếu VN30 vẫn ra một
phiên mới nên hầu hết tệp có đổi; nhưng khi một mã lấy dữ liệu thất bại, khi
người dùng bấm cập nhật hai lần liên tiếp, hay khi chạy ngoài phiên giao dịch,
việc này tránh được các lượt gọi ``POST /git/blobs`` không cần thiết — đây là
bước tốn thời gian nhất của khâu đồng bộ vì có một lượt gọi HTTP cho MỖI tệp.

Token KHÔNG bao giờ nằm trong mã nguồn. Nó chỉ đến từ ``st.secrets`` hoặc biến
môi trường. Thiếu token thì chỉ tính năng đồng bộ bị tắt, dashboard vẫn chạy.
"""

from __future__ import annotations

import base64
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import requests

from src.logging_config import get_logger

logger = get_logger(__name__)

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
    except (ImportError, FileNotFoundError, KeyError, AttributeError) as exc:
        # Không chạy trong Streamlit, hoặc chưa có tệp secrets. Cả hai đều hợp lệ:
        # đọc tiếp biến môi trường.
        logger.debug("Không đọc được st.secrets (%s), chuyển sang biến môi trường", type(exc).__name__)
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


def _git_blob_sha(content: bytes) -> str:
    """SHA-1 mà Git sẽ gán cho nội dung này, tính cục bộ không cần gọi mạng.

    Git băm blob theo dạng ``"blob {số byte}\\0{nội dung}"``. Tính được giá trị
    này ở máy cục bộ cho phép so sánh với sha đã có trên GitHub mà không cần
    tải nội dung lên trước, nên bỏ qua được lượt gọi ``POST /git/blobs`` cho
    những tệp không đổi.
    """
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def _existing_blob_shas(repo: str, base_tree: str, token: str) -> dict[str, str]:
    """sha hiện có trên GitHub của từng tệp, theo đường dẫn tương đối trong repo."""
    data = _request("GET", f"{API}/repos/{repo}/git/trees/{base_tree}?recursive=1", token)
    return {
        entry["path"]: entry["sha"]
        for entry in data.get("tree", [])
        if entry.get("type") == "blob"
    }


def sync_files(
    files: Sequence[Path],
    repo: str,
    branch: str,
    message: str,
    root: Path,
    token: str | None = None,
) -> SyncResult:
    """Đẩy các tệp THỰC SỰ thay đổi lên GitHub trong một commit.

    So sha nội dung cục bộ với sha đã có trên GitHub trước khi tải: tệp không
    đổi không tốn một lượt gọi ``POST /git/blobs`` nào, đây là bước tốn thời
    gian nhất của khâu đồng bộ vì có một lượt gọi HTTP cho mỗi tệp.
    """
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
    existing_shas = _existing_blob_shas(repo, base_tree, token)

    tree_entries = []
    for path in files:
        content = path.read_bytes()
        relpath = path.relative_to(root).as_posix()
        if existing_shas.get(relpath) == _git_blob_sha(content):
            continue  # nội dung giống hệt bản đã có trên GitHub, không cần tải lại
        blob = _request(
            "POST",
            f"{API}/repos/{repo}/git/blobs",
            token,
            json={"content": base64.b64encode(content).decode("ascii"), "encoding": "base64"},
        )
        tree_entries.append(
            {"path": relpath, "mode": "100644", "type": "blob", "sha": blob["sha"]}
        )

    if not tree_entries:
        return SyncResult(False, files=len(files), message="Dữ liệu không thay đổi nên không tạo commit mới.")

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
        files=len(tree_entries),
        message=f"Đã đồng bộ {len(tree_entries)}/{len(files)} tệp thay đổi trong một commit.",
    )
