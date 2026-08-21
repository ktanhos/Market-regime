from __future__ import annotations

import base64
from pathlib import Path
import requests

API = "https://api.github.com"


def upload_file(token: str, repo: str, branch: str, local_path: Path, repo_path: str, message: str) -> None:
    if not token:
        raise ValueError("Thiếu GitHub token")
    url = f"{API}/repos/{repo}/contents/{repo_path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    params = {"ref": branch}
    existing = requests.get(url, headers=headers, params=params, timeout=30)
    sha = None
    if existing.status_code == 200:
        sha = existing.json().get("sha")
    elif existing.status_code != 404:
        raise RuntimeError(existing.text)
    payload = {
        "message": message,
        "content": base64.b64encode(local_path.read_bytes()).decode(),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    response = requests.put(url, headers=headers, json=payload, timeout=120)
    if response.status_code not in {200, 201}:
        raise RuntimeError(response.text)


def upload_tree(token: str, repo: str, branch: str, root: Path, prefix: str, message_prefix: str) -> int:
    count = 0
    for path in root.rglob("*.parquet"):
        rel = path.relative_to(root).as_posix()
        upload_file(token, repo, branch, path, f"{prefix}/{rel}", f"{message_prefix}: {rel}")
        count += 1
    return count
