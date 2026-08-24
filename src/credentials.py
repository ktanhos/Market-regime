"""Nơi DUY NHẤT xác định API key của vnstock.

Module này cố tình không import Streamlit và không import vnstock. Nó chỉ trả
lời một câu hỏi: *khóa nào sẽ được dùng, và nó đến từ đâu*. Việc đọc
``st.secrets`` thuộc về ``app.py``; việc cấu hình client thuộc về
``src.vnstock_data``.

Thứ tự ưu tiên::

    1. Khóa người dùng nhập trong phiên Streamlit
    2. Streamlit Secrets
    3. Biến môi trường VNSTOCK_API_KEY
    4. Không có khóa (chạy ở gói Khách)

Giá trị khóa không bao giờ được ghi ra đĩa, vào nhật ký, vào dữ liệu hay lên
GitHub. Chỗ duy nhất được phép hiển thị là dạng che ``mask_secret``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from src import config

SOURCE_SESSION = "session"
SOURCE_SECRETS = "secrets"
SOURCE_ENV = "env"
SOURCE_NONE = "none"

SOURCE_LABELS = {
    SOURCE_SESSION: "khóa nhập trong phiên",
    SOURCE_SECRETS: "Streamlit Secrets",
    SOURCE_ENV: f"biến môi trường {config.VNSTOCK_API_KEY_ENV}",
    SOURCE_NONE: "chưa cấu hình",
}

# Khóa ngắn hơn mức này bị chính vnstock từ chối, không cần thử.
MIN_KEY_LENGTH = 10


def mask_secret(value: str | None) -> str:
    """Dạng an toàn để hiển thị. Không bao giờ lộ đủ ký tự để dùng lại."""
    if not value:
        return "—"
    text = str(value).strip()
    if len(text) <= 4:
        return "•" * len(text)
    return f"{'•' * 8}{text[-4:]}"


@dataclass(frozen=True)
class ApiCredentials:
    """Khóa đã được xác định, kèm nguồn gốc của nó.

    ``__repr__`` bị ghi đè để một lần in nhầm đối tượng này ra log cũng không
    làm lộ khóa.
    """

    key: str | None = None
    source: str = SOURCE_NONE

    @property
    def configured(self) -> bool:
        return bool(self.key)

    @property
    def usable(self) -> bool:
        """Đủ dài để vnstock chấp nhận."""
        return bool(self.key) and len(self.key) >= MIN_KEY_LENGTH

    @property
    def source_label(self) -> str:
        return SOURCE_LABELS.get(self.source, self.source)

    @property
    def masked(self) -> str:
        return mask_secret(self.key)

    def __repr__(self) -> str:  # pragma: no cover - chỉ để tránh lộ khóa
        return f"ApiCredentials(source={self.source!r}, configured={self.configured})"

    __str__ = __repr__


def _clean(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_vnstock_api_key(
    session_key: object = None,
    secrets_key: object = None,
    env: Mapping[str, str] | None = None,
) -> ApiCredentials:
    """Xác định khóa theo đúng thứ tự ưu tiên.

    ``session_key`` và ``secrets_key`` do tầng giao diện truyền vào, vì chỉ
    ``app.py`` mới biết tới ``st.session_state`` và ``st.secrets``. Chạy ngoài
    Streamlit thì cả hai đều None và chỉ còn biến môi trường.
    """
    env = os.environ if env is None else env

    for value, source in (
        (session_key, SOURCE_SESSION),
        (secrets_key, SOURCE_SECRETS),
        (env.get(config.VNSTOCK_API_KEY_ENV), SOURCE_ENV),
    ):
        cleaned = _clean(value)
        if cleaned:
            return ApiCredentials(key=cleaned, source=source)

    return ApiCredentials()


def contains_secret(payload: object, credentials: ApiCredentials) -> bool:
    """Kiểm tra một cấu trúc bất kỳ có lẫn giá trị khóa hay không.

    Dùng trong kiểm thử để bảo đảm khóa không lọt vào nhật ký hay dữ liệu.
    """
    if not credentials.configured:
        return False
    return credentials.key in _flatten(payload)


def _flatten(payload: object) -> str:
    if isinstance(payload, Mapping):
        return " ".join(f"{k} {_flatten(v)}" for k, v in payload.items())
    if isinstance(payload, (list, tuple, set)):
        return " ".join(_flatten(item) for item in payload)
    return str(payload)
