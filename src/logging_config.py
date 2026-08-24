"""Cấu hình ghi log dùng chung.

Mỗi module lấy logger qua ``get_logger(__name__)``. Streamlit gọi ``setup()``
một lần khi khởi động, script gọi trong ``main()``.
"""

from __future__ import annotations

import logging

from src import config

_CONFIGURED = False


def setup(level: str | None = None) -> None:
    """Cấu hình handler gốc đúng một lần."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, (level or config.LOG_LEVEL).upper(), logging.INFO),
        format=config.LOG_FORMAT,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup()
    return logging.getLogger(name)
