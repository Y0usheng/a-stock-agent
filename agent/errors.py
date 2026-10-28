"""可用于 MCP 子进程的轻量错误脱敏。"""
import os
import re


def safe_error(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        text = "; ".join(safe_error(child) for child in exc.exceptions)
    else:
        text = f"{type(exc).__name__}: {exc}"
    for key, value in os.environ.items():
        if any(tag in key.upper() for tag in ("KEY", "SECRET", "TOKEN", "PASSWORD")) and len(value) >= 8:
            text = text.replace(value, "[已隐藏]")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[已隐藏]", text)[:2000]
