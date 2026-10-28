"""显式选择认证路径，订阅失败时不自动转用 API。"""
import os


def resolve_model_access(value: str | None = None) -> str:
    access = (value or os.environ.get("ASTOCK_MODEL_ACCESS", "subscription")).strip().lower()
    if access not in {"subscription", "api"}:
        raise ValueError("ASTOCK_MODEL_ACCESS 必须是 subscription 或 api")
    return access
