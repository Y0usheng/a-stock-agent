"""运行前检查、单市场进程锁、运行状态和失败隔离。"""
from __future__ import annotations

from contextlib import contextmanager, closing
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import sqlite3
import subprocess
from uuid import uuid4

from agent.runtime import model_settings, ModelBudget, positive_env
from agent.sdk_runner import resolve_role_routing
from agent.errors import safe_error
from agent.model_access import resolve_model_access
from config import DATA_DIR

OPS_DB = Path(os.environ.get("ASTOCK_OPS_DB", DATA_DIR / "operations.db"))


def preflight(mock=False, framework=None, model_access=None) -> dict:
    checks = []
    access = resolve_model_access(model_access)
    framework = framework or ("sdk" if mock or access == "api" else "langgraph")
    try:
        models = {role: value.model for role, value in resolve_role_routing().items()}
        if access == "subscription" and not mock:
            from agent.subscription import subscription_settings
            settings = subscription_settings()
        else:
            settings = {"reasoning_effort": model_settings().reasoning.effort}
            ModelBudget()
        positive_env("ASTOCK_ANALYSIS_TIMEOUT", 600)
        checks.append({"name": "模型配置", "ok": True})
    except (ValueError, TypeError) as exc:
        models = {}
        settings = None
        checks.append({"name": "模型配置", "ok": False, "message": str(exc)})
    if mock:
        checks.append({"name": "认证", "ok": True, "message": "规则模式不调用模型"})
    elif access == "subscription":
        checks.append({"name": "订阅编排", "ok": framework == "langgraph",
                       "message": "订阅模式使用 langgraph，SDK 编排仅供显式 API 对照"})
        try:
            from agent.subscription import check_subscription_login
            check_subscription_login()
            checks.append({"name": "ChatGPT 订阅登录", "ok": True})
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            checks.append({"name": "ChatGPT 订阅登录", "ok": False, "message": safe_error(exc)})
    else:
        checks.append({"name": "本机 API 密钥", "ok": bool(os.environ.get("OPENAI_API_KEY")),
                       "message": "显式 API 模式；只检查配置，不输出密钥"})
    modules = ["mcp", "agents", "pandas", "akshare", "yfinance"]
    if framework == "langgraph":
        modules += ["langgraph", "langgraph.checkpoint.sqlite"]
    for module in modules:
        try:
            exists = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            exists = False
        checks.append({"name": module, "ok": exists})
    return {"ok": all(check["ok"] for check in checks), "framework": framework, "model_access": access,
            "models": models, "reasoning_effort": settings["reasoning_effort"] if settings else None,
            "checks": checks}


@contextmanager
def market_lock(market: str, root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{market}.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"{market} 已有运行中的任务，请等待其结束") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def _database(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY, market TEXT, started_at TEXT, finished_at TEXT,
        status TEXT, mode TEXT, error TEXT, log_path TEXT)""")
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def monitored_markets(markets, runner, kwargs: dict, *, db_path=OPS_DB) -> list[dict]:
    results = []
    for market in markets:
        run_id = uuid4().hex
        mode = ("mock" if kwargs.get("mock") else kwargs.get("framework", "sdk")) + (
            ":shadow" if kwargs.get("shadow") else ":paper")
        try:
            with market_lock(market, Path(db_path).parent / "run_locks"):
                with _database(db_path) as conn:
                    conn.execute("UPDATE runs SET status='interrupted',finished_at=? WHERE market=? AND status='running'",
                                 (_now(), market))
                    conn.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                                 (run_id, market, _now(), None, "running", mode, "", ""))
                try:
                    response = await runner(market=market, **kwargs) or {"status": "complete"}
                    status = response.get("status", "complete")
                    error = ""
                except Exception as exc:
                    response = {}
                    status, error = "failed", safe_error(exc)
                    print(f"{market} 本轮失败: {error}")
                except BaseException:
                    with _database(db_path) as conn:
                        conn.execute("UPDATE runs SET status='interrupted',finished_at=? WHERE id=?", (_now(), run_id))
                    raise
                with _database(db_path) as conn:
                    conn.execute("UPDATE runs SET status=?,finished_at=?,error=?,log_path=? WHERE id=?",
                                 (status, _now(), error, response.get("log_path", ""), run_id))
                results.append({"market": market, "status": status, "error": error, **response})
        except Exception as exc:
            results.append({"market": market, "status": "failed", "error": safe_error(exc)})
            print(f"{market} 未能开始或保存运行: {safe_error(exc)}")
    return results


def operation_summary(db_path=OPS_DB) -> dict:
    if not Path(db_path).exists():
        return {"latest": {}, "recent": []}
    with closing(sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(row) for row in conn.execute("SELECT * FROM runs ORDER BY started_at DESC,rowid DESC LIMIT 40")]
    latest = {}
    for row in rows:
        latest.setdefault(row["market"], row)
    return {"latest": latest, "recent": rows}
