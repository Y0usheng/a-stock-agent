# -*- coding: utf-8 -*-
"""本地实验注册表：固定数据指纹、Prompt/模型版本和评分结果。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EXPERIMENT_DB
from evals.metrics import uses_synthetic_data


SCHEMA_VERSION = "experiment-runs-v1"


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _connect(db_path: Path | str = EXPERIMENT_DB) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS experiment_runs (
        run_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        name TEXT NOT NULL,
        market TEXT NOT NULL,
        dataset_fingerprint TEXT NOT NULL,
        dataset_json TEXT NOT NULL,
        variant_fingerprint TEXT NOT NULL,
        variant_json TEXT NOT NULL,
        metrics_json TEXT NOT NULL,
        status TEXT NOT NULL,
        notes TEXT NOT NULL DEFAULT ''
    );
    CREATE INDEX IF NOT EXISTS idx_experiment_created ON experiment_runs(created_at DESC);
    """)
    return conn


@contextmanager
def _database(db_path: Path | str = EXPERIMENT_DB):
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def record_experiment(name: str, market: str, dataset: dict, variant: dict,
                      metrics: dict, status: str = "complete", notes: str = "",
                      db_path: Path | str = EXPERIMENT_DB,
                      created_at: str | None = None) -> dict:
    if market not in {"cn", "us", "both"}:
        raise ValueError(f"不支持的市场:{market}")
    if status not in {"complete", "no_data", "failed"}:
        raise ValueError(f"不支持的实验状态:{status}")
    dataset_fingerprint = str(dataset.get("fingerprint") or canonical_hash(dataset))
    variant_fingerprint = canonical_hash(variant)
    created_at = created_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_id = uuid.uuid4().hex
    row = {
        "run_id": run_id, "created_at": created_at, "name": str(name),
        "market": market, "dataset_fingerprint": dataset_fingerprint,
        "dataset": dataset, "variant_fingerprint": variant_fingerprint,
        "variant": variant, "metrics": metrics, "status": status, "notes": str(notes),
    }
    with _database(db_path) as conn:
        conn.execute("""
            INSERT INTO experiment_runs(
                run_id,created_at,name,market,dataset_fingerprint,dataset_json,
                variant_fingerprint,variant_json,metrics_json,status,notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            run_id, created_at, name, market, dataset_fingerprint,
            json.dumps(dataset, ensure_ascii=False, sort_keys=True), variant_fingerprint,
            json.dumps(variant, ensure_ascii=False, sort_keys=True),
            json.dumps(metrics, ensure_ascii=False, sort_keys=True), status, notes,
        ))
    return row


def list_experiments(db_path: Path | str = EXPERIMENT_DB, limit: int = 30) -> list[dict]:
    if not Path(db_path).exists():
        return []
    with _database(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM experiment_runs ORDER BY created_at DESC,run_id DESC LIMIT ?",
            (max(1, min(int(limit), 200)),),
        ).fetchall()
    return [{
        "run_id": row["run_id"], "created_at": row["created_at"],
        "name": row["name"], "market": row["market"],
        "dataset_fingerprint": row["dataset_fingerprint"],
        "dataset": json.loads(row["dataset_json"]),
        "variant_fingerprint": row["variant_fingerprint"],
        "variant": json.loads(row["variant_json"]),
        "metrics": json.loads(row["metrics_json"]),
        "status": row["status"], "notes": row["notes"],
    } for row in rows]


def comparable_experiments(db_path: Path | str = EXPERIMENT_DB) -> list[dict]:
    """只把实验名、市场和数据指纹相同且变体不同的运行组成可比较 A/B 组。"""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for run in reversed(list_experiments(db_path, 200)):
        if run["status"] != "complete":
            continue
        key = (run["name"], run["market"], run["dataset_fingerprint"])
        groups.setdefault(key, []).append(run)
    output = []
    for (name, market, fingerprint), runs in groups.items():
        latest_by_variant = {run["variant_fingerprint"]: run for run in runs}
        variants = list(latest_by_variant.values())
        if len(variants) < 2:
            continue
        baseline = variants[0]
        baseline_accuracy = baseline["metrics"].get("accuracy")
        entries = []
        for run in variants:
            accuracy = run["metrics"].get("accuracy")
            entries.append({
                "run_id": run["run_id"], "created_at": run["created_at"],
                "variant": run["variant"], "metrics": run["metrics"],
                "accuracy_delta_vs_baseline": (
                    round(float(accuracy) - float(baseline_accuracy), 6)
                    if accuracy is not None and baseline_accuracy is not None else None
                ),
            })
        output.append({"name": name, "market": market,
                       "dataset_fingerprint": fingerprint,
                       "baseline_run_id": baseline["run_id"], "variants": entries})
    return output


def decision_log_variant(log: dict) -> dict:
    sdk = log.get("agent_sdk") or {}
    prompt = sdk.get("prompt") or {}
    return {
        "strategy_version": log.get("strategy_version", "unknown"),
        "prompt_version": prompt.get("version", "unversioned"),
        # 完整 prompt 含行情日/候选池，仅用于日志审计，不能充当跨日版本标识。
        # 旧日志没有模板哈希时按原版本分组，与有模板哈希的新日志保持隔离。
        "prompt_template_sha256": prompt.get("template_sha256"),
        "framework": sdk.get("framework", log.get("mode", "unknown")),
        "model_access": sdk.get("model_access", "api"),
        "model_settings": {key: (sdk.get("budget") or {}).get(key)
                           for key in ("reasoning_effort", "max_output_tokens")},
        "role_models": sdk.get("role_models", {}),
        "mode": log.get("mode", "unknown"),
        "execution_mode": log.get("execution_mode", "paper"),
        "offline": uses_synthetic_data(log),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="查看第三阶段实验注册表")
    parser.add_argument("--db", type=Path, default=EXPERIMENT_DB)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps({"schema_version": SCHEMA_VERSION,
                      "runs": list_experiments(args.db, args.limit)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
