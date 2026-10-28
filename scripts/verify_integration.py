"""临时账本双市场验收；不联网、不调用收费模型，不修改正式账本。"""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        env = {**os.environ, "ASTOCK_DATA_DIR": str(root / "data"), "ASTOCK_LOG_DIR": str(root / "logs"),
               "ASTOCK_NEWS_DB": str(root / "news.db"), "ASTOCK_EXPERIMENT_DB": str(root / "experiments.db"),
               "ASTOCK_OPS_DB": str(root / "ops.db"), "ASTOCK_OFFLINE": "1"}
        def run(*args):
            if args and args[0] == "agent/run_daily.py" and "--market" not in args:
                args = (*args, "--market", "both")  # 历史双市场离线验收显式指定，生产默认只跑A股。
            result = subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
            if result.returncode:
                raise RuntimeError(result.stdout[-2500:] + result.stderr[-2500:])
            return result.stdout
        def counts():
            values = {}
            for market in ("cn", "us"):
                with closing(sqlite3.connect(root / "data" / f"account_{market}.db")) as db:
                    values[market] = {table: db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                                      for table in ("trades", "run_authorizations", "valuation_snapshots")}
            return values
        run("agent/run_daily.py", "--mock", "--offline", "--shadow")
        assert all(all(value == 0 for value in item.values()) for item in counts().values())
        logs = {path: path.read_bytes() for path in (root / "logs" / "shadow").glob("*.json")}
        assert len(logs) == 2
        run("agent/run_daily.py", "--mock", "--offline", "--shadow")
        assert all(path.read_bytes() == data for path, data in logs.items())
        run("agent/run_daily.py", "--mock", "--offline")
        after = counts()
        assert all(item["run_authorizations"] == 1 and item["valuation_snapshots"] == 1 for item in after.values())
        run("agent/run_daily.py", "--mock", "--offline", "--force")
        assert counts() == after
        run("agent/run_daily.py", "--mock", "--offline", "--shadow", "--force")
        assert counts() == after
        run("evals/score.py", "--shadow", "--record-experiment")
        run("evals/report.py")
        prices = root / "prices.json"
        first = json.loads(run("evals/backtest.py", "--market", "us", "--window", "60", "--end-date", "2026-09-30",
                              "--dynamic-candidates", "--export-prices", str(prices), "--json"))["us"]
        replay = json.loads(run("evals/backtest.py", "--market", "us", "--window", "60", "--end-date", "2026-09-30",
                               "--dynamic-candidates", "--price-source", "file", "--prices-file", str(prices), "--json"))["us"]
        assert first["price_dataset"]["fingerprint"] == replay["price_dataset"]["fingerprint"]
        assert first["with_news"]["trades"] == replay["with_news"]["trades"]
        # 仪表盘在独立进程中继承临时目录，覆盖所有只读接口。
        check = "from dashboard.app import app; c=app.test_client(); urls=['/','/api/overview','/api/score','/api/performance','/api/experiments','/api/operations']; assert all(c.get(u).status_code==200 for u in urls)"
        run("-c", check)
        print(json.dumps({"status": "passed", "shadow_read_only": True, "daily_idempotency": True,
                          "ledger": after, "price_replay": True, "dashboard_endpoints": 6}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
