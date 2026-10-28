"""订阅连接检查；默认不推理，--smoke 只发送固定测试文字，无市场或账户数据。"""
import argparse
import asyncio
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.subscription import SubscriptionClient, check_subscription_login
from agent.sdk_runner import resolve_role_routing
from agent.errors import safe_error


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="消耗少量订阅额度，验证一次真实 JSON 响应")
    args = parser.parse_args()
    started = perf_counter()
    try:
        login = check_subscription_login()
        model = resolve_role_routing()["investment-manager"].model
        report = {"auth_mode": login["auth_mode"], "model": model, "inference_tested": False}
        if args.smoke:
            client = SubscriptionClient(model)
            result = await client.complete('Return exactly the JSON object with ok=true. Do not use tools.',
                {"type": "object", "properties": {"ok": {"type": "boolean"}},
                 "required": ["ok"], "additionalProperties": False}, role="connection-smoke")
            if result != {"ok": True}:
                raise RuntimeError("订阅测试返回了不符合预期的结果")
            report.update(inference_tested=True, usage=client.report())
        print(json.dumps({"status": "passed", **report,
                          "duration_ms": round((perf_counter() - started) * 1000)}, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": safe_error(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
