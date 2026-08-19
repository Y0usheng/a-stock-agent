# -*- coding: utf-8 -*-
"""决策输出与真实成交的一致性契约。"""
import math
from collections import defaultdict

from config import MARKETS


def validate_decision_result(result: dict, market: str, require_roles: bool = True,
                             expected_symbols: set[str] | None = None) -> None:
    """拒绝不完整或越界的模型结果，避免污染日志和后续 Evals。"""
    if not isinstance(result, dict) or not result.get("summary"):
        raise ValueError("决策结果缺少 summary")
    configured = {item["symbol"] for item in MARKETS[market]["candidate_universe"]}
    expected = set(expected_symbols) if expected_symbols is not None else {
        item["symbol"] for item in MARKETS[market]["watchlist"]
    }
    if not expected or not expected <= configured:
        raise ValueError("expected_symbols 必须是非空且受控的候选宇宙子集")
    lot = MARKETS[market]["lot_size"]

    decisions = result.get("decisions")
    if not isinstance(decisions, list) or {item.get("symbol") for item in decisions} != expected:
        raise ValueError("decisions 必须逐只覆盖股票池且不得包含池外标的")
    for item in decisions:
        action, shares, price = item.get("action"), item.get("shares"), item.get("price")
        if action not in {"BUY", "SELL", "HOLD"}:
            raise ValueError(f"非法 action:{action}")
        if isinstance(shares, bool) or not isinstance(shares, int) or shares < 0 or shares % lot:
            raise ValueError(f"{item.get('symbol')} 股数不符合市场最小单位")
        if (action == "HOLD" and shares != 0) or (action != "HOLD" and shares <= 0):
            raise ValueError(f"{item.get('symbol')} action 与 shares 不一致")
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            raise ValueError(f"{item.get('symbol')} price 非法")
        if not str(item.get("reason", "")).strip():
            raise ValueError(f"{item.get('symbol')} 缺少交易理由")

    predictions = result.get("predictions")
    if not isinstance(predictions, list) or {item.get("symbol") for item in predictions} != expected:
        raise ValueError("predictions 必须逐只覆盖股票池且不得包含池外标的")
    for item in predictions:
        confidence = item.get("confidence")
        if item.get("direction") not in {"up", "down", "flat"} or item.get("horizon_days") != 5:
            raise ValueError(f"{item.get('symbol')} 预测方向或期限非法")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError(f"{item.get('symbol')} 置信度必须在0到1之间")

    if require_roles:
        views = result.get("analyst_views")
        if not isinstance(views, dict) or any(not str(views.get(role, "")).strip()
                                              for role in ("technical", "fundamental", "risk")):
            raise ValueError("多角色结果必须包含技术、基本面和风控真实结论")
        if not str(result.get("memory_assessment", "")).strip():
            raise ValueError("多角色结果缺少 memory_assessment")


def reconcile_executions(result: dict, new_trades: list[dict]) -> dict:
    """核对最终非 HOLD 决策与本轮账本新增成交，返回可写入日志的审计结果。"""
    claimed = defaultdict(int)
    actual = defaultdict(int)
    for item in result.get("decisions", []):
        if item.get("action") != "HOLD":
            claimed[(item["symbol"], item["action"])] += int(item["shares"])
    for item in new_trades:
        actual[(item.get("symbol"), item.get("action"))] += int(item.get("shares", 0))
    keys = set(claimed) | set(actual)
    mismatches = [{"symbol": key[0], "action": key[1],
                   "claimed_shares": claimed[key], "ledger_shares": actual[key]}
                  for key in sorted(keys) if claimed[key] != actual[key]]
    audit = {
        "ok": not mismatches,
        "claimed_trade_count": sum(1 for item in result.get("decisions", []) if item.get("action") != "HOLD"),
        "ledger_trade_count": len(new_trades),
        "share_mismatches": mismatches,
    }
    if not audit["ok"]:
        raise ValueError(f"最终决策与本轮账本成交不一致:{audit}")
    return audit
