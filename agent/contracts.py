# -*- coding: utf-8 -*-
"""决策输出与真实成交的一致性契约。"""
import math
from copy import deepcopy
from collections import defaultdict

from config import MARKETS


def _complete_rows(rows, expected: set[str], label: str) -> None:
    if not isinstance(rows, list) or any(not isinstance(item, dict) for item in rows):
        raise ValueError(f"{label} 必须是对象数组")
    symbols = [item.get("symbol") for item in rows]
    if any(not isinstance(symbol, str) for symbol in symbols):
        raise ValueError(f"{label} 缺少有效股票代码")
    if len(symbols) != len(set(symbols)):
        raise ValueError(f"{label} 不得重复股票代码")
    if set(symbols) != expected:
        raise ValueError(f"{label} 必须逐只覆盖股票池且不得包含池外标的")


def validate_trade_plan(decisions: list[dict], market: str,
                        expected_symbols: set[str]) -> None:
    configured = {item["symbol"] for item in MARKETS[market]["candidate_universe"]}
    if not expected_symbols or not expected_symbols <= configured:
        raise ValueError("expected_symbols 必须是非空且受控的候选宇宙子集")
    _complete_rows(decisions, expected_symbols, "decisions")
    lot = MARKETS[market]["lot_size"]
    for item in decisions:
        action, shares, price = item.get("action"), item.get("shares"), item.get("price")
        if action not in {"BUY", "SELL", "HOLD"}:
            raise ValueError(f"非法 action:{action}")
        if isinstance(shares, bool) or not isinstance(shares, int) or shares < 0 or shares % lot:
            raise ValueError(f"{item['symbol']} 股数不符合市场最小单位")
        if (action == "HOLD" and shares != 0) or (action != "HOLD" and shares <= 0):
            raise ValueError(f"{item['symbol']} action 与 shares 不一致")
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            raise ValueError(f"{item['symbol']} price 非法")
        for key in ("name", "reason"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"{item['symbol']} 缺少 {key}")


def capture_risk_review(proposed: list[dict], review: dict, market: str,
                        expected_symbols: set[str]) -> dict:
    """仅由风险专家工具出口调用；绑定输入计划，拒绝含糊或不完整的审批。"""
    validate_trade_plan(proposed, market, expected_symbols)
    if not isinstance(review, dict) or not isinstance(review.get("summary"), str) or not review["summary"].strip():
        raise ValueError("风控审批缺少 summary")
    assessments = review.get("assessments")
    _complete_rows(assessments, expected_symbols, "风控 assessments")
    by_symbol = {item["symbol"]: item for item in proposed}
    lot = MARKETS[market]["lot_size"]
    for item in assessments:
        plan = by_symbol[item["symbol"]]
        verdict, maximum = item.get("verdict"), item.get("max_shares")
        if item.get("action") != plan["action"]:
            raise ValueError(f"{item['symbol']} 风控审批与送审方向不同")
        if verdict not in {"APPROVE", "REJECT", "RESIZE"}:
            raise ValueError("非法风控 verdict")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0 or maximum % lot:
            raise ValueError("风控 max_shares 不符合市场最小单位")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("风控审批缺少 reason")
        if verdict == "REJECT" or plan["action"] == "HOLD":
            if maximum != 0 or verdict == "RESIZE":
                raise ValueError("拒绝或 HOLD 的批准股数必须为0")
        elif verdict == "APPROVE":
            if maximum != plan["shares"]:
                raise ValueError("APPROVE 必须保持送审股数；缩量请使用 RESIZE")
        elif not 0 < maximum < plan["shares"]:
            raise ValueError("RESIZE 必须严格减少送审股数且大于0")
    return {"source": "risk-tool", "proposed_decisions": deepcopy(proposed),
            "summary": review["summary"], "assessments": deepcopy(assessments)}


def validate_risk_approval(result: dict, market: str,
                           expected_symbols: set[str]) -> None:
    """执行前整批校验；正文结论和投资经理自行编造的审批不能替代工具审计。"""
    audit = result.get("risk_review")
    if not isinstance(audit, dict) or audit.get("source") != "risk-tool":
        raise ValueError("缺少宿主捕获的风险专家审批，拒绝执行")
    capture_risk_review(audit.get("proposed_decisions"), audit, market, expected_symbols)
    validate_trade_plan(result.get("decisions"), market, expected_symbols)
    approvals = {item["symbol"]: item for item in audit["assessments"]}
    for decision in result["decisions"]:
        if decision["action"] == "HOLD":
            continue
        approval = approvals[decision["symbol"]]
        if (approval["verdict"] == "REJECT" or approval["action"] != decision["action"]
                or decision["shares"] > approval["max_shares"]):
            raise ValueError(f"{decision['symbol']} 最终订单未获批准、改变方向或超过批准股数")


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
    decisions = result.get("decisions")
    validate_trade_plan(decisions, market, expected)

    predictions = result.get("predictions")
    _complete_rows(predictions, expected, "predictions")
    for item in predictions:
        confidence = item.get("confidence")
        from agent.investment import VERSION as INVESTMENT_VERSION
        horizon = 252 if result.get("investment_policy") == INVESTMENT_VERSION else 5
        if item.get("direction") not in {"up", "down", "flat"} or item.get("horizon_days") != horizon:
            raise ValueError(f"{item.get('symbol')} 预测方向或期限非法")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError(f"{item.get('symbol')} 置信度必须在0到1之间")
    if result.get("investment_policy") is not None:
        from agent.investment import validate_cases
        validate_cases(result, expected)

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
