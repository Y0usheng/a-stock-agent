# -*- coding: utf-8 -*-
"""只读分析截面与独立实验日志。"""
from __future__ import annotations

from copy import deepcopy
import json
import os
import tempfile
from pathlib import Path

from evals.experiments import canonical_hash, decision_log_variant


READ_TOOLS = {
    "candidate_snapshot", "candidate_fundamentals", "get_history", "get_indicators",
    "get_fundamentals", "get_symbol_news", "get_news_evidence", "get_account", "get_trades",
}


class AnalysisHost:
    """所有变体共用只读缓存，防止分析期间重复拉取不同的账户/基本面截面。"""

    def __init__(self, host, snapshot: list[dict], fundamentals: list[dict],
                 account: dict, recent_trades: list[dict]):
        self._host = host
        self.market = host.market
        self.active_symbols = list(host.active_symbols)
        self.news_screen = deepcopy(host.news_screen)
        self._cache: dict[str, object] = {}
        self._requests: dict[str, dict] = {}
        self._snapshot = {item["symbol"]: deepcopy(item) for item in snapshot}
        self._fundamentals = {item["symbol"]: deepcopy(item) for item in fundamentals}
        self._account = deepcopy(account)
        self._recent_trades = deepcopy(recent_trades)

    async def call(self, name: str, args: dict):
        if name not in READ_TOOLS:
            raise ValueError(f"只读分析禁止调用 {name}")
        symbols = args.get("symbols", [])
        requested = [args["symbol"]] if "symbol" in args else symbols
        if any(symbol not in self.active_symbols for symbol in requested):
            raise ValueError("分析工具只允许访问当前候选池")
        if name == "candidate_snapshot":
            return [deepcopy(self._snapshot[symbol]) for symbol in symbols]
        if name == "get_indicators":
            return deepcopy(self._snapshot[args["symbol"]])
        if name == "candidate_fundamentals":
            return [deepcopy(self._fundamentals[symbol]) for symbol in symbols]
        if name == "get_fundamentals":
            return deepcopy(self._fundamentals[args["symbol"]])
        if name == "get_account":
            return deepcopy(self._account)
        if name == "get_trades" and not args.get("run_id") and args.get("limit", 20) <= 20:
            return deepcopy(self._recent_trades[:max(0, args.get("limit", 20))])
        key = canonical_hash({"tool": name, "args": args})
        if key not in self._cache:
            value = deepcopy(await self._host.call(name, args))
            if name == "get_history":
                cutoff = self.news_screen.get("as_of_date")
                if cutoff:
                    value = [item for item in value if str(item.get("date", "")) <= cutoff]
            self._cache[key] = value
            self._requests[key] = {"tool": name, "args": deepcopy(args)}
        return deepcopy(self._cache[key])

    def dataset(self, as_of_date: str, memory: dict) -> dict:
        data = {
            "schema": "analysis-input-v1", "market": self.market, "as_of_date": as_of_date,
            "active_symbols": self.active_symbols,
            "snapshot": list(self._snapshot.values()),
            "fundamentals": list(self._fundamentals.values()),
            "news_screen": self.news_screen, "account": self._account,
            "recent_trades": self._recent_trades, "memory": memory,
            "tool_data": [{**self._requests[key], "result": self._cache[key]}
                          for key in sorted(self._cache)],
        }
        if getattr(self, "investment_context", None) is not None:
            data["investment_context"] = self.investment_context
        return deepcopy(data)


def shadow_log_path(log_dir: Path, as_of_date: str, market: str, metadata: dict) -> Path:
    variant_id = canonical_hash(decision_log_variant(metadata))[:16]
    return log_dir / "shadow" / f"{as_of_date}.{variant_id}.{market}.json"


def write_log(path: Path, result: dict) -> None:
    """同目录临时文件原子替换，评分器不会读取半份 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
