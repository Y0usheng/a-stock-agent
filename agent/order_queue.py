"""Host-owned, confirmed future-session paper orders. No model or broker access."""
from contextlib import closing, contextmanager
from copy import deepcopy
import json
import math
from pathlib import Path
import sqlite3

from agent.contracts import validate_decision_result, validate_risk_approval
from agent.trading_clock import VERSION, now
from config import DATA_DIR, MARKETS
from evals.experiments import canonical_hash
from servers.data_quality import event_time

DB = DATA_DIR / "paper_order_queue.db"


def account_state(account):
    return {"cash": round(account["cash"], 2), "positions": sorted([
        {"symbol": row["symbol"], "shares": row["shares"], "avg_cost": round(row["avg_cost"], 6),
         "last_buy_date": row.get("last_buy_date")} for row in account["positions"] if row["shares"]], key=lambda row: row["symbol"])}


def state_after(base, fills):
    result = deepcopy(base)
    holdings = {row["symbol"]: row for row in result["positions"]}
    for fill in fills:
        symbol, qty = fill["symbol"], fill["shares"]
        row = holdings.setdefault(symbol, {"symbol": symbol, "shares": 0, "avg_cost": 0., "last_buy_date": None})
        if fill["action"] == "BUY":
            cost = row["avg_cost"] * row["shares"] + qty * fill["price"] + fill["fee"]
            row["shares"] += qty
            row["avg_cost"] = cost / row["shares"]
            row["last_buy_date"] = fill["trade_date"]
            result["cash"] = round(result["cash"] - qty * fill["price"] - fill["fee"], 2)
        else:
            if qty > row["shares"]:
                raise ValueError("队列成交无法重建账户")
            row["shares"] -= qty
            result["cash"] = round(result["cash"] + qty * fill["price"] - fill["fee"], 2)
    result["positions"] = list(holdings.values())
    return account_state(result)


class OrderQueue:
    def __init__(self, path=DB):
        self.path = Path(path)

    @contextmanager
    def db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("""CREATE TABLE IF NOT EXISTS plans (
                id TEXT PRIMARY KEY,payload TEXT NOT NULL,status TEXT NOT NULL,
                approved_at TEXT,execution_snapshot TEXT,error TEXT,updated_at TEXT NOT NULL)""")
            conn.execute("CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,plan_id TEXT,effective_at TEXT,recorded_at TEXT,status TEXT)")
            with conn:
                yield conn

    def rows(self):
        if not self.path.exists():
            return []
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            rows = [dict(row) for row in conn.execute("SELECT * FROM plans ORDER BY rowid")]
        for row in rows:
            row["payload"] = json.loads(row["payload"])
            row["execution_snapshot"] = json.loads(row["execution_snapshot"]) if row["execution_snapshot"] else None
            if canonical_hash(row["payload"]) != row["id"]:
                raise ValueError("待执行计划内容被改写")
        return rows

    def stage(self, result, timing, account):
        symbols = {row["symbol"] for row in result["decisions"]}
        validate_decision_result(result, "cn", expected_symbols=symbols)
        validate_risk_approval(result, "cn", symbols)
        if timing["version"] != VERSION or event_time(timing["information_cutoff_at"], "cn") >= event_time(timing["approval_deadline"], "cn"):
            raise ValueError("未来执行时点非法")
        if timing["price_as_of_date"] != result.get("as_of_date", result.get("date")):
            raise ValueError("计划行情日与冻结决策不一致")
        active = [row for row in result["decisions"] if row["action"] != "HOLD"]
        if not active:
            return None
        if now() >= event_time(timing["approval_deadline"], "cn"):
            raise ValueError("研究完成时已过确认期限，不安排事后成交")
        payload = {"version": VERSION, "market": "cn", "result": deepcopy(result), "timing": deepcopy(timing),
                   "account_state": account_state(account)}
        identifier = canonical_hash(payload)
        with self.db() as conn:
            stamp = now().isoformat()
            cursor = conn.execute("INSERT OR IGNORE INTO plans VALUES(?,?,?,NULL,NULL,NULL,?)",
                         (identifier, json.dumps(payload, ensure_ascii=False, allow_nan=False), "awaiting_confirmation", stamp))
            if cursor.rowcount:
                conn.execute("INSERT INTO events(plan_id,effective_at,recorded_at,status) VALUES(?,?,?,?)", (identifier, stamp, stamp, "awaiting_confirmation"))
        return identifier

    def update(self, identifier, status, *, error=None, snapshot=None, approved_at=None, effective_at=None):
        with self.db() as conn:
            conn.execute("UPDATE plans SET status=?,error=?,updated_at=?,execution_snapshot=COALESCE(?,execution_snapshot),approved_at=COALESCE(?,approved_at) WHERE id=?",
                (status, error, now().isoformat(), json.dumps(snapshot, ensure_ascii=False) if snapshot else None, approved_at, identifier))
            stamp = now().isoformat()
            conn.execute("INSERT INTO events(plan_id,effective_at,recorded_at,status) VALUES(?,?,?,?)", (identifier, effective_at or approved_at or stamp, stamp, status))

    def public_state(self, day):
        """Only state effective by the requested day; never expose local approval hashes."""
        summaries = []
        if not self.path.exists():
            return summaries
        rows = self.rows()
        states = self.states_at(day)
        for row in rows:
            if row["id"] not in states:
                continue
            status, recorded = states[row["id"]]
            timing = row["payload"]["timing"]
            for decision in row["payload"]["result"]["decisions"]:
                if decision["action"] != "HOLD":
                    summaries.append({"symbol": decision["symbol"], "action": decision["action"], "shares": decision["shares"],
                        "status": status, "execution_session": timing["execution_session"], "approval_deadline": timing["approval_deadline"],
                        "information_cutoff_at": timing["information_cutoff_at"], "state_recorded_at": recorded,
                        "reference_price": decision["price"], "max_price_deviation_pct": timing["max_price_deviation_pct"]})
        return summaries

    def states_at(self, day):
        from servers.data_quality import market_close
        end = market_close(day, "cn").replace(hour=23, minute=59, second=59)
        if not self.path.exists():
            return {}
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            events = conn.execute("SELECT plan_id,effective_at,recorded_at,status FROM events ORDER BY id").fetchall()
        states = {}
        for identifier, effective, recorded, status in events:
            if event_time(effective, "cn") <= end:
                states[identifier] = (status, recorded)
        return states

    def confirm(self, identifier, approve=True, at=None):
        row = next((row for row in self.rows() if row["id"] == identifier), None)
        if row and not approve and row["status"] in {"awaiting_confirmation", "confirmed", "executing", "blocked"}:
            self.update(identifier, "rejected", error="user_cancelled_remaining_orders")
            return {"status": "rejected_remaining_orders", "plan_hash": identifier, "orders_submitted": 0}
        if not row or row["status"] != "awaiting_confirmation":
            raise ValueError("没有匹配的待确认计划；确认不会重新运行模型")
        at = event_time(at, "cn") if at else now()
        if at >= event_time(row["payload"]["timing"]["approval_deadline"], "cn"):
            self.update(identifier, "expired", error="confirmation_after_deadline")
            raise ValueError("已过确认期限，不能事后选择收盘成交")
        if at < event_time(row["payload"]["timing"]["information_cutoff_at"], "cn"):
            raise ValueError("确认不能早于计划的信息截止")
        self.update(identifier, "confirmed" if approve else "rejected", approved_at=at.isoformat())
        return {"status": "confirmed_for_future_session" if approve else "rejected", "plan_hash": identifier,
                "execution_session": row["payload"]["timing"]["execution_session"], "orders_submitted": 0}


def validate_quote(quote, symbol, day):
    fields = {key: value for key, value in quote.items() if key not in {"fingerprint", "observed_at"}}
    if (quote.get("symbol") != symbol or quote.get("date") != day or quote.get("price_basis") != "unadjusted"
            or quote.get("data_quality") != "live_prices" or canonical_hash(fields) != quote.get("fingerprint")):
        raise ValueError("执行行情日期、来源或指纹不符")
    for field in ("close", "high", "low"):
        if isinstance(quote.get(field), bool) or not isinstance(quote.get(field), (int, float)) or not math.isfinite(quote[field]) or quote[field] <= 0:
            raise ValueError("执行价格无效")
    if type(quote.get("volume")) is not int or quote["volume"] <= 0:
        raise ValueError("执行日无有效成交量")


async def execute_due(host, queue=None, at=None):
    queue = queue or OrderQueue()
    at = event_time(at, "cn") if at else now()
    receipts = []
    for row in queue.rows():
        payload, identifier = row["payload"], row["id"]
        timing, result = payload["timing"], payload["result"]
        if row["status"] == "awaiting_confirmation" and at >= event_time(timing["approval_deadline"], "cn"):
            queue.update(identifier, "expired", error="not_confirmed_before_deadline")
            receipts.append({"plan_hash": identifier, "status": "expired", "fills": 0})
            continue
        if row["status"] not in {"confirmed", "executing"} or at < event_time(timing["execution_at"], "cn"):
            continue
        if not row["approved_at"] or event_time(row["approved_at"], "cn") >= event_time(timing["approval_deadline"], "cn"):
            raise ValueError("执行计划缺少有效的事前确认")
        decisions = sorted((decision for decision in result["decisions"] if decision["action"] != "HOLD"), key=lambda d: (d["action"] != "SELL", d["symbol"]))
        day, run_id = timing["execution_session"], "cn:" + timing["execution_session"] + ":queued:" + identifier[:24]
        fills = await host.call("get_trades", {"limit": 100, "run_id": run_id})
        expected_ids = {identifier + ":" + decision["symbol"]: decision for decision in decisions}
        for fill in fills:
            decision = expected_ids.get(fill["order_id"])
            if not decision or fill["shares"] != decision["shares"] or fill["action"] != decision["action"] or fill["trade_date"] != day:
                raise ValueError("成交账本与冻结队列不一致")
        quotes = row["execution_snapshot"]
        try:
            symbols = sorted({d["symbol"] for d in decisions} | {p["symbol"] for p in payload["account_state"]["positions"]})
            if not quotes:
                quotes = {symbol: await host.call("execution_session_quote", {"symbol": symbol, "day": day}) for symbol in symbols}
                for symbol in symbols:
                    validate_quote(quotes[symbol], symbol, day)
                queue.update(identifier, "executing", snapshot=quotes)
            for symbol in symbols:
                validate_quote(quotes[symbol], symbol, day)
            prices = {symbol: quotes[symbol]["close"] for symbol in symbols}
            account = await host.call("get_account", {"current_prices": prices})
            expected = state_after(payload["account_state"], fills)
            actual = account_state(account)
            same = (abs(actual["cash"] - expected["cash"]) <= .01 and len(actual["positions"]) == len(expected["positions"]))
            for left, right in zip(actual["positions"], expected["positions"]):
                same = same and left["symbol"] == right["symbol"] and left["shares"] == right["shares"] and left["last_buy_date"] == right["last_buy_date"] and abs(left["avg_cost"] - right["avg_cost"]) <= .000002
            if not same:
                queue.update(identifier, "blocked", error="account_changed_outside_this_plan")
                receipts.append({"plan_hash": identifier, "status": "blocked", "fills": len(fills)})
                continue
            authorization = await host.call("authorize_run", {"run_id": run_id, "trade_date": day, "approved_symbols": symbols})
            if not authorization.get("ok"):
                raise RuntimeError("账户拒绝未来计划运行授权")
            errors, seen = [], {fill["order_id"] for fill in fills}
            cases = {case["symbol"]: case for case in result.get("investment_cases", [])}
            for decision in decisions:
                symbol, action = decision["symbol"], decision["action"]
                order_id = identifier + ":" + symbol
                if order_id in seen:
                    continue
                quote = quotes[symbol]
                direction = 1 if action == "BUY" else -1
                fill_price = round(quote["close"] * (1 + direction * MARKETS["cn"]["slippage_bps"] / 10000), 4)
                bound = decision["price"] * (1 + direction * timing["max_price_deviation_pct"])
                if quote["high"] == quote["low"] or (action == "BUY" and fill_price > bound) or (action == "SELL" and fill_price < bound):
                    errors.append({"symbol": symbol, "reason": "price_limit_or_uncertain_one_price_session"})
                    continue
                account = await host.call("get_account", {"current_prices": prices})
                if action == "BUY" and symbol in cases:
                    held = sum(p["shares"] for p in account["positions"] if p["symbol"] == symbol)
                    if (held + decision["shares"]) * fill_price > account["total_assets"] * cases[symbol]["target_weight"] + .01:
                        errors.append({"symbol": symbol, "reason": "execution_exceeds_frozen_target_weight"})
                        continue
                args = {"symbol": symbol, "shares": decision["shares"], "price": quote["close"],
                        "reason": "已事前确认的次交易日收盘模拟计划", "run_id": run_id, "order_id": order_id,
                        "trade_date": day, "daily_volume": quote["volume"]}
                if action == "BUY":
                    args.update(name=decision["name"], current_prices=prices)
                response = await host.call(action.lower(), args)
                if not response.get("ok"):
                    errors.append({"symbol": symbol, "reason": "account_execution_rejected"})
            final_fills = await host.call("get_trades", {"limit": 100, "run_id": run_id})
            status = "filled" if len(final_fills) == len(decisions) else "partially_filled" if final_fills else "cancelled"
            valuation = await host.call("record_valuation_snapshot", {"run_id": run_id, "snapshot_date": day, "current_prices": prices,
                                                          "data_sources": sorted({q["source"] for q in quotes.values()})})
            if not valuation.get("ok"):
                raise RuntimeError("执行后估值未登记，保留幂等恢复状态")
            queue.update(identifier, status, error=json.dumps(errors, ensure_ascii=False) if errors else None, effective_at=timing["execution_at"])
            receipts.append({"plan_hash": identifier, "status": status, "fills": len(final_fills), "errors": errors,
                             "execution_session": day, "price_basis": "unadjusted"})
        except (OSError, ValueError, RuntimeError) as exc:
            # Keep executable state for source outages or interrupted calls; native order IDs recover fills.
            queue.update(identifier, "executing", error=type(exc).__name__ + ": execution_pending_or_source_unavailable")
            receipts.append({"plan_hash": identifier, "status": "execution_pending", "fills": len(fills)})
    return receipts
