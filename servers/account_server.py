# -*- coding: utf-8 -*-
"""
模拟账户 MCP Server(市场参数化)
=================================
通过环境变量 ASTOCK_MARKET(cn/us)加载对应市场规则,同一份代码服务两个市场:
- A股:一手100股、T+1、佣金万2.5(最低5元)、卖出印花税万5
- 美股:可买1股、T+0、零佣金

学习要点:工具的返回值要对 LLM 友好——出错时返回结构化的
{"ok": False, "error": ...} 而不是抛异常,agent 才能理解并调整。
"""
import os
import sys
import math
import json
import sqlite3
import warnings
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MARKETS, MAX_POSITION_PCT

warnings.filterwarnings("ignore", message=r"Field 'lifespan' has an incomplete definition.*")
from mcp.server.fastmcp import FastMCP

MARKET = os.environ.get("ASTOCK_MARKET", "cn")
if MARKET not in MARKETS:
    raise ValueError(f"不支持的 ASTOCK_MARKET={MARKET!r},可选: {', '.join(MARKETS)}")
M = MARKETS[MARKET]
DB_PATH = M["db"]
INITIAL_CASH = M["initial_cash"]
LOT = M["lot_size"]
# 账户只接受配置中的受控候选宇宙；每日真正可交易的更小集合仍由宿主
# 决策契约限定，避免新闻文本直接扩大交易权限。
WATCHLIST = {item["symbol"]: item["name"] for item in M["candidate_universe"]}

mcp = FastMCP(f"stock-account-{MARKET}", log_level="ERROR")


def _db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value REAL);
    CREATE TABLE IF NOT EXISTS positions (
        symbol TEXT PRIMARY KEY, name TEXT, shares INTEGER,
        avg_cost REAL, last_buy_date TEXT);
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT CURRENT_TIMESTAMP,
        trade_date TEXT, symbol TEXT, name TEXT, action TEXT,
        shares INTEGER, price REAL, fee REAL, reason TEXT,
        run_id TEXT, order_id TEXT);
    CREATE TABLE IF NOT EXISTS run_authorizations (
        run_id TEXT PRIMARY KEY, trade_date TEXT NOT NULL,
        approved_symbols TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
    """)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(trades)")}
    if "run_id" not in columns:
        conn.execute("ALTER TABLE trades ADD COLUMN run_id TEXT")
    if "order_id" not in columns:
        conn.execute("ALTER TABLE trades ADD COLUMN order_id TEXT")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_trades_order_id "
        "ON trades(order_id) WHERE order_id IS NOT NULL AND order_id <> ''"
    )
    if conn.execute("SELECT value FROM meta WHERE key='cash'").fetchone() is None:
        conn.execute("INSERT INTO meta VALUES ('cash', ?)", (INITIAL_CASH,))
        conn.commit()
    return conn


def _cash(conn) -> float:
    return conn.execute("SELECT value FROM meta WHERE key='cash'").fetchone()[0]


def _set_cash(conn, v: float):
    conn.execute("UPDATE meta SET value=? WHERE key='cash'", (v,))


def _fee(amount: float, is_sell: bool) -> float:
    fee = max(amount * M["commission_rate"], M["commission_min"]) if M["commission_rate"] > 0 else 0.0
    if is_sell:
        fee += amount * M["stamp_tax_rate"]
    return round(fee, 2)


def _price_error(price: float) -> str | None:
    if isinstance(price, bool) or not isinstance(price, (int, float)):
        return "价格必须是数字"
    if not math.isfinite(float(price)) or price <= 0:
        return "价格必须是大于0的有限数"
    return None


def _shares_error(shares: int) -> str | None:
    if isinstance(shares, bool) or not isinstance(shares, int):
        return "股数必须是整数"
    if shares <= 0 or shares % LOT != 0:
        return f"股数必须为{LOT}的正整数倍"
    return None


def _date_error(trade_date: str) -> str | None:
    try:
        parsed = date.fromisoformat(str(trade_date))
    except (TypeError, ValueError):
        return "trade_date 必须是 YYYY-MM-DD 格式"
    if parsed.isoformat() != trade_date:
        return "trade_date 必须是 YYYY-MM-DD 格式"
    return None


def _order_context(conn, run_id: str, order_id: str, trade_date: str,
                   symbol: str, action: str, shares: int, price: float):
    """校验宿主授权；返回 (error, existing_trade)。"""
    if not run_id or len(run_id) > 160:
        return "缺少或非法 run_id", None
    if not order_id or len(order_id) > 220:
        return "缺少或非法 order_id", None
    if error := _date_error(trade_date):
        return error, None
    authorization = conn.execute(
        "SELECT trade_date, approved_symbols FROM run_authorizations WHERE run_id=?",
        (run_id,),
    ).fetchone()
    if authorization is None:
        return "当前运行尚未获得交易授权", None
    approved = set(json.loads(authorization["approved_symbols"]))
    if authorization["trade_date"] != trade_date:
        return "trade_date 与运行授权不一致", None
    if symbol not in approved:
        return f"{symbol} 不在当前运行的每日批准名单内", None
    existing = conn.execute("SELECT * FROM trades WHERE order_id=?", (order_id,)).fetchone()
    if existing is not None:
        same_order = (
            existing["run_id"] == run_id
            and existing["trade_date"] == trade_date
            and existing["symbol"] == symbol
            and existing["action"] == action
            and existing["shares"] == shares
            and abs(float(existing["price"]) - price) < 1e-8
        )
        if not same_order:
            return "order_id 已被不同订单使用，拒绝重复执行", None
    return None, existing


@mcp.tool()
def authorize_run(run_id: str, trade_date: str, approved_symbols: list[str]) -> dict:
    """由宿主在执行前登记本轮唯一标识、行情日和每日批准股票名单。"""
    run_id = str(run_id).strip()
    if not run_id or len(run_id) > 160:
        return {"ok": False, "error": "缺少或非法 run_id"}
    if error := _date_error(trade_date):
        return {"ok": False, "error": error}
    normalized = sorted(set(str(symbol).strip().upper() for symbol in approved_symbols))
    if not normalized:
        return {"ok": False, "error": "每日批准名单不能为空"}
    invalid = [symbol for symbol in normalized if symbol not in WATCHLIST]
    if invalid:
        return {"ok": False, "error": f"批准名单包含未配置股票:{', '.join(invalid)}"}
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    conn = _db()
    existing = conn.execute(
        "SELECT trade_date, approved_symbols FROM run_authorizations WHERE run_id=?", (run_id,)
    ).fetchone()
    if existing is not None:
        if existing["trade_date"] != trade_date or existing["approved_symbols"] != payload:
            conn.close()
            return {"ok": False, "error": "run_id 已存在且授权内容不同"}
        conn.close()
        return {"ok": True, "duplicate": True, "run_id": run_id,
                "trade_date": trade_date, "approved_symbols": normalized}
    conn.execute(
        "INSERT INTO run_authorizations(run_id,trade_date,approved_symbols) VALUES (?,?,?)",
        (run_id, trade_date, payload),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "duplicate": False, "run_id": run_id,
            "trade_date": trade_date, "approved_symbols": normalized}


def _estimated_total_assets(conn, cash: float,
                            current_prices: dict[str, float] | None = None) -> float:
    """用调用方提供的最新价估算总资产，缺失价格时保守回退到持仓成本。"""
    prices = current_prices or {}
    market_value = 0.0
    for row in conn.execute("SELECT symbol, shares, avg_cost FROM positions WHERE shares>0"):
        px = prices.get(row["symbol"], row["avg_cost"])
        if _price_error(px):
            px = row["avg_cost"]
        market_value += row["shares"] * float(px)
    return cash + market_value


@mcp.tool()
def get_account(current_prices: dict[str, float] | None = None) -> dict:
    """查询账户:现金、持仓明细、总资产。
    可选传入 current_prices({symbol: 最新价})以计算市值和浮动盈亏"""
    conn = _db()
    cash = _cash(conn)
    positions, mv = [], 0.0
    for r in conn.execute("SELECT * FROM positions WHERE shares>0"):
        p = dict(r)
        px = (current_prices or {}).get(r["symbol"])
        if px:
            p["market_value"] = round(px * r["shares"], 2)
            p["pnl_pct"] = round((px / r["avg_cost"] - 1) * 100, 2)
            mv += p["market_value"]
        positions.append(p)
    conn.close()
    return {"market": MARKET, "currency": M["currency"],
            "cash": round(cash, 2), "positions": positions,
            "total_assets": round(cash + mv, 2) if current_prices else None,
            "initial_cash": INITIAL_CASH}


@mcp.tool()
def buy(symbol: str, name: str, shares: int, price: float, reason: str,
        run_id: str, order_id: str, trade_date: str,
        current_prices: dict[str, float] | None = None) -> dict:
    """买入股票。A股股数须为100的整数倍,美股可买1股。
    受限制:仅限股票池、现金不足或单股仓位超过总资产30%会被拒绝。
    current_prices 可传股票池最新价以精确计算仓位；reason 记录买入理由供复盘"""
    symbol = symbol.strip().upper()
    if symbol not in WATCHLIST:
        return {"ok": False, "error": f"{symbol} 不在当前市场股票池内"}
    if error := _shares_error(shares):
        return {"ok": False, "error": error}
    if error := _price_error(price):
        return {"ok": False, "error": error}
    price = float(price)
    name = WATCHLIST[symbol]  # 账户统一使用配置中的规范名称
    conn = _db()
    error, existing = _order_context(
        conn, run_id, order_id, trade_date, symbol, "BUY", shares, price
    )
    if error:
        conn.close()
        return {"ok": False, "error": error}
    if existing is not None:
        cash = _cash(conn)
        conn.close()
        return {"ok": True, "duplicate": True, "symbol": symbol, "shares": shares,
                "price": price, "fee": existing["fee"], "cash_after": round(cash, 2)}
    amount = shares * price
    fee = _fee(amount, is_sell=False)
    cash = _cash(conn)
    if amount + fee > cash:
        conn.close()
        return {"ok": False, "error": f"现金不足:需 {amount+fee:.2f},仅有 {cash:.2f}"}
    row = conn.execute("SELECT shares, avg_cost FROM positions WHERE symbol=?", (symbol,)).fetchone()
    held_cost = (row["shares"] * row["avg_cost"]) if row else 0.0
    prices = dict(current_prices or {})
    prices[symbol] = price
    total_est = _estimated_total_assets(conn, cash, prices)
    held_shares = row["shares"] if row else 0
    position_after = (held_shares + shares) * price
    if position_after > total_est * MAX_POSITION_PCT:
        conn.close()
        return {"ok": False, "error": f"超过单股最大仓位 {MAX_POSITION_PCT:.0%},买入后市值 {position_after:.0f},总资产估算 {total_est:.0f}"}
    if row:
        new_shares = row["shares"] + shares
        new_cost = (held_cost + amount + fee) / new_shares
        conn.execute("UPDATE positions SET shares=?, avg_cost=?, last_buy_date=? WHERE symbol=?",
                     (new_shares, new_cost, trade_date, symbol))
    else:
        conn.execute("INSERT INTO positions VALUES (?,?,?,?,?)",
                     (symbol, name, shares, (amount + fee) / shares, trade_date))
    _set_cash(conn, cash - amount - fee)
    conn.execute("INSERT INTO trades (trade_date,symbol,name,action,shares,price,fee,reason,run_id,order_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (trade_date, symbol, name, "BUY", shares, price, fee, reason, run_id, order_id))
    conn.commit()
    new_cash = _cash(conn)
    conn.close()
    return {"ok": True, "symbol": symbol, "shares": shares, "price": price,
            "fee": fee, "cash_after": round(new_cash, 2)}


@mcp.tool()
def sell(symbol: str, shares: int, price: float, reason: str,
         run_id: str, order_id: str, trade_date: str) -> dict:
    """卖出股票。A股遵守 T+1(当日买入不可当日卖出),美股无此限制。
    reason 记录卖出理由供复盘"""
    symbol = symbol.strip().upper()
    if error := _shares_error(shares):
        return {"ok": False, "error": error}
    if error := _price_error(price):
        return {"ok": False, "error": error}
    price = float(price)
    conn = _db()
    error, existing = _order_context(
        conn, run_id, order_id, trade_date, symbol, "SELL", shares, price
    )
    if error:
        conn.close()
        return {"ok": False, "error": error}
    if existing is not None:
        cash = _cash(conn)
        conn.close()
        return {"ok": True, "duplicate": True, "symbol": symbol, "shares": shares,
                "price": price, "fee": existing["fee"], "cash_after": round(cash, 2)}
    row = conn.execute("SELECT * FROM positions WHERE symbol=?", (symbol,)).fetchone()
    if not row or row["shares"] < shares:
        conn.close()
        return {"ok": False, "error": f"持仓不足:持有 {row['shares'] if row else 0} 股"}
    if M["t_plus_1"]:
        bought_today = conn.execute(
            "SELECT COALESCE(SUM(shares),0) FROM trades "
            "WHERE symbol=? AND action='BUY' AND trade_date=?", (symbol, trade_date)).fetchone()[0]
        sellable = max(row["shares"] - bought_today, 0)
        if shares > sellable:
            conn.close()
            return {"ok": False, "error": f"T+1限制:今日可卖 {sellable} 股,请求卖出 {shares} 股"}
    amount = shares * price
    fee = _fee(amount, is_sell=True)
    remaining = row["shares"] - shares
    if remaining == 0:
        conn.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
    else:
        conn.execute("UPDATE positions SET shares=? WHERE symbol=?", (remaining, symbol))
    _set_cash(conn, _cash(conn) + amount - fee)
    pnl = (price - row["avg_cost"]) * shares - fee
    conn.execute("INSERT INTO trades (trade_date,symbol,name,action,shares,price,fee,reason,run_id,order_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (trade_date, symbol, row["name"], "SELL", shares, price, fee, reason, run_id, order_id))
    conn.commit()
    new_cash = _cash(conn)
    conn.close()
    return {"ok": True, "symbol": symbol, "shares": shares, "price": price,
            "fee": fee, "realized_pnl": round(pnl, 2), "cash_after": round(new_cash, 2)}


@mcp.tool()
def get_trades(limit: int = 20, run_id: str = "") -> list[dict]:
    """查询最近成交记录(含每笔交易的理由),复盘时使用"""
    conn = _db()
    if run_id:
        query = "SELECT * FROM trades WHERE run_id=? ORDER BY id DESC LIMIT ?"
        params = (run_id, limit)
    else:
        query = "SELECT * FROM trades ORDER BY id DESC LIMIT ?"
        params = (limit,)
    rows = [dict(r) for r in conn.execute(query, params)]
    conn.close()
    return rows


if __name__ == "__main__":
    mcp.run()
