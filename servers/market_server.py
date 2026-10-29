# -*- coding: utf-8 -*-
"""
A股数据 MCP Server
==================
提供行情、历史K线、技术指标查询工具。
数据源:akshare(东方财富)。只有显式设置 ASTOCK_OFFLINE=1 时才使用
「离线模拟数据」(带种子的随机游走,可复现)；实时模式失败时停止，不静默造数。

学习要点:
- FastMCP 用装饰器把普通函数变成 MCP tool,docstring 即工具说明(会喂给 LLM)
- server 通过 stdio 与 client 通信,可被任何 MCP client 复用(Claude Desktop、Agent SDK、LangGraph...)
"""
import sys
import io
import os
import warnings
from datetime import datetime, timedelta

import pandas as pd
warnings.filterwarnings("ignore", message=r"Field 'lifespan' has an incomplete definition.*")
from mcp.server.fastmcp import FastMCP

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from config import MARKETS
from servers.synthetic_data import random_walk_history
from servers.data_quality import completed_history
from servers.fundamentals_data import mock_fundamentals

WATCHLIST = MARKETS["cn"]["watchlist"]
UNIVERSE = {item["symbol"]: item["name"] for item in MARKETS["cn"]["candidate_universe"]}

# Windows 控制台编码兜底
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    import akshare as ak
    _HAS_AKSHARE = True
except Exception:
    _HAS_AKSHARE = False

mcp = FastMCP("astock-market", log_level="ERROR")

_cache: dict[str, pd.DataFrame] = {}
_data_source = "unknown"
_FORCE_OFFLINE = os.environ.get("ASTOCK_OFFLINE", "").lower() in {"1", "true", "yes"}


def _mock_history(symbol: str, days: int = 120) -> pd.DataFrame:
    """离线模拟数据:种子随机游走,同一股票每次生成结果一致"""
    return random_walk_history(
        symbol, days,
        {"600519": 1450, "300750": 260, "600036": 38,
         "002594": 290, "601899": 18, "600900": 28},
        drift=0.0003, volatility=0.018,
        volume_low=2_000_000, volume_high=80_000_000, high_low_sigma=0.008)


def _fetch_history(symbol: str, days: int = 120) -> pd.DataFrame:
    """离线模式生成 mock；实时模式必须成功获取 akshare 数据。"""
    global _data_source
    key = f"{symbol}:{days}"
    if key in _cache:
        return _cache[key]
    if _FORCE_OFFLINE:
        df = _mock_history(symbol, days)
        _cache[key] = df
        _data_source = "mock(显式离线模拟数据,仅供工程回归)"
        return df
    if _HAS_AKSHARE:
        try:
            start = (datetime.now() - timedelta(days=days * 2)).strftime("%Y%m%d")
            provider = os.environ.get("ASTOCK_CN_PRICE_PROVIDER", "eastmoney")
            if provider == "sina":
                from agent.trading_clock import bounded_requests
                with bounded_requests():
                    df = ak.stock_zh_a_daily(symbol=("sh" if symbol.startswith("6") else "sz") + symbol,
                        start_date=start, end_date=datetime.now().strftime("%Y%m%d"), adjust="qfq")
            elif provider == "eastmoney":
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, adjust="qfq", timeout=30)
                df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume"})
                df["volume"] = df["volume"] * 100
            else:
                raise ValueError("未支持的行情提供商")
            df["date"] = df["date"].astype(str)
            df = completed_history(df, "cn")
            df = df[["date", "open", "high", "low", "close", "volume"]].tail(days).reset_index(drop=True)
            if len(df) > 0:
                _cache[key] = df
                _data_source = "akshare(真实数据)" if provider == "eastmoney" else "akshare(Sina真实前复权日线)"
                return df
        except Exception as e:
            raise RuntimeError(f"akshare 获取 {symbol} 行情失败，实时模式已停止") from e
    raise RuntimeError("akshare 未安装，实时模式禁止使用模拟行情；请安装依赖或使用 --offline")


@mcp.tool()
def get_quote(symbol: str) -> dict:
    """查询单只A股最新行情(收盘价、涨跌幅、成交量)。symbol 为6位代码,如 600519"""
    df = _fetch_history(symbol, 10)
    last, prev = df.iloc[-1], df.iloc[-2]
    return {
        "symbol": symbol,
        "date": last["date"],
        "close": float(last["close"]),
        "change_pct": round((last["close"] / prev["close"] - 1) * 100, 2),
        "volume": int(last["volume"]),
        "data_source": _data_source,
    }


@mcp.tool()
def get_history(symbol: str, days: int = 30) -> list[dict]:
    """查询A股日K线历史(前复权)。返回最近 days 个交易日的 OHLCV"""
    df = _fetch_history(symbol, max(days, 30))
    return df.tail(days).to_dict("records")


@mcp.tool()
def get_indicators(symbol: str) -> dict:
    """计算技术指标:MA5/MA20/MA60、RSI14、MACD。用于判断趋势与超买超卖"""
    df = _fetch_history(symbol, 120)
    close = df["close"]
    ma5, ma20 = close.rolling(5).mean().iloc[-1], close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
    rsi = 100 - 100 / (1 + gain / loss) if loss > 0 else 100.0
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    previous_close = float(close.iloc[-2])
    return {
        "symbol": symbol,
        "date": str(df["date"].iloc[-1]),
        "close": float(close.iloc[-1]),
        "change_pct": round((float(close.iloc[-1]) / previous_close - 1) * 100, 3),
        "volume": int(df["volume"].iloc[-1]),
        "ma5": round(float(ma5), 2), "ma20": round(float(ma20), 2),
        "ma60": round(float(ma60), 2),
        "rsi14": round(float(rsi), 1),
        "macd_dif": round(float(dif.iloc[-1]), 3),
        "macd_dea": round(float(dea.iloc[-1]), 3),
        "above_ma20": bool(close.iloc[-1] > ma20),
        "data_source": _data_source,
    }


@mcp.tool()
def watchlist_snapshot() -> list[dict]:
    """一次性获取股票池全部股票的行情+关键指标快照,适合决策前总览"""
    out = []
    for s in WATCHLIST:
        ind = get_indicators(s["symbol"])
        ind["name"] = s["name"]
        out.append(ind)
    return out


@mcp.tool()
def candidate_snapshot(symbols: list[str]) -> list[dict]:
    """读取新闻筛选后的动态候选池行情；仅允许配置中的候选宇宙。"""
    normalized = list(dict.fromkeys(str(symbol).strip() for symbol in symbols))
    invalid = [symbol for symbol in normalized if symbol not in UNIVERSE]
    if invalid:
        raise ValueError(f"候选池包含未批准A股代码:{', '.join(invalid)}")
    out = []
    for symbol in normalized:
        item = get_indicators(symbol)
        item["name"] = UNIVERSE[symbol]
        out.append(item)
    return out


def _clean_number(value):
    try:
        number = float(value)
        return number if pd.notna(number) else None
    except (TypeError, ValueError):
        return None


@mcp.tool()
def get_fundamentals(symbol: str) -> dict:
    """查询单只A股最新基本面摘要:收入/利润增长、ROE、毛利率、负债率及估值。
    离线模式返回明确标注的参考数据；实时失败时返回 unavailable，不伪造财报。"""
    symbol = symbol.strip()
    if symbol not in UNIVERSE:
        return {"ok": False, "error": f"{symbol} 不在A股候选宇宙内"}
    if _HAS_AKSHARE and not _FORCE_OFFLINE:
        try:
            start_year = str(datetime.now().year - 3)
            from agent.trading_clock import bounded_requests
            with bounded_requests(seconds=8):
                df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=start_year)
            if len(df):
                row = df.sort_values("日期").iloc[-1]
                return {
                    "symbol": symbol, "report_date": str(row.get("日期")),
                    "revenue_growth_pct": _clean_number(row.get("主营业务收入增长率(%)")),
                    "net_income_growth_pct": _clean_number(row.get("净利润增长率(%)")),
                    "roe_pct": _clean_number(row.get("净资产收益率(%)")),
                    "gross_margin_pct": _clean_number(row.get("销售毛利率(%)")),
                    "debt_to_equity_pct": _clean_number(row.get("负债与所有者权益比率(%)")),
                    "pe_ttm": None, "price_to_book": None, "free_cash_flow": None,
                    "data_source": "akshare(真实财务指标)",
                }
        except Exception as exc:
            return {"ok": False, "symbol": symbol,
                    "error": f"akshare 获取基本面失败:{exc}",
                    "data_source": "unavailable(实时基本面不可用)"}
    if _FORCE_OFFLINE:
        return mock_fundamentals(symbol)
    return {"ok": False, "symbol": symbol,
            "error": "akshare 未安装，实时基本面不可用",
            "data_source": "unavailable(实时基本面不可用)"}


@mcp.tool()
def watchlist_fundamentals() -> list[dict]:
    """一次性获取A股股票池全部标的的基本面摘要，供基本面分析角色比较。"""
    return [get_fundamentals(item["symbol"]) for item in WATCHLIST]


@mcp.tool()
def candidate_fundamentals(symbols: list[str]) -> list[dict]:
    """读取动态候选池全部标的的基本面摘要。"""
    normalized = list(dict.fromkeys(str(symbol).strip() for symbol in symbols))
    invalid = [symbol for symbol in normalized if symbol not in UNIVERSE]
    if invalid:
        raise ValueError(f"候选池包含未批准A股代码:{', '.join(invalid)}")
    return [get_fundamentals(symbol) for symbol in normalized]


@mcp.tool()
def trading_sessions() -> dict:
    """宿主专用：真实交易日历，不通过星期推算休市日。"""
    from agent.trading_clock import calendar
    return calendar()


@mcp.tool()
def execution_session_quote(symbol: str, day: str) -> dict:
    """宿主专用：指定已收盘交易日的不复权执行行情；禁止未来报价。"""
    if _FORCE_OFFLINE:
        raise ValueError("离线模式禁止真实执行队列")
    from agent.trading_clock import session_quote
    return session_quote(symbol, day)


if __name__ == "__main__":
    mcp.run()  # stdio 模式
