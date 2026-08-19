# -*- coding: utf-8 -*-
"""
美股数据 MCP Server
===================
数据源:yfinance(Yahoo Finance)。只有显式 ASTOCK_OFFLINE=1 才生成离线模拟数据；
实时模式网络不可用或依赖缺失时直接停止，不静默造数。
工具接口与 A股 market_server 完全一致(get_quote / get_history / get_indicators /
watchlist_snapshot),因此决策循环无需改动就能复用——这正是 MCP 抽象的好处。
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
from servers.fundamentals_data import mock_fundamentals

WATCHLIST = MARKETS["us"]["watchlist"]
UNIVERSE = {item["symbol"]: item["name"] for item in MARKETS["us"]["candidate_universe"]}

if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

try:
    import yfinance as yf
    _HAS_YF = True
except Exception:
    _HAS_YF = False

mcp = FastMCP("usstock-market", log_level="ERROR")

_cache: dict[str, pd.DataFrame] = {}
_data_source = "unknown"
_FORCE_OFFLINE = os.environ.get("ASTOCK_OFFLINE", "").lower() in {"1", "true", "yes"}


def _mock_history(symbol: str, days: int = 120) -> pd.DataFrame:
    return random_walk_history(
        symbol, days,
        {"AAPL": 230, "MSFT": 500, "NVDA": 175,
         "TSLA": 320, "GOOGL": 200, "AMZN": 225},
        drift=0.0005, volatility=0.02,
        volume_low=5_000_000, volume_high=120_000_000, high_low_sigma=0.009)


def _fetch_history(symbol: str, days: int = 120) -> pd.DataFrame:
    global _data_source
    key = f"{symbol}:{days}"
    if key in _cache:
        return _cache[key]
    if _FORCE_OFFLINE:
        df = _mock_history(symbol, days)
        _cache[key] = df
        _data_source = "mock(显式离线模拟数据,仅供工程回归)"
        return df
    if _HAS_YF:
        try:
            # days 是交易日数量，按约 1.8 倍自然日留足周末和节假日。
            start = (datetime.now() - timedelta(days=max(days * 2, 365))).strftime("%Y-%m-%d")
            df = yf.Ticker(symbol).history(start=start, auto_adjust=True)
            if len(df) > 0:
                df = df.reset_index()
                df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
                df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                        "Close": "close", "Volume": "volume"})
                df = df[["date", "open", "high", "low", "close", "volume"]].round(2)
                df = df.tail(days).reset_index(drop=True)
                _cache[key] = df
                _data_source = "yfinance(真实数据)"
                return df
        except Exception as e:
            raise RuntimeError(f"yfinance 获取 {symbol} 行情失败，实时模式已停止") from e
    raise RuntimeError("yfinance 未安装，实时模式禁止使用模拟行情；请安装依赖或使用 --offline")


@mcp.tool()
def get_quote(symbol: str) -> dict:
    """查询单只美股最新行情(收盘价、涨跌幅、成交量)。symbol 为代码,如 AAPL"""
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
    """查询美股日K线历史(复权)。返回最近 days 个交易日的 OHLCV"""
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
    return {
        "symbol": symbol,
        "date": str(df["date"].iloc[-1]),
        "close": float(close.iloc[-1]),
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
    """一次性获取美股股票池全部股票的行情+关键指标快照,适合决策前总览"""
    out = []
    for s in WATCHLIST:
        ind = get_indicators(s["symbol"])
        ind["name"] = s["name"]
        out.append(ind)
    return out


@mcp.tool()
def candidate_snapshot(symbols: list[str]) -> list[dict]:
    """读取新闻筛选后的动态候选池行情；仅允许配置中的候选宇宙。"""
    normalized = list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols))
    invalid = [symbol for symbol in normalized if symbol not in UNIVERSE]
    if invalid:
        raise ValueError(f"候选池包含未批准美股代码:{', '.join(invalid)}")
    out = []
    for symbol in normalized:
        item = get_indicators(symbol)
        item["name"] = UNIVERSE[symbol]
        out.append(item)
    return out


def _clean_number(value, scale: float = 1.0):
    try:
        number = float(value) * scale
        return number if pd.notna(number) else None
    except (TypeError, ValueError):
        return None


@mcp.tool()
def get_fundamentals(symbol: str) -> dict:
    """查询单只美股最新基本面摘要:收入/利润增长、ROE、毛利率、负债及估值。
    离线模式返回明确标注的参考数据；实时失败时返回 unavailable，不伪造财报。"""
    symbol = symbol.strip().upper()
    if symbol not in UNIVERSE:
        return {"ok": False, "error": f"{symbol} 不在美股候选宇宙内"}
    if _HAS_YF and not _FORCE_OFFLINE:
        try:
            info = yf.Ticker(symbol).info
            if info:
                return {
                    "symbol": symbol, "report_date": datetime.now().date().isoformat(),
                    "revenue_growth_pct": _clean_number(info.get("revenueGrowth"), 100),
                    "net_income_growth_pct": _clean_number(info.get("earningsGrowth"), 100),
                    "roe_pct": _clean_number(info.get("returnOnEquity"), 100),
                    "gross_margin_pct": _clean_number(info.get("grossMargins"), 100),
                    "debt_to_equity_pct": _clean_number(info.get("debtToEquity")),
                    "pe_ttm": _clean_number(info.get("trailingPE")),
                    "price_to_book": _clean_number(info.get("priceToBook")),
                    "free_cash_flow": _clean_number(info.get("freeCashflow")),
                    "data_source": "yfinance(真实基本面)",
                }
        except Exception as exc:
            return {"ok": False, "symbol": symbol,
                    "error": f"yfinance 获取基本面失败:{exc}",
                    "data_source": "unavailable(实时基本面不可用)"}
    if _FORCE_OFFLINE:
        return mock_fundamentals(symbol)
    return {"ok": False, "symbol": symbol,
            "error": "yfinance 未安装，实时基本面不可用",
            "data_source": "unavailable(实时基本面不可用)"}


@mcp.tool()
def watchlist_fundamentals() -> list[dict]:
    """一次性获取美股股票池全部标的的基本面摘要，供基本面分析角色比较。"""
    return [get_fundamentals(item["symbol"]) for item in WATCHLIST]


@mcp.tool()
def candidate_fundamentals(symbols: list[str]) -> list[dict]:
    """读取动态候选池全部标的的基本面摘要。"""
    normalized = list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols))
    invalid = [symbol for symbol in normalized if symbol not in UNIVERSE]
    if invalid:
        raise ValueError(f"候选池包含未批准美股代码:{', '.join(invalid)}")
    return [get_fundamentals(symbol) for symbol in normalized]


if __name__ == "__main__":
    mcp.run()
