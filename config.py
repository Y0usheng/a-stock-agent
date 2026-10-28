# -*- coding: utf-8 -*-
"""全局配置(双市场:A股 cn / 美股 us)"""
import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = Path(os.environ.get("ASTOCK_DATA_DIR", ROOT / "data"))
NEWS_DB = Path(os.environ.get("ASTOCK_NEWS_DB", DATA_DIR / "news_events.db"))
EXPERIMENT_DB = Path(os.environ.get("ASTOCK_EXPERIMENT_DB", DATA_DIR / "experiments.db"))

MAX_POSITION_PCT = 0.30      # 单只股票最大仓位占总资产比例(两市场通用)

MARKETS = {
    "cn": {
        "name": "A股",
        "currency": "CNY",
        "watchlist": [
            {"symbol": "600519", "name": "贵州茅台"},
            {"symbol": "300750", "name": "宁德时代"},
            {"symbol": "600036", "name": "招商银行"},
            {"symbol": "002594", "name": "比亚迪"},
            {"symbol": "601899", "name": "紫金矿业"},
            {"symbol": "600900", "name": "长江电力"},
        ],
        "candidate_universe": [
            {"symbol": "600519", "name": "贵州茅台"},
            {"symbol": "300750", "name": "宁德时代"},
            {"symbol": "600036", "name": "招商银行"},
            {"symbol": "002594", "name": "比亚迪"},
            {"symbol": "601899", "name": "紫金矿业"},
            {"symbol": "600900", "name": "长江电力"},
            {"symbol": "601318", "name": "中国平安"},
            {"symbol": "600030", "name": "中信证券"},
            {"symbol": "601138", "name": "工业富联"},
            {"symbol": "002475", "name": "立讯精密"},
            {"symbol": "600276", "name": "恒瑞医药"},
            {"symbol": "300308", "name": "中际旭创"},
        ],
        "candidate_limit": 8,
        "investment_watchlist": ["300750", "600036", "600519"],
        "initial_cash": 1_000_000.0,
        "lot_size": 100,          # 一手100股
        "t_plus_1": True,         # 当日买入不可当日卖出
        "commission_rate": 0.00025,
        "commission_min": 5.0,
        "stamp_tax_rate": 0.0005,  # 卖出印花税
        "slippage_bps": 5.0,       # 模拟成交相对快照价的单边滑点
        "max_volume_participation_pct": 0.05,  # 单笔最多占当日成交量5%
        "db": DATA_DIR / "account_cn.db",
        "market_server": ROOT / "servers" / "market_server.py",
        "news_server": ROOT / "servers" / "news_server.py",
    },
    "us": {
        "name": "美股",
        "currency": "USD",
        "watchlist": [
            {"symbol": "AAPL", "name": "Apple"},
            {"symbol": "MSFT", "name": "Microsoft"},
            {"symbol": "NVDA", "name": "NVIDIA"},
            {"symbol": "TSLA", "name": "Tesla"},
            {"symbol": "GOOGL", "name": "Alphabet"},
            {"symbol": "AMZN", "name": "Amazon"},
        ],
        "candidate_universe": [
            {"symbol": "AAPL", "name": "Apple"},
            {"symbol": "MSFT", "name": "Microsoft"},
            {"symbol": "NVDA", "name": "NVIDIA"},
            {"symbol": "TSLA", "name": "Tesla"},
            {"symbol": "GOOGL", "name": "Alphabet"},
            {"symbol": "AMZN", "name": "Amazon"},
            {"symbol": "META", "name": "Meta Platforms"},
            {"symbol": "AVGO", "name": "Broadcom"},
            {"symbol": "AMD", "name": "AMD"},
            {"symbol": "NFLX", "name": "Netflix"},
            {"symbol": "JPM", "name": "JPMorgan Chase"},
            {"symbol": "XOM", "name": "Exxon Mobil"},
        ],
        "candidate_limit": 8,
        "initial_cash": 150_000.0,
        "lot_size": 1,            # 可买1股
        "t_plus_1": False,        # T+0
        "commission_rate": 0.0,   # 主流券商零佣金
        "commission_min": 0.0,
        "stamp_tax_rate": 0.0,
        "slippage_bps": 3.0,
        "max_volume_participation_pct": 0.02,
        "db": DATA_DIR / "account_us.db",
        "market_server": ROOT / "servers" / "us_market_server.py",
        "news_server": ROOT / "servers" / "news_server.py",
    },
}

LOG_DIR = Path(os.environ.get("ASTOCK_LOG_DIR", ROOT / "logs" / "decisions"))

# 四个角色统一使用 OpenAI Responses API；共享一个模型配置。
DEFAULT_OPENAI_MODEL = "gpt-6.1-sol"
DEFAULT_MANAGER_PROVIDER = "openai"
DEFAULT_RISK_PROVIDER = "openai"
DEFAULT_PROMPT_VERSION = "manager-v1"
