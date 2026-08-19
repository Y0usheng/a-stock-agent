# -*- coding: utf-8 -*-
"""离线基本面快照。

这些数值只用于无网络教学演示，不代表当前真实财务数据。真实运行优先调用
AkShare/yfinance；日志会明确标注 data_source，避免和真实数据混淆。
"""

MOCK_FUNDAMENTALS = {
    "600519": {"revenue_growth_pct": 15.7, "net_income_growth_pct": 16.1,
               "roe_pct": 35.2, "gross_margin_pct": 91.8,
               "debt_to_equity_pct": 28.0, "pe_ttm": 28.0, "price_to_book": 9.5},
    "300750": {"revenue_growth_pct": 22.0, "net_income_growth_pct": 31.0,
               "roe_pct": 24.0, "gross_margin_pct": 25.0,
               "debt_to_equity_pct": 85.0, "pe_ttm": 24.0, "price_to_book": 5.0},
    "600036": {"revenue_growth_pct": 3.5, "net_income_growth_pct": 6.2,
               "roe_pct": 15.5, "gross_margin_pct": None,
               "debt_to_equity_pct": 1150.0, "pe_ttm": 7.0, "price_to_book": 1.1},
    "002594": {"revenue_growth_pct": 29.0, "net_income_growth_pct": 34.0,
               "roe_pct": 25.0, "gross_margin_pct": 20.5,
               "debt_to_equity_pct": 145.0, "pe_ttm": 25.0, "price_to_book": 5.5},
    "601899": {"revenue_growth_pct": 11.0, "net_income_growth_pct": 20.0,
               "roe_pct": 22.0, "gross_margin_pct": 18.0,
               "debt_to_equity_pct": 72.0, "pe_ttm": 15.0, "price_to_book": 3.0},
    "600900": {"revenue_growth_pct": 8.0, "net_income_growth_pct": 12.0,
               "roe_pct": 16.0, "gross_margin_pct": 48.0,
               "debt_to_equity_pct": 95.0, "pe_ttm": 19.0, "price_to_book": 3.0},
    "AAPL": {"revenue_growth_pct": 6.0, "net_income_growth_pct": 8.0,
             "roe_pct": 150.0, "gross_margin_pct": 46.0,
             "debt_to_equity_pct": 145.0, "pe_ttm": 31.0, "price_to_book": 45.0},
    "MSFT": {"revenue_growth_pct": 16.0, "net_income_growth_pct": 18.0,
             "roe_pct": 34.0, "gross_margin_pct": 69.0,
             "debt_to_equity_pct": 42.0, "pe_ttm": 35.0, "price_to_book": 12.0},
    "NVDA": {"revenue_growth_pct": 55.0, "net_income_growth_pct": 65.0,
             "roe_pct": 80.0, "gross_margin_pct": 73.0,
             "debt_to_equity_pct": 18.0, "pe_ttm": 42.0, "price_to_book": 35.0},
    "TSLA": {"revenue_growth_pct": 8.0, "net_income_growth_pct": -12.0,
             "roe_pct": 11.0, "gross_margin_pct": 18.0,
             "debt_to_equity_pct": 20.0, "pe_ttm": 85.0, "price_to_book": 12.0},
    "GOOGL": {"revenue_growth_pct": 14.0, "net_income_growth_pct": 20.0,
              "roe_pct": 30.0, "gross_margin_pct": 58.0,
              "debt_to_equity_pct": 10.0, "pe_ttm": 25.0, "price_to_book": 8.0},
    "AMZN": {"revenue_growth_pct": 11.0, "net_income_growth_pct": 35.0,
             "roe_pct": 24.0, "gross_margin_pct": 49.0,
             "debt_to_equity_pct": 45.0, "pe_ttm": 38.0, "price_to_book": 8.0},
}


def mock_fundamentals(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "report_date": "offline-reference",
        **MOCK_FUNDAMENTALS.get(symbol, {}),
        "free_cash_flow": None,
        "data_source": "mock(离线基本面参考,非实时财报)",
    }
