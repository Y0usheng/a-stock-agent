# -*- coding: utf-8 -*-
"""可复现的离线行情生成器。"""
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd

SYNTHETIC_START_DATE = "2020-01-01"


def _seed_for_symbol(symbol: str) -> int:
    return int.from_bytes(hashlib.sha256(symbol.encode("utf-8")).digest()[:4], "big")


def random_walk_history(symbol: str, days: int, base_prices: dict[str, float],
                        drift: float, volatility: float,
                        volume_low: int, volume_high: int,
                        high_low_sigma: float) -> pd.DataFrame:
    """生成跨进程一致的 OHLCV 随机游走，用于无网络演示和测试。"""
    if not isinstance(days, int) or isinstance(days, bool) or days <= 0:
        raise ValueError("days 必须是正整数")
    # 从固定日期生成完整序列再截取，保证不同查询窗口的重叠日期完全一致。
    dates = pd.bdate_range(start=SYNTHETIC_START_DATE, end=datetime.now().date())
    sample_size = len(dates)
    rng = np.random.default_rng(_seed_for_symbol(symbol))
    base = base_prices.get(symbol, 100.0)
    returns = rng.normal(drift, volatility, sample_size)
    close = base * pd.Series(returns).add(1).cumprod()
    frame = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": (close * (1 + rng.normal(0, 0.004, sample_size))).round(2),
        "high": (close * (1 + abs(rng.normal(0, high_low_sigma, sample_size)))).round(2),
        "low": (close * (1 - abs(rng.normal(0, high_low_sigma, sample_size)))).round(2),
        "close": close.round(2),
        "volume": rng.integers(volume_low, volume_high, sample_size),
    })
    return frame.tail(days).reset_index(drop=True)
