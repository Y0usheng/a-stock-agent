"""历史价格读取、严格校验与可重放数据指纹。"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import MARKETS
from evals.experiments import canonical_hash
from servers.synthetic_data import random_walk_history


def normalize_history(symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "close", "volume"}
    if not required <= set(frame.columns):
        raise ValueError(f"{symbol} 缺少行情字段: {sorted(required - set(frame.columns))}")
    clean = frame[[key for key in ("date", "open", "high", "low", "close", "volume")
                   if key in frame.columns]].copy()
    clean["date"] = [date.fromisoformat(str(value)).isoformat() for value in clean["date"]]
    if clean["date"].duplicated().any():
        raise ValueError(f"{symbol} 存在重复行情日期")
    for key in clean.columns.drop("date"):
        clean[key] = pd.to_numeric(clean[key], errors="raise")
        if not np.isfinite(clean[key]).all():
            raise ValueError(f"{symbol} {key} 含缺失或无穷值")
        if key == "volume":
            if (clean[key] < 0).any() or (clean[key] % 1 != 0).any():
                raise ValueError(f"{symbol} volume 必须为非负整数股数")
            clean[key] = clean[key].astype("int64")
        elif (clean[key] <= 0).any():
            raise ValueError(f"{symbol} {key} 必须为正数")
        else:
            clean[key] = clean[key].astype(float)
    return clean.sort_values("date").reset_index(drop=True)


def load_price_dataset(market: str, source: str = "synthetic", *, end_date: str | None = None,
                       path: Path | None = None, window: int = 90) -> tuple[dict, dict]:
    end = date.fromisoformat(end_date) if end_date else date.today()
    if end > date.today():
        raise ValueError("行情截至日期不得在未来")
    universe = {item["symbol"] for item in MARKETS[market]["candidate_universe"]}
    histories = {}
    basis = "synthetic"
    quality = "synthetic_prices"
    if source == "file":
        if path is None:
            raise ValueError("file 模式需要 --prices-file")
        if path.is_dir():
            path = path / f"{market}.csv"
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, dtype={"symbol": str, "date": str})
            if "market" in frame:
                frame = frame[frame["market"] == market]
            if "symbol" not in frame:
                raise ValueError("CSV 需要 symbol 列")
            histories = {symbol: rows.drop(columns=["symbol", "market"], errors="ignore")
                         for symbol, rows in frame.groupby("symbol")}
            basis = "unspecified"
            quality = "imported_prices_unverified"
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("market") != market:
                raise ValueError("价格文件市场不匹配")
            histories = {symbol: pd.DataFrame(rows)
                         for symbol, rows in payload["histories"].items()}
            basis = payload.get("manifest", {}).get("price_basis", "unspecified")
            # 文件声明仅是来源记录，不把未验证的文件自动标为真实价格。
            quality = "replayed_prices" if payload.get("manifest", {}).get("fingerprint") else "imported_prices_unverified"
    elif source == "live":
        from servers.data_quality import completed_history
        start = end - timedelta(days=(window + 90) * 3)
        for symbol in sorted(universe):
            if market == "cn":
                import akshare as ak
                frame = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq",
                                          start_date=start.strftime("%Y%m%d"),
                                          end_date=end.strftime("%Y%m%d"), timeout=30)
                frame = frame.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                                               "最低": "low", "收盘": "close", "成交量": "volume"})
                frame["date"] = frame["date"].astype(str)
                frame["volume"] = frame["volume"] * 100  # 东财日线单位为手
            else:
                import yfinance as yf
                frame = yf.Ticker(symbol).history(start=start.isoformat(),
                    end=(end + timedelta(days=1)).isoformat(), auto_adjust=True, timeout=30)
                frame = frame.reset_index().rename(columns={"Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume"})
                frame["date"] = frame["Date"].dt.strftime("%Y-%m-%d")
            histories[symbol] = completed_history(frame, market)
        quality = "historical_prices"
        basis = "qfq" if market == "cn" else "yahoo_auto_adjust"
    elif source == "synthetic":
        for symbol in sorted(universe):
            histories[symbol] = random_walk_history(symbol, window + 100, {},
                drift=0.0003, volatility=0.018, volume_low=2_000_000,
                volume_high=80_000_000, high_low_sigma=0.008, end_date=end.isoformat())
    else:
        raise ValueError(f"未知价格来源: {source}")
    if not histories or set(histories) - universe:
        raise ValueError("行情为空或含受控股票宇宙之外的代码")
    if source == "file" and path.suffix.lower() != ".csv" and payload.get("manifest", {}).get("fingerprint"):
        full = {symbol: normalize_history(symbol, frame).to_dict("records")
                for symbol, frame in sorted(histories.items())}
        if canonical_hash({"market": market, "prices": full, "basis": basis}) != payload["manifest"]["fingerprint"]:
            raise ValueError("价格文件内容与保存的指纹不符，请核查数据；没有使用该文件")
    histories = {symbol: normalize_history(symbol, frame[frame["date"].astype(str) <= end.isoformat()])
                 for symbol, frame in histories.items()}
    common = sorted(set.intersection(*(set(frame["date"]) for frame in histories.values())))
    if len(common) < window + 26:
        raise ValueError(f"共同历史仅 {len(common)} 日，至少需要 {window + 26} 日")
    # 只保留实际回测与60日指标预热用到的内容，路径不参与指纹。
    cutoff = common[-min(len(common), window + 60)]
    histories = {symbol: frame[frame["date"] >= cutoff].reset_index(drop=True)
                 for symbol, frame in histories.items()}
    content = {symbol: frame.to_dict("records") for symbol, frame in sorted(histories.items())}
    manifest = {"schema_version": "price-dataset-v1", "market": market, "source": source,
                "data_quality": quality, "price_basis": basis, "volume_unit": "shares",
                "start_date": cutoff, "end_date": common[-1], "symbols": sorted(histories),
                "row_count": sum(len(frame) for frame in histories.values()),
                "fingerprint": canonical_hash({"market": market, "prices": content, "basis": basis}),
                "limitations": "复权价格按标准化持仓单位模拟，未逐笔重建分红/拆股；当前股票宇宙有存活者偏差。"}
    return histories, manifest


def export_price_dataset(path: Path, market: str, histories: dict, manifest: dict):
    from agent.analysis import write_log
    write_log(path, {"market": market, "manifest": manifest,
                     "histories": {symbol: frame.to_dict("records") for symbol, frame in histories.items()}})
