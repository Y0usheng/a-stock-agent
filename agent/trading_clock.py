"""Observed information time and precommitted next-session-close execution."""
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
from pathlib import Path
from unittest.mock import patch

from config import DATA_DIR
from evals.experiments import canonical_hash
from servers.data_quality import event_time, market_close, market_zone

VERSION = "after-close-next-session-v1"
PRICE_DEVIATION = .02


def now():
    return datetime.now(market_zone("cn"))


@contextmanager
def bounded_requests(seconds=15):
    import requests
    request = requests.sessions.Session.request
    def bounded(session, *args, **kwargs):
        kwargs["timeout"] = (5, max(1, seconds))
        return request(session, *args, **kwargs)
    with patch.object(requests.sessions.Session, "request", bounded):
        yield


def calendar(path=None):
    path = Path(path or DATA_DIR / "calendar" / "cn-sessions.json")
    cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if cached and canonical_hash(cached["sessions"]) != cached.get("fingerprint"):
        raise ValueError("交易日历指纹不符")
    if cached and (now() - event_time(cached["fetched_at"], "cn")).total_seconds() < 86400:
        return cached
    try:
        import akshare as ak
        with bounded_requests():
            frame = ak.tool_trade_date_hist_sina()
        days = sorted(set(frame["trade_date"].astype(str)))
        if len(days) < 1000 or days[-1] < now().date().isoformat():
            raise ValueError("日历没有覆盖当前日期")
        value = {"sessions": days, "fingerprint": canonical_hash(days), "fetched_at": now().isoformat(),
                 "source": "Sina exchange-session calendar via AKShare", "quality": "provider_calendar"}
        from agent.analysis import write_log
        write_log(path, value)
        return value
    except Exception:
        # A bounded stale cache can cover a short outage; never guess weekdays as sessions.
        if cached and (now() - event_time(cached["fetched_at"], "cn")).total_seconds() <= 7 * 86400:
            return {**cached, "cache_warning": "calendar_refresh_failed"}
        raise RuntimeError("真实交易日历不可用，停止安排模拟成交")


def make_timing(price_day, sessions, cutoff=None, calendar_hash=None):
    cutoff = event_time(cutoff, "cn") if cutoff else now()
    if cutoff > now() + timedelta(seconds=5) or cutoff < market_close(price_day, "cn"):
        raise ValueError("信息截止必须在真实行情收盘之后且不能位于未来")
    if price_day not in sessions:
        raise ValueError("行情日不在真实交易日历内")
    completed = [day for day in sessions if market_close(day, "cn") <= cutoff]
    if not completed or max(completed) != price_day:
        raise ValueError("行情不是决策时最近已收盘的交易日")
    future = [day for day in sessions if market_close(day, "cn") > cutoff]
    if not future:
        raise ValueError("交易日历没有覆盖下一交易日，不推算节假日")
    execution_day = min(future)
    return {"version": VERSION, "price_as_of_date": price_day, "information_cutoff_at": cutoff.isoformat(),
        "execution_session": execution_day, "execution_at": market_close(execution_day, "cn").isoformat(),
        "approval_deadline": (market_close(execution_day, "cn") - timedelta(minutes=5)).isoformat(),
        "order_type": "precommitted_next_session_close_limit", "max_price_deviation_pct": PRICE_DEVIATION,
        "calendar_fingerprint": calendar_hash,
        "fill_model": "observed_unadjusted_close_plus_slippage; ex_post_daily_volume_guard; not_actual_auction_fills"}


def session_quote(symbol, day):
    from config import MARKETS
    if symbol not in {row["symbol"] for row in MARKETS["cn"]["candidate_universe"]}:
        raise ValueError("不支持的执行股票")
    if now() < market_close(day, "cn"):
        raise ValueError("执行交易日尚未收盘")
    import akshare as ak
    with bounded_requests():
        frame = ak.stock_zh_a_daily(symbol=("sh" if symbol.startswith("6") else "sz") + symbol,
                                   start_date=day.replace("-", ""), end_date=day.replace("-", ""), adjust="")
    rows = frame[frame["date"].astype(str) == day]
    if len(rows) != 1:
        raise ValueError("执行日真实不复权行情缺失，不滚动到其他日期成交")
    row = rows.iloc[0]
    result = {"symbol": symbol, "date": day, "open": float(row["open"]), "close": float(row["close"]),
        "high": float(row["high"]), "low": float(row["low"]), "volume": int(row["volume"]),
        "price_basis": "unadjusted", "data_quality": "live_prices", "source": "AKShare/Sina raw daily session",
        "observed_at": now().isoformat()}
    result["fingerprint"] = canonical_hash({k: v for k, v in result.items() if k != "observed_at"})
    return result
