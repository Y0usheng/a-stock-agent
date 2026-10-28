"""排除未收盘日线并验证各标的同一行情截面。"""
from datetime import datetime, time, date, timedelta
from zoneinfo import ZoneInfo
import pandas as pd


def market_zone(market: str):
    return ZoneInfo("Asia/Shanghai" if market == "cn" else "America/New_York")


def market_close(day: str, market: str) -> datetime:
    return datetime.combine(date.fromisoformat(day), time(15, 0) if market == "cn" else time(16, 0),
                            tzinfo=market_zone(market))


def event_time(published_at: str, market: str) -> datetime:
    text = str(published_at).replace("Z", "+00:00")
    # 只有日期时按当地日末处理，避免把未知时刻的公告用于当天收盘信号。
    value = (datetime.combine(date.fromisoformat(text), time(23, 59, 59)) if len(text) == 10
             else datetime.fromisoformat(text))
    return value.replace(tzinfo=market_zone(market)) if value.tzinfo is None else value.astimezone(market_zone(market))


def event_available_date(published_at: str, market: str) -> date:
    timestamp = event_time(published_at, market)
    day = timestamp.date()
    return day if timestamp <= market_close(day.isoformat(), market) else day + timedelta(days=1)


def completed_history(frame: pd.DataFrame, market: str, now: datetime | None = None) -> pd.DataFrame:
    zone = market_zone(market)
    local = (now or datetime.now(zone)).astimezone(zone)
    today = local.date().isoformat()
    close = time(15, 0) if market == "cn" else time(16, 0)
    include_today = local.weekday() < 5 and local.time().replace(tzinfo=None) >= close
    dates = frame["date"].astype(str)
    return frame[(dates <= today) if include_today else (dates < today)].reset_index(drop=True)


def snapshot_date(snapshot: list[dict]) -> str:
    dates = {item.get("date") for item in snapshot}
    if not snapshot or None in dates or "" in dates or len(dates) != 1:
        raise RuntimeError("行情日期缺失或不一致，停止本轮")
    value = dates.pop()
    datetime.strptime(value, "%Y-%m-%d")
    return value
