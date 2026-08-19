# -*- coding: utf-8 -*-
"""评分所用的纯函数，供命令行、仪表盘和测试共同复用。"""

MOVE_THRESHOLD = 0.005
EVALUATION_HISTORY_DAYS = 1500


def actual_direction_from_frame(df, start: str, horizon: int,
                                threshold: float = MOVE_THRESHOLD) -> str | None:
    """返回 horizon 个交易日后的方向；样本未完整到期时返回 None。"""
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
        raise ValueError("horizon 必须是正整数交易日")
    dates = df["date"].astype(str).tolist()
    if start not in dates:
        return None
    start_index = dates.index(start)
    target_index = start_index + horizon
    if target_index >= len(df):
        return None
    change = float(df["close"].iloc[target_index]) / float(df["close"].iloc[start_index]) - 1
    if change > threshold:
        return "up"
    if change < -threshold:
        return "down"
    return "flat"


def prediction_start_date(log: dict) -> str | None:
    """新日志优先使用行情截面日期，旧日志兼容 date 字段。"""
    return log.get("as_of_date") or log.get("date")


def uses_synthetic_data(log: dict) -> bool:
    """判断日志是否由离线/自动降级模拟行情生成。"""
    if log.get("offline"):
        return True
    return any(str(source).startswith("mock") for source in log.get("data_sources", []))
