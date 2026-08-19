# -*- coding: utf-8 -*-
"""
预测评分(evals 雏形,双市场)
==============================
读取 logs/decisions/*.json 里的 predictions,对已到期(horizon_days 已过)的预测
对比实际涨跌,按市场分别输出方向命中率。这就是 agent 的"考试成绩单"——
以后改 prompt/换模型,跑同一批历史数据对比分数,就是最朴素的 eval 闭环。

用法: python evals/score.py
"""
import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EXPERIMENT_DB, LOG_DIR
from evals.calibration import chronological_calibration
from evals.experiments import (canonical_hash, decision_log_variant,
                               record_experiment)
from evals.metrics import (EVALUATION_HISTORY_DAYS,
                           actual_direction_from_frame,
                           prediction_start_date,
                           uses_synthetic_data)

sys.path.insert(0, str(Path(__file__).parent.parent / "servers"))
import market_server        # A股数据层
import us_market_server     # 美股数据层

DATA = {"cn": market_server, "us": us_market_server}


def actual_direction(market: str, symbol: str, start: str, horizon: int,
                     synthetic: bool = False) -> str | None:
    fetch = DATA[market]._mock_history if synthetic else DATA[market]._fetch_history
    df = fetch(symbol, EVALUATION_HISTORY_DAYS)
    return actual_direction_from_frame(df, start, horizon)


def main():
    parser = argparse.ArgumentParser(description="评分预测并可登记 Prompt/模型实验")
    parser.add_argument("--record-experiment", action="store_true")
    parser.add_argument("--experiment-db", type=Path, default=EXPERIMENT_DB)
    args = parser.parse_args()
    logs = sorted(LOG_DIR.glob("*.json"))
    if not logs:
        print("暂无决策日志,先运行 agent/run_daily.py")
        return
    stats = {"cn": [0, 0], "us": [0, 0]}  # market -> [total, hit]
    pending = 0
    rows = []
    for f in logs:
        data = json.loads(f.read_text(encoding="utf-8"))
        market = data.get("market")
        start = prediction_start_date(data)
        if market not in DATA or not start:
            continue
        for p in data.get("predictions", []):
            act = actual_direction(market, p["symbol"], start,
                                   p.get("horizon_days", 5),
                                   synthetic=uses_synthetic_data(data))
            if act is None:
                pending += 1
                continue
            ok = act == p["direction"]
            stats[market][0] += 1
            stats[market][1] += ok
            rows.append({
                "date": start, "market": market, "symbol": p["symbol"],
                "pred": p["direction"], "actual": act, "correct": ok,
                "mark": "✓" if ok else "✗", "confidence": p.get("confidence"),
                "horizon_days": p.get("horizon_days", 5),
                "variant": decision_log_variant(data),
            })
    print(f"{'日期':12s}{'市场':4s}{'代码':10s}{'预测':6s}{'实际':6s}{'命中':4s}置信度")
    for r in rows:
        print(f"{r['date']:12s}{r['market']:4s}{r['symbol']:10s}{r['pred']:6s}"
              f"{r['actual']:6s}{r['mark']:4s}{r['confidence']}")
    print()
    for mkt, (total, hit) in stats.items():
        if total:
            print(f"[{mkt}] 已到期预测 {total} 条,命中 {hit},方向命中率 {hit/total:.1%}")
    if not any(t for t, _ in stats.values()):
        print("所有预测均未到期,过几个交易日再来评分")
    if pending:
        print(f"未到期: {pending} 条")
    calibration = chronological_calibration(rows)
    if calibration["status"] == "complete":
        print("置信度未来集: 原始 Brier={:.3f}, ECE={:.3f}; 校准后 Brier={:.3f}, ECE={:.3f}".format(
            calibration["raw"]["brier_score"], calibration["raw"]["expected_calibration_error"],
            calibration["calibrated"]["brier_score"],
            calibration["calibrated"]["expected_calibration_error"],
        ))
    else:
        print("置信度校准:", calibration["message"])
    if args.record_experiment:
        grouped = {}
        for row in rows:
            key = row["market"] + ":" + canonical_hash(row["variant"])
            grouped.setdefault(key, []).append(row)
        for group in grouped.values():
            references = [{key: row[key] for key in
                           ("date", "market", "symbol", "actual", "horizon_days")}
                          for row in group]
            references.sort(key=lambda item: (item["date"], item["market"], item["symbol"]))
            dataset = {"schema": "matured-direction-predictions-v1", "items": references}
            dataset["fingerprint"] = canonical_hash(dataset)
            hit = sum(row["correct"] for row in group)
            record_experiment(
                "prompt-model-direction-score", group[0]["market"], dataset,
                group[0]["variant"],
                {"sample_count": len(group), "hit_count": hit,
                 "accuracy": round(hit / len(group), 6),
                 "calibration": chronological_calibration(group)},
                db_path=args.experiment_db,
                notes="只有 dataset_fingerprint 相同的不同 variant 才能直接做 A/B。",
            )
        print(f"已登记 {len(grouped)} 个 Prompt/模型变体实验")


if __name__ == "__main__":
    main()
