"""Past-only feedback and validated confidence parameters for the paper agent."""
from copy import deepcopy
import json
from pathlib import Path

from config import DATA_DIR, LOG_DIR
from evals.calibration import apply_calibrator, calibration_metrics, fit_histogram_calibrator
from evals.experiments import canonical_hash
from evals.metrics import MOVE_THRESHOLD, uses_synthetic_data
from evals.price_data import normalize_history
from servers.data_quality import event_time, market_close

VERSION = "paper-outcome-learning-v1"
HOME = DATA_DIR / "learning"
HORIZON = 252


def family_home(model):
    return HOME / canonical_hash(["cn", model, HORIZON, "long-term-investment-v1"])[:16]


def past_logs(cutoff, root=LOG_DIR, model="gpt-6.1-sol"):
    """Only host-validated formal logs; QA shadows and failed runs never train policy."""
    from agent.evidence import validate_evidence_audit
    from agent.contracts import validate_risk_approval
    rows = []
    for path in sorted((Path(root) / "after_close").glob("*.cn.json"))[-800:]:
        try:
            if path.stat().st_size > 5_000_000:
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            stamp = row.get("timing", {}).get("decision_at")
            if (row.get("market") != "cn" or row.get("execution_mode") != "paper" or uses_synthetic_data(row)
                    or not stamp or event_time(stamp, "cn") >= event_time(cutoff, "cn")
                    or row.get("agent_sdk", {}).get("role_models", {}).get("investment-manager", {}).get("model") != model
                    or canonical_hash(row.get("input_dataset")) != row.get("dataset_fingerprint")):
                continue
            validate_evidence_audit(row)
            validate_risk_approval(row, "cn", {r["symbol"] for r in row["decisions"]})
            rows.append(row)
        except (ValueError, KeyError, TypeError, OSError):
            continue
    return rows


def score(logs, histories, sessions, cutoff):
    """Exact future-session anchor + 252 exchange sessions; missing/immature stays pending."""
    import pandas as pd
    end = event_time(cutoff, "cn")
    closed = [day for day in sorted(set(sessions)) if market_close(day, "cn") <= end]
    prices = {}
    for symbol, rows in histories.items():
        try:
            frame = normalize_history(symbol, pd.DataFrame(rows))
            prices[symbol] = {r["date"]: float(r["close"]) for r in frame.to_dict("records") if market_close(r["date"], "cn") <= end}
        except (ValueError, TypeError, KeyError):
            continue
    outcomes, pending, seen = [], 0, set()
    for log in logs:
        timing = log.get("timing", {})
        start, decided = timing.get("execution_session"), timing.get("decision_at")
        if not decided or not start or event_time(decided, "cn") >= end or market_close(start, "cn") <= event_time(decided, "cn"):
            continue
        for prediction in log.get("predictions", []):
            symbol = prediction.get("symbol")
            confidence = prediction.get("confidence")
            if (prediction.get("horizon_days") != HORIZON or prediction.get("direction") not in {"up", "down", "flat"}
                    or type(confidence) not in {float, int} or not 0 <= confidence <= 1):
                continue
            identity = canonical_hash([canonical_hash(log), symbol])
            if identity in seen:
                continue
            seen.add(identity)
            index = closed.index(start) if start in closed else len(closed)
            if index + HORIZON >= len(closed):
                pending += 1
                continue
            target = closed[index + HORIZON]
            pair = prices.get(symbol, {})
            if start not in pair or target not in pair:
                pending += 1
                continue
            change = pair[target] / pair[start] - 1
            actual = "up" if change > MOVE_THRESHOLD else "down" if change < -MOVE_THRESHOLD else "flat"
            outcomes.append({"id": identity, "symbol": symbol, "date": str(decided)[:10], "decision_at": decided,
                "outcome_date": target, "start_date": start, "confidence": float(confidence),
                "predicted": prediction["direction"], "actual": actual, "correct": actual == prediction["direction"],
                "price_return_pct": round(change * 100, 6), "source_log_hash": canonical_hash(log),
                "price_pair_hash": canonical_hash([symbol, start, pair[start], target, pair[target]])})
    return sorted(outcomes, key=lambda r: (r["decision_at"], r["symbol"])), pending


def propose(rows, incumbent=None, *, minimum_train=30, minimum_test=10, validated_until=None):
    """Purged chronological evaluation: every training label predates held-out decisions."""
    dates = sorted({row["date"] for row in rows})
    if len(dates) < 2:
        return {"status": "insufficient_data", "train_count": 0, "test_count": 0}, incumbent
    split = dates[max(1, min(len(dates)-1, int(len(dates)*.8)))]
    test = [r for r in rows if r["date"] >= split]
    watermark = max(filter(None, [validated_until, incumbent["validated_until"] if incumbent else None]), default=None)
    if watermark:
        test = [r for r in test if r["date"] > watermark]
    first_test = min((r["decision_at"] for r in test), default=None)
    train = [r for r in rows if first_test and market_close(r["outcome_date"], "cn") < event_time(first_test, "cn")]
    report = {"status": "insufficient_data", "train_count": len(train), "test_count": len(test)}
    if len(train) < minimum_train or len(test) < minimum_test:
        return report, incumbent
    candidate = fit_histogram_calibrator(train)
    calibrated = calibration_metrics(apply_calibrator(test, candidate))
    baseline = calibration_metrics(apply_calibrator(test, incumbent["calibrator"]) if incumbent else test)
    report.update(status="rejected", baseline_brier=baseline["brier_score"], candidate_brier=calibrated["brier_score"],
        train_last_outcome=max(r["outcome_date"] for r in train), test_first_decision=first_test,
        minimum_improvement=.01, test_ids_hash=canonical_hash([r["id"] for r in test]))
    if calibrated["brier_score"] + .01 < baseline["brier_score"]:
        report = {**report, "status": "adopted"}
        parameters = {"calibrator": candidate, "validated_until": max(r["outcome_date"] for r in test),
                      "validation": report, "method": "purged_past_only_histogram"}
        parameters["version"] = canonical_hash(parameters)
        return report, parameters
    return report, incumbent


def review(logs, histories, sessions, cutoff, *, home=HOME):
    from agent.analysis import write_log
    home = Path(home)
    latest = home / "latest.json"
    if latest.exists() and event_time(json.loads(latest.read_text(encoding="utf-8"))["information_cutoff_at"], "cn") > event_time(cutoff, "cn"):
        raise ValueError("学习不能回填过去；历史重放须使用独立目录")
    from agent.strategy_evolution import deploy, healthcheck, load
    current = home / "active.json"
    code = healthcheck(home / "strategy", cutoff)
    incumbent = load(home / "strategy", code["version"], cutoff)["parameters"] if code.get("version") else None
    if incumbent and market_close(incumbent["validated_until"], "cn") >= event_time(cutoff, "cn"):
        raise ValueError("学习参数包含截止时点之后的验证结果")
    watermark_path = home / "evaluation-watermark.json"
    watermark = json.loads(watermark_path.read_text(encoding="utf-8")) if watermark_path.exists() else None
    if watermark and (canonical_hash({k: v for k, v in watermark.items() if k != "fingerprint"}) != watermark.get("fingerprint")
            or market_close(watermark["validated_until"], "cn") > event_time(cutoff, "cn")):
        raise ValueError("学习验证时点记录无效")
    outcomes, pending = score(logs, histories, sessions, cutoff)
    evaluation, parameters = propose(outcomes, incumbent, validated_until=watermark["validated_until"] if watermark else None)
    if parameters and parameters != incumbent:
        code = deploy(parameters, home / "strategy", cutoff)
        write_log(home / "parameters" / (parameters["version"] + ".json"), parameters)
        watermark = {"validated_until": parameters["validated_until"], "parameter_version": parameters["version"]}
        watermark["fingerprint"] = canonical_hash(watermark)
        write_log(watermark_path, watermark)
        write_log(current, parameters)
    elif parameters:
        write_log(current, parameters)
    elif current.exists():
        write_log(current, None)
    snapshot = {"version": VERSION, "information_cutoff_at": cutoff, "mature_samples": len(outcomes), "pending_samples": pending,
        "status": evaluation["status"], "raw_metrics": calibration_metrics(outcomes), "evaluation": evaluation,
        "active_parameter_version": parameters["version"] if parameters else None,
        "active_parameters": parameters, "strategy_code": code,
        "recent_outcomes": outcomes[-6:], "dataset_hash": canonical_hash(outcomes),
        "boundary": "past_prediction_feedback_not_external_fact; price_direction_only_not_portfolio_profit; bounded_calibration_code_only; risk_rules_unchanged"}
    snapshot["fingerprint"] = canonical_hash(snapshot)
    write_log(home / "snapshots" / (str(cutoff)[:10] + "." + snapshot["fingerprint"][:16] + ".json"), snapshot)
    write_log(latest, snapshot)
    return snapshot


async def prepare(reader, day, sessions, cutoff, *, shadow=False):
    import asyncio
    from agent.sdk_runner import resolve_role_routing
    model = resolve_role_routing()["investment-manager"].model
    logs = past_logs(cutoff, model=model)
    histories = {}
    # No extra price requests before an exchange-calendar outcome could have matured.
    closed = [d for d in sessions if market_close(d, "cn") <= event_time(cutoff, "cn")]
    needed = {p["symbol"] for row in logs if row["timing"]["execution_session"] in closed
        and closed.index(row["timing"]["execution_session"]) + HORIZON < len(closed) for p in row.get("predictions", [])}
    for symbol in sorted(needed & set(reader.active_symbols))[:3]:
        try:
            histories[symbol] = await asyncio.wait_for(reader.call("get_history", {"symbol": symbol, "days": 756}), timeout=15)
        except (OSError, RuntimeError, asyncio.TimeoutError):
            continue
    home = family_home(model)
    if shadow:
        home = home / "shadow"
    return review(logs, histories, sessions, cutoff, home=home)


def review_cached(cutoff):
    """Zero model/network calls. Run feedback even if later news discovery fails."""
    from agent.sdk_runner import resolve_role_routing
    model = resolve_role_routing()["investment-manager"].model
    price_path = DATA_DIR / "market_views" / "cn-prices.json"
    calendar_path = DATA_DIR / "calendar" / "cn-sessions.json"
    histories, sessions = {}, []
    if price_path.exists():
        price = json.loads(price_path.read_text(encoding="utf-8"))
        if (price.get("market") != "cn" or price.get("data_quality") != "live_prices" or price.get("price_basis") != "qfq"
                or canonical_hash(price["histories"]) != price.get("fingerprint")):
            raise ValueError("学习价格缓存来源或指纹无效")
        histories = price["histories"]
    if calendar_path.exists():
        calendar = json.loads(calendar_path.read_text(encoding="utf-8"))
        if calendar.get("quality") != "provider_calendar" or canonical_hash(calendar["sessions"]) != calendar.get("fingerprint"):
            raise ValueError("学习交易日历来源或指纹无效")
        sessions = calendar["sessions"]
    return review(past_logs(cutoff, model=model), histories, sessions, cutoff, home=family_home(model))


def calibrate_predictions(result, snapshot):
    parameters = snapshot.get("active_parameters")
    if parameters:
        from agent.strategy_evolution import checked_function
        calibrate = checked_function(snapshot["strategy_code"]["source"], parameters["calibrator"])
        for row in result["predictions"]:
            row["calibrated_confidence"] = round(calibrate(row["confidence"]), 6)
            row["calibration_version"] = parameters["version"]


def public_summary(snapshot):
    result = {key: deepcopy(snapshot[key]) for key in ("version", "status", "information_cutoff_at", "mature_samples", "pending_samples", "active_parameter_version") if key in snapshot}
    if snapshot.get("strategy_code"):
        result["strategy_status"] = snapshot["strategy_code"]["status"]
    return result
