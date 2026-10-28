# -*- coding: utf-8 -*-
"""置信度校准指标与严格按时间切分的直方图校准器。"""
from __future__ import annotations

from collections import defaultdict


def _confidence(row: dict) -> float:
    value = float(row.get("calibrated_confidence", row.get("confidence")))
    if not 0 <= value <= 1:
        raise ValueError(f"confidence 必须在0到1之间:{value}")
    return value


def calibration_metrics(rows: list[dict], bins: int = 10) -> dict:
    if not rows:
        return {"sample_count": 0, "brier_score": None,
                "expected_calibration_error": None, "bins": []}
    bins = max(2, min(int(bins), 20))
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    squared = []
    for row in rows:
        confidence = _confidence(row)
        actual = 1.0 if bool(row.get("correct", row.get("hit"))) else 0.0
        index = min(int(confidence * bins), bins - 1)
        grouped[index].append((confidence, actual))
        squared.append((confidence - actual) ** 2)
    output_bins = []
    ece = 0.0
    for index in range(bins):
        samples = grouped.get(index, [])
        if not samples:
            continue
        average_confidence = sum(item[0] for item in samples) / len(samples)
        accuracy = sum(item[1] for item in samples) / len(samples)
        ece += len(samples) / len(rows) * abs(average_confidence - accuracy)
        output_bins.append({
            "lower": round(index / bins, 4), "upper": round((index + 1) / bins, 4),
            "count": len(samples), "average_confidence": round(average_confidence, 6),
            "accuracy": round(accuracy, 6),
        })
    return {
        "sample_count": len(rows),
        "brier_score": round(sum(squared) / len(squared), 6),
        "expected_calibration_error": round(ece, 6),
        "bins": output_bins,
    }


def fit_histogram_calibrator(rows: list[dict], bins: int = 5,
                             prior_weight: float = 2.0) -> dict:
    if not rows:
        raise ValueError("校准训练集不能为空")
    bins = max(2, min(int(bins), 20))
    global_accuracy = sum(bool(row.get("correct", row.get("hit"))) for row in rows) / len(rows)
    counts = [0] * bins
    hits = [0] * bins
    for row in rows:
        index = min(int(_confidence(row) * bins), bins - 1)
        counts[index] += 1
        hits[index] += int(bool(row.get("correct", row.get("hit"))))
    probabilities = [
        (hits[index] + prior_weight * global_accuracy) / (counts[index] + prior_weight)
        for index in range(bins)
    ]
    return {"bins": bins, "probabilities": probabilities,
            "training_samples": len(rows), "global_accuracy": global_accuracy}


def apply_calibrator(rows: list[dict], calibrator: dict) -> list[dict]:
    bins = int(calibrator["bins"])
    probabilities = calibrator["probabilities"]
    output = []
    for row in rows:
        confidence = _confidence(row)
        index = min(int(confidence * bins), bins - 1)
        output.append({**row, "calibrated_confidence": round(float(probabilities[index]), 6)})
    return output


def chronological_calibration(rows: list[dict], train_fraction: float = 0.7,
                              min_samples: int = 8) -> dict:
    """只用更早日期拟合，再在更晚日期评估；同一日期不会跨训练/测试集。"""
    valid = [row for row in rows if row.get("date") and row.get("confidence") is not None]
    valid.sort(key=lambda row: (str(row["date"]), str(row.get("market", "")),
                                str(row.get("symbol", ""))))
    dates = sorted({str(row["date"]) for row in valid})
    if len(valid) < min_samples or len(dates) < 2:
        return {"status": "insufficient_data", "sample_count": len(valid),
                "required_samples": min_samples,
                "message": "至少需要跨两个日期的已到期预测，才能做过去训练/未来验证。"}
    split_index = max(1, min(len(dates) - 1, int(len(dates) * train_fraction)))
    cutoff = dates[split_index]
    train = [row for row in valid if str(row["date"]) < cutoff]
    test = [row for row in valid if str(row["date"]) >= cutoff]
    if not train or not test:
        return {"status": "insufficient_data", "sample_count": len(valid),
                "required_samples": min_samples, "message": "时间切分后训练集或测试集为空。"}
    calibrator = fit_histogram_calibrator(train)
    calibrated = apply_calibrator(test, calibrator)
    return {
        "status": "complete", "sample_count": len(valid),
        "train_count": len(train), "test_count": len(test),
        "train_end": max(str(row["date"]) for row in train),
        "test_start": min(str(row["date"]) for row in test),
        "raw": calibration_metrics(test),
        "calibrated": calibration_metrics(calibrated),
        "method": "past-only histogram calibration with smoothed bins",
    }


def variant_calibration(rows: list[dict]) -> dict:
    """分市场/策略变体校准，避免把不同模型的小样本合成一个可校准大样本。"""
    from evals.experiments import canonical_hash
    groups = {}
    for row in rows:
        identity = canonical_hash({"market": row.get("market"), "variant": row.get("variant", {})})
        groups.setdefault(identity, []).append(row)
    entries = [{"market": group[0].get("market"), "variant": group[0].get("variant", {}),
                "calibration": chronological_calibration(group)} for group in groups.values()]
    if len(entries) == 1:
        aggregate = entries[0]["calibration"]
    elif not entries:
        aggregate = chronological_calibration([])
    else:
        aggregate = {"status": "multiple_variants", "sample_count": len(rows),
                     "message": "多个市场/策略变体分别校准，请查看各变体结果。"}
    return {"calibration": aggregate, "calibration_by_variant": entries}
