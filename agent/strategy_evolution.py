"""Generate bounded calibration strategy code; test, compare, rollback, then activate.

Only the exact source generated below may execute. No imports, model-written arbitrary
Python, account writes, network, or edits to the application/risk rules are allowed.
"""
from copy import deepcopy
import math
from pathlib import Path

from agent.analysis import write_log
from evals.experiments import canonical_hash
from servers.data_quality import event_time, market_close

VERSION = "tested-calibration-code-v1"


def render(calibrator):
    bins, probabilities = calibrator["bins"], calibrator["probabilities"]
    if (type(bins) is not int or not 2 <= bins <= 20 or len(probabilities) != bins
            or any(type(p) not in {int, float} or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)):
        raise ValueError("策略校准参数非法")
    return ("def calibrate(confidence):\n"
            "    if type(confidence) not in (int, float) or not 0 <= confidence <= 1:\n"
            "        raise ValueError('invalid confidence')\n"
            f"    return {tuple(probabilities)!r}[min(int(confidence * {bins}), {bins - 1})]\n")


def checked_function(source, calibrator):
    # Byte-for-byte allowlist: never execute arbitrary text read from a local candidate.
    if source != render(calibrator):
        raise ValueError("策略代码与允许的纯函数模板不符")
    namespace = {"__builtins__": {"type": type, "int": int, "float": float, "min": min, "ValueError": ValueError}}
    exec(compile(source, "<validated-paper-calibration>", "exec"), namespace)
    function = namespace["calibrate"]
    for i in range(1001):
        confidence = i / 1000
        expected = calibrator["probabilities"][min(int(confidence * calibrator["bins"]), calibrator["bins"] - 1)]
        if function(confidence) != expected:
            raise ValueError("策略代码与冻结参数的对照不一致")
    for invalid in (-1, 2, True, None, float("nan"), float("inf")):
        try:
            function(invalid)
        except (ValueError, TypeError):
            continue
        raise ValueError("策略代码边界测试失败")
    return function


def load(home, pointer, cutoff):
    import json
    import re
    if not isinstance(pointer, str) or not re.fullmatch(r"[0-9a-f]{64}", pointer):
        raise ValueError("策略版本身份非法")
    entry = json.loads((Path(home) / "versions" / (pointer + ".json")).read_text(encoding="utf-8"))
    if canonical_hash({k: v for k, v in entry.items() if k != "id"}) != pointer:
        raise ValueError("策略版本指纹不符")
    if event_time(entry["activated_at"], "cn") > event_time(cutoff, "cn"):
        raise ValueError("不能使用未来策略")
    if (Path(home) / "versions" / (pointer + ".py")).read_text(encoding="utf-8") != entry["source"]:
        raise ValueError("策略代码文件被修改")
    checked_function(entry["source"], entry["parameters"]["calibrator"])
    return entry


def deploy(parameters, home, cutoff):
    """Requires an improving purged control experiment, then exercises pointer rollback."""
    import json
    home = Path(home)
    validation = parameters["validation"]
    if (canonical_hash({k: v for k, v in parameters.items() if k != "version"}) != parameters["version"]
            or validation["status"] != "adopted" or validation["train_count"] < 30 or validation["test_count"] < 10
            or not validation["candidate_brier"] + .01 < validation["baseline_brier"]
            or market_close(validation["train_last_outcome"], "cn") >= event_time(validation["test_first_decision"], "cn")
            or market_close(parameters["validated_until"], "cn") > event_time(cutoff, "cn")):
        raise ValueError("策略代码未通过时间隔离的对照实验")
    source = render(parameters["calibrator"])
    checked_function(source, parameters["calibrator"])
    active = home / "active.json"
    before = json.loads(active.read_text(encoding="utf-8")) if active.exists() else {"active": None, "previous": None}
    if before["active"]:
        load(home, before["active"], cutoff)
    entry = {"version": VERSION, "parameters": deepcopy(parameters), "source": source, "activated_at": cutoff,
             "tests": "1001_equivalence_cases_and_invalid_inputs_passed", "control": validation,
             "rollback": "pointer_restore_and_previous_code_load_passed"}
    entry["id"] = canonical_hash(entry)
    write_log(home / "versions" / (entry["id"] + ".json"), entry)
    (home / "versions" / (entry["id"] + ".py")).write_text(source, encoding="utf-8")
    # Exercise on a separate pointer. A crash during rehearsal cannot switch production.
    rehearsal = home / "rollback-check.json"
    write_log(rehearsal, {"active": entry["id"]})
    load(home, json.loads(rehearsal.read_text(encoding="utf-8"))["active"], cutoff)
    write_log(rehearsal, before)
    if json.loads(rehearsal.read_text(encoding="utf-8")) != before:
        raise ValueError("策略回滚演练失败")
    write_log(active, {"active": entry["id"], "previous": before["active"]})
    return {"status": "activated", "version": entry["id"], "source": source, "tests": "passed", "control": "passed", "rollback": "passed"}


def healthcheck(home, cutoff):
    """Corruption fails back to the last checked code, otherwise to raw confidence."""
    import json
    import re
    home = Path(home)
    active = home / "active.json"
    if not active.exists():
        return {"status": "waiting_for_mature_samples"}
    try:
        pointer = json.loads(active.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pointer = {"active": None, "previous": None}
    if isinstance(pointer.get("active"), str) and re.fullmatch(r"[0-9a-f]{64}", pointer["active"]):
        try:
            saved = json.loads((home / "versions" / (pointer["active"] + ".json")).read_text(encoding="utf-8"))
            if event_time(saved["activated_at"], "cn") > event_time(cutoff, "cn"):
                raise RuntimeError("历史重放不能回滚未来生效的策略，请使用独立学习目录")
        except (OSError, ValueError, KeyError, TypeError):
            pass
    try:
        entry = load(home, pointer["active"], cutoff)
        return {"status": "active", "version": entry["id"], "source": entry["source"]}
    except (OSError, ValueError, KeyError, TypeError):
        try:
            entry = load(home, pointer.get("previous"), cutoff)
            write_log(active, {"active": entry["id"], "previous": None})
            result = {"status": "rolled_back", "version": entry["id"], "source": entry["source"]}
        except (OSError, ValueError, KeyError, TypeError):
            write_log(active, {"active": None, "previous": None})
            result = {"status": "rolled_back_to_raw"}
        write_log(home / "last-rollback.json", {**result, "checked_at": cutoff})
        return result
