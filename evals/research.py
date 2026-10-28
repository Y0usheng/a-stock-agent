"""A股研究闭环：不可改写的计划、冻结输入、同成本重放、复盘和下一轮建议。"""
from __future__ import annotations

from contextlib import closing, contextmanager
from copy import deepcopy
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import math
import re
from pathlib import Path
import sqlite3
from uuid import uuid4

import pandas as pd

from agent.contracts import validate_decision_result
from agent.evidence import validate_evidence_audit
from config import EXPERIMENT_DB, MARKETS, MAX_POSITION_PCT
from evals.backtest import _technical_signal, walk_forward_backtest
from evals.calibration import calibration_metrics
from evals.experiments import canonical_hash, decision_log_variant, record_experiment
from evals.metrics import MOVE_THRESHOLD, actual_direction_from_frame
from evals.news_store import _canonical_event
from evals.price_data import normalize_history
from servers.data_quality import event_time, market_close, market_zone

VERSION = "cn-research-loop-v1"
PHASES = ("train", "validation", "test")
KINDS = {"buy_hold", "technical", "news_gate", "agent"}
COST_KEYS = ("initial_cash", "lot_size", "commission_rate", "commission_min", "stamp_tax_rate",
             "slippage_bps", "max_volume_participation_pct")


def now():
    return datetime.now(timezone.utc)


def code_identity():
    root = Path(__file__).resolve().parent
    paths = [root / name for name in ("research.py", "research_agent.py", "backtest.py", "metrics.py", "performance.py", "calibration.py")]
    paths += [root.parent / name for name in ("config.py", "agent/evidence.py", "agent/graph_runner.py", "agent/subscription_roles.py")]
    return canonical_hash({"files": {str(path.relative_to(root.parent)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                           "pandas_version": pd.__version__})


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _day(value):
    return date.fromisoformat(str(value)).isoformat()


def validate_plan(spec):
    value = deepcopy(spec)
    if value.get("market", "cn") != "cn":
        raise ValueError("研究闭环本阶段仅支持A股，不能注册美股实验")
    if not str(value.get("name", "")).strip() or not str(value.get("hypothesis", "")).strip():
        raise ValueError("必须事先填写研究名称和可检验假设")
    symbols = value.get("symbols")
    allowed = {row["symbol"] for row in MARKETS["cn"]["candidate_universe"]}
    if not isinstance(symbols, list) or not symbols or len(set(symbols)) != len(symbols) or not set(symbols) <= allowed:
        raise ValueError("股票池必须是无重复的受控A股代码")
    if value.get("horizon_days", 5) != 5:
        raise ValueError("当前模型预测协议固定为5个交易日")
    splits = value.get("splits", {})
    if set(splits) != set(PHASES):
        raise ValueError("必须固定train、validation和test三个时间区间")
    previous = None
    for phase in PHASES:
        span = splits[phase]
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError("每个时间区间需要两个日期")
        span[:] = [_day(day) for day in span]
        if span[0] > span[1] or (previous is not None and span[0] <= previous):
            raise ValueError("训练、验证和测试区间必须严格按时间排列且不重叠")
        previous = span[1]
    variants = value.get("variants")
    if not isinstance(variants, list) or not 2 <= len(variants) <= 8:
        raise ValueError("至少需要两个、最多八个预先固定的变体")
    ids = [variant.get("id") for variant in variants]
    if any(not isinstance(key, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", key) is None for key in ids) or len(set(ids)) != len(ids):
        raise ValueError("变体ID必须有效且不得重复")
    for variant in variants:
        if variant.get("kind") not in KINDS:
            raise ValueError("变体仅支持buy_hold、technical、news_gate和agent")
        if variant["kind"] == "agent" and (not isinstance(variant.get("variant_fingerprint"), str)
                                          or re.fullmatch(r"[0-9a-f]{64}", variant["variant_fingerprint"]) is None):
            raise ValueError("Agent变体须预先绑定decision_log_variant的完整指纹")
    if value.get("baseline") not in ids or value.get("candidate") not in ids or value["baseline"] == value["candidate"]:
        raise ValueError("必须预先指定不同的基线和待检验变体")
    mode = value.get("mode", "retrospective")
    if mode not in {"engineering", "retrospective", "prospective"}:
        raise ValueError("未知研究模式")
    if mode == "prospective" and splits["test"][0] <= now().astimezone(market_zone("cn")).date().isoformat():
        raise ValueError("向前实验必须在测试期开始前注册，不能倒填预注册时间")
    criteria = {"metric": "accuracy_pct", "min_gain": 0., "min_matured_predictions": 20,
                "min_nonoverlapping_dates": 4, "min_coverage": 1., "max_drawdown_pct": 20.,
                "max_total_cost_pct": 2., "require_semantic_review": False, **value.get("criteria", {})}
    if criteria["metric"] not in {"accuracy_pct", "net_return_pct"}:
        raise ValueError("主要评价指标为accuracy_pct或net_return_pct")
    for key in ("min_matured_predictions", "min_nonoverlapping_dates"):
        if isinstance(criteria[key], bool) or not isinstance(criteria[key], int) or criteria[key] < 1:
            raise ValueError(f"{key}必须为正整数")
    for key in ("min_gain", "min_coverage", "max_drawdown_pct", "max_total_cost_pct"):
        if isinstance(criteria[key], bool) or not isinstance(criteria[key], (int, float)) or not math.isfinite(criteria[key]):
            raise ValueError(f"{key}必须为有限数值")
    if not 0 < criteria["min_coverage"] <= 1 or min(criteria["max_drawdown_pct"], criteria["max_total_cost_pct"]) < 0:
        raise ValueError("覆盖率或成本/回撤上限无效")
    if type(criteria["require_semantic_review"]) is not bool:
        raise ValueError("require_semantic_review必须为布尔值")
    value.update(market="cn", symbols=sorted(symbols), horizon_days=5, mode=mode, criteria=criteria,
                 visibility_policy="first_seen" if mode == "prospective" else "published_by_close")
    return value


def normalize_bundle(plan, bundle):
    if bundle.get("market") != "cn" or set(bundle.get("histories", {})) != set(plan["symbols"]):
        raise ValueError("数据市场或股票池与计划不一致")
    prices = {symbol: normalize_history(symbol, pd.DataFrame(rows)).to_dict("records")
              for symbol, rows in sorted(bundle["histories"].items())}
    if any(not rows or rows[-1]["date"] > now().astimezone(market_zone("cn")).date().isoformat()
           for rows in prices.values()):
        raise ValueError("行情不能为空或包含未来日期")
    events = {}
    for raw in bundle.get("news_events", []):
        if raw.get("symbol") not in plan["symbols"]:
            raise ValueError("新闻包含股票池外标的")
        if not math.isfinite(float(raw.get("sentiment_score") or 0)):
            raise ValueError("新闻分数必须为有限数值")
        item = _canonical_event(raw, "cn", raw.get("data_quality", "imported"))
        if raw.get("content_hash") and raw["content_hash"] != item["content_hash"]:
            raise ValueError("新闻保存内容与其指纹不符")
        item["first_seen_at"] = raw.get("first_seen_at") or raw.get("ingested_at")
        if item["first_seen_at"]:
            event_time(item["first_seen_at"], "cn")
        if item["event_id"] in events and events[item["event_id"]] != item:
            raise ValueError("同一个新闻ID有不同内容")
        events[item["event_id"]] = item
    value = {"market": "cn", "histories": prices, "news_events": sorted(events.values(), key=lambda row: row["event_id"]),
        "price_basis": str(bundle.get("price_basis", "unspecified")),
        "price_quality": str(bundle.get("price_quality", "imported_unverified")),
        "news_coverage_note": str(bundle.get("news_coverage_note", "仅证明保存事件存在，未证明供应商全量覆盖"))}
    _json(value)
    return value


def bundle_from_files(prices_file, news_file=None, symbols=None):
    raw = json.loads(Path(prices_file).read_text(encoding="utf-8"))
    if raw.get("market") != "cn":
        raise ValueError("需要A股冻结价格JSON")
    histories = {symbol: normalize_history(symbol, pd.DataFrame(rows)).to_dict("records")
                 for symbol, rows in sorted(raw["histories"].items())}
    manifest = raw.get("manifest", {})
    basis = manifest.get("price_basis", raw.get("price_basis", "unspecified"))
    if manifest.get("fingerprint") and canonical_hash({"market": "cn", "prices": histories, "basis": basis}) != manifest["fingerprint"]:
        raise ValueError("价格文件与保存的内容指纹不符")
    if symbols is not None:
        histories = {symbol: histories[symbol] for symbol in symbols}
    news = json.loads(Path(news_file).read_text(encoding="utf-8")) if news_file else []
    news = news.get("events", []) if isinstance(news, dict) else news
    return {"market": "cn", "histories": histories, "news_events": news, "price_basis": basis,
            "price_quality": manifest.get("data_quality", raw.get("price_quality", "imported_unverified"))}


def _available(event, policy):
    published = event_time(event["published_at"], "cn")
    if policy == "published_by_close":
        return published
    if not event.get("first_seen_at"):
        return None
    return max(published, event_time(event["first_seen_at"], "cn"))


class ResearchBook:
    def __init__(self, path=EXPERIMENT_DB):
        self.path = Path(path)

    @contextmanager
    def db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS research_plans(id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                family_id TEXT NOT NULL, plan_json TEXT NOT NULL, plan_hash TEXT NOT NULL, code_hash TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS research_data(id TEXT PRIMARY KEY, study_id TEXT NOT NULL,
                created_at TEXT NOT NULL, fingerprint TEXT NOT NULL, data_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS research_logs(study_id TEXT NOT NULL, variant_id TEXT NOT NULL,
                day TEXT NOT NULL, captured_at TEXT NOT NULL, log_hash TEXT NOT NULL, log_json TEXT NOT NULL,
                semantic_json TEXT, PRIMARY KEY(study_id,variant_id,day));
            CREATE TABLE IF NOT EXISTS research_inputs(study_id TEXT NOT NULL, day TEXT NOT NULL,
                input_hash TEXT NOT NULL, PRIMARY KEY(study_id,day));
            CREATE TABLE IF NOT EXISTS research_attempts(id TEXT PRIMARY KEY, study_id TEXT NOT NULL,
                phase TEXT NOT NULL, variant_id TEXT NOT NULL, created_at TEXT NOT NULL,
                status TEXT NOT NULL, result_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS research_reviews(id TEXT PRIMARY KEY, study_id TEXT NOT NULL,
                created_at TEXT NOT NULL, report_json TEXT NOT NULL);
            """)
            yield conn

    def create(self, spec):
        plan = validate_plan(spec)
        if plan.get("parent_id"):
            parent = self.get(plan["parent_id"])
            if plan["splits"]["test"][0] <= parent["plan"]["splits"]["test"][1]:
                raise ValueError("下一轮研究需要独立的新测试期，不能继续在父任务测试集挑参数")
            with self.db() as conn:
                if not conn.execute("SELECT 1 FROM research_reviews WHERE study_id=?", (parent["id"],)).fetchone():
                    raise ValueError("下一轮研究须关联已有复盘的父任务")
            family_id = parent["family_id"]
        else:
            family_id = canonical_hash({"hypothesis": plan["hypothesis"].strip(), "symbols": plan["symbols"], "market": "cn"})
        plan["cost_model"] = {key: MARKETS["cn"][key] for key in COST_KEYS}
        plan["position_cap"] = MAX_POSITION_PCT
        plan["target_weight"] = min(.28, 1 / len(plan["symbols"]))
        plan["label_threshold"] = MOVE_THRESHOLD
        identifier = "research_" + uuid4().hex[:16]
        with self.db() as conn:
            conn.execute("INSERT INTO research_plans VALUES(?,?,?,?,?,?)", (
                identifier, now().isoformat(), family_id, _json(plan), canonical_hash(plan), code_identity()))
        return self.get(identifier)

    def get(self, identifier):
        with self.db() as conn:
            row = conn.execute("SELECT * FROM research_plans WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("研究任务不存在")
        plan = json.loads(row["plan_json"])
        if canonical_hash(plan) != row["plan_hash"]:
            raise ValueError("研究计划或验收标准被改写")
        return {"id": row["id"], "created_at": row["created_at"], "family_id": row["family_id"],
                "plan": plan, "plan_hash": row["plan_hash"], "code_hash": row["code_hash"]}

    def latest_data(self, identifier):
        with self.db() as conn:
            row = conn.execute("SELECT * FROM research_data WHERE study_id=? ORDER BY rowid DESC LIMIT 1", (identifier,)).fetchone()
        if row is None:
            return None
        value = json.loads(row["data_json"])
        if canonical_hash(value) != row["fingerprint"]:
            raise ValueError("冻结数据内容被改写")
        return {"id": row["id"], "fingerprint": row["fingerprint"], "data": value}

    def freeze(self, identifier, bundle):
        study = self.get(identifier)
        value = normalize_bundle(study["plan"], bundle)
        previous = self.latest_data(identifier)
        if previous:
            old = previous["data"]
            if canonical_hash(value) == previous["fingerprint"]:
                return previous
            for symbol in study["plan"]["symbols"]:
                fresh = {row["date"]: row for row in value["histories"][symbol]}
                if any(fresh.get(row["date"]) != row for row in old["histories"][symbol]):
                    raise ValueError("只能追加未来价格，不能改写或删除已冻结日线")
                end = old["histories"][symbol][-1]["date"]
                old_dates = {row["date"] for row in old["histories"][symbol]}
                if any(row["date"] <= end and row["date"] not in old_dates for row in value["histories"][symbol]):
                    raise ValueError("历史补录会改变交易日序列，需要新研究任务")
            if any(value[key] != old[key] for key in ("price_basis", "price_quality", "news_coverage_note")):
                raise ValueError("不能在同一研究中变更价格口径或来源质量")
            fresh_events = {row["event_id"]: row for row in value["news_events"]}
            if any(fresh_events.get(row["event_id"]) != row for row in old["news_events"]):
                raise ValueError("不能改写或删除已冻结新闻")
            old_ids = {row["event_id"] for row in old["news_events"]}
            cutoff = market_close(min(rows[-1]["date"] for rows in old["histories"].values()), "cn")
            for event in value["news_events"]:
                available = _available(event, study["plan"]["visibility_policy"])
                if event["event_id"] not in old_ids and (available is None or available <= cutoff):
                    raise ValueError("回填历史新闻会改变已冻结输入，请创建新研究")
        with self.db() as conn:
            conn.execute("INSERT INTO research_data VALUES(?,?,?,?,?)", (
                "data_" + uuid4().hex, identifier, now().isoformat(), canonical_hash(value), _json(value)))
        return self.latest_data(identifier)

    def attach_log(self, identifier, variant_id, log, semantic=None):
        study, frozen = self.get(identifier), self.latest_data(identifier)
        if frozen is None:
            raise ValueError("请先冻结价格和新闻输入")
        plan, data = study["plan"], frozen["data"]
        variant = next((row for row in plan["variants"] if row["id"] == variant_id), None)
        if variant is None or variant["kind"] != "agent":
            raise ValueError("只能给预注册Agent变体关联模型日志")
        if canonical_hash(decision_log_variant(log)) != variant["variant_fingerprint"]:
            raise ValueError("模型、Prompt或接入路径与预注册变体不符")
        day = log.get("as_of_date") or log.get("date")
        if any(row.get("horizon_days") != plan["horizon_days"] for row in log.get("predictions", [])):
            raise ValueError("预测期限与预注册研究不同，不能把长期输出计入5日实验")
        if log.get("market") != "cn" or not any(start <= day <= end for start, end in plan["splits"].values()):
            raise ValueError("模型日志市场或日期不属于研究范围")
        validate_decision_result(log, "cn", expected_symbols=set(plan["symbols"]))
        validate_evidence_audit(log)
        if log.get("execution", {}).get("orders_submitted") != 0 or log.get("execution_mode") != "shadow":
            raise ValueError("研究只接收零订单的shadow日志")
        inputs = log.get("input_dataset")
        if not isinstance(inputs, dict) or canonical_hash(inputs) != log.get("dataset_fingerprint"):
            raise ValueError("模型冻结输入与指纹不一致")
        if inputs.get("as_of_date") != day or set(inputs.get("active_symbols", [])) != set(plan["symbols"]):
            raise ValueError("冻结输入截面日或候选池与研究计划不符")
        for item in inputs.get("tool_data", []):
            if item.get("tool") == "get_history" and any(row.get("date", "") > day for row in item.get("result", [])):
                raise ValueError("模型读取了截面日以后的价格历史")
        snapshot = {row["symbol"]: row for row in inputs.get("snapshot", [])}
        for symbol in plan["symbols"]:
            price = next((row for row in data["histories"][symbol] if row["date"] == day), None)
            if (price is None or symbol not in snapshot or snapshot[symbol].get("date") != day
                    or abs(snapshot[symbol]["close"] - price["close"]) > 1e-8):
                raise ValueError("模型输入收盘价与研究冻结价格不一致")
            if int(snapshot[symbol].get("volume", price["volume"])) != price["volume"]:
                raise ValueError("模型输入成交量与冻结价格不一致")
        events = {row["event_id"]: row for row in data["news_events"]}
        for candidate in inputs.get("news_screen", {}).get("candidates", []):
            for event in candidate.get("top_events", []):
                canonical = _canonical_event(event, "cn", event.get("data_quality", "live"))
                if canonical["event_id"] not in events or events[canonical["event_id"]]["content_hash"] != canonical["content_hash"]:
                    raise ValueError("模型读取了未冻结或内容不同的新闻")
                available = _available(events[canonical["event_id"]], plan["visibility_policy"])
                if available is None or available > market_close(day, "cn"):
                    raise ValueError("模型读取的新闻在研究截面日尚不可见")
        generated = log.get("generated_at")
        if not generated or event_time(generated, "cn") > now().astimezone(market_zone("cn")):
            raise ValueError("模型日志缺少可信生成时间或时间在未来")
        if plan["mode"] == "prospective" and event_time(generated, "cn") < event_time(study["created_at"], "cn"):
            raise ValueError("向前研究不能关联注册前生成的测试日志")
        if plan["mode"] == "prospective" and event_time(generated, "cn") < market_close(day, "cn"):
            raise ValueError("收盘信号不能由收盘前生成的模型输出倒填")
        if semantic is not None:
            from evals.semantic_review import VERSION as REVIEW_VERSION, ReviewOptions, prepare_review, summarize, validate_judgments
            if semantic.get("version") != REVIEW_VERSION or semantic.get("source_log_fingerprint") != canonical_hash(log):
                raise ValueError("语义报告不属于该模型日志或协议版本")
            expected = prepare_review(log, ReviewOptions(**semantic["options"]))
            expected_by_id = {row["claim_id"]: row for row in expected["items"]}
            if [row["claim_id"] for row in semantic["items"]] != [row["claim_id"] for row in expected["items"]]:
                raise ValueError("语义报告命题范围被改写")
            for item in semantic["items"]:
                original = expected_by_id[item["claim_id"]]
                for key in ("text", "parent_text", "role", "evidence", "selected", "rule_findings", "sampling_stratum"):
                    if item[key] != original[key]:
                        raise ValueError("语义报告的命题、抽样或规则记录被改写")
                if item.get("review") is not None:
                    validate_judgments({"reviews": [item["review"]]}, [original], log["evidence_audit"]["catalog"])
            semantic = {"report_fingerprint": canonical_hash(semantic), "summary": summarize(semantic),
                        "status": semantic["status"]}
        with self.db() as conn:
            existing = conn.execute("SELECT log_hash,semantic_json FROM research_logs WHERE study_id=? AND variant_id=? AND day=?",
                                    (identifier, variant_id, day)).fetchone()
            if existing:
                if existing["log_hash"] != canonical_hash(log):
                    raise ValueError("同变体同日只接受首次输出，不能事后挑选更好的模型回答")
                if semantic is not None:
                    if existing["semantic_json"] and existing["semantic_json"] != _json(semantic):
                        raise ValueError("不能覆盖已关联的语义复核结果，请注册下一轮研究")
                    conn.execute("UPDATE research_logs SET semantic_json=? WHERE study_id=? AND variant_id=? AND day=?",
                                 (_json(semantic), identifier, variant_id, day))
                    return {"status": "semantic_attached", "day": day}
                return {"status": "already_attached", "day": day}
            feature = conn.execute("SELECT input_hash FROM research_inputs WHERE study_id=? AND day=?", (identifier, day)).fetchone()
            if feature and feature["input_hash"] != log["dataset_fingerprint"]:
                raise ValueError("同日模型变体的冻结账户、记忆或其他输入不同，不能直接A/B")
            conn.execute("INSERT OR IGNORE INTO research_inputs VALUES(?,?,?)", (identifier, day, log["dataset_fingerprint"]))
            conn.execute("INSERT INTO research_logs VALUES(?,?,?,?,?,?,?)", (
                identifier, variant_id, day, now().isoformat(), canonical_hash(log), _json(log), _json(semantic) if semantic else None))
        return {"status": "attached", "day": day, "variant_id": variant_id}

    def _logs(self, identifier, phase):
        start, end = self.get(identifier)["plan"]["splits"][phase]
        with self.db() as conn:
            rows = conn.execute("SELECT * FROM research_logs WHERE study_id=? AND day>=? AND day<=? ORDER BY day,variant_id",
                                (identifier, start, end)).fetchall()
        result = []
        for row in rows:
            log = json.loads(row["log_json"])
            if canonical_hash(log) != row["log_hash"]:
                raise ValueError("保存的模型日志被改写")
            result.append({"variant_id": row["variant_id"], "day": row["day"], "captured_at": row["captured_at"],
                           "log": log, "semantic": json.loads(row["semantic_json"]) if row["semantic_json"] else None})
        return result

    def attempts(self, identifier):
        with self.db() as conn:
            rows = conn.execute("SELECT * FROM research_attempts WHERE study_id=? ORDER BY rowid", (identifier,)).fetchall()
        return [{"attempt_id": row["id"], "phase": row["phase"], "variant_id": row["variant_id"],
                 "created_at": row["created_at"], "status": row["status"], **json.loads(row["result_json"])} for row in rows]

    def _package(self, identifier, phase, as_of):
        study, frozen = self.get(identifier), self.latest_data(identifier)
        if study["code_hash"] != code_identity():
            raise ValueError("研究代码与注册时不同，请创建下一轮研究而非覆盖旧结果")
        if frozen is None:
            return study, None
        plan, data = study["plan"], frozen["data"]
        start, end = plan["splits"][phase]
        cutoff = min(end, as_of)
        histories = {symbol: [row for row in rows if row["date"] <= cutoff]
                     for symbol, rows in data["histories"].items()}
        events = [row for row in data["news_events"] if event_time(row["published_at"], "cn") <= market_close(cutoff, "cn")]
        logs = [row for row in self._logs(identifier, phase) if row["day"] <= cutoff]
        package = {"phase": phase, "start": start, "end": end, "cutoff": cutoff,
            "histories": histories, "news_events": events, "logs": logs,
            "price_basis": data["price_basis"], "price_quality": data["price_quality"],
            "visibility_policy": plan["visibility_policy"], "cost_model": plan["cost_model"],
            "target_weight": plan["target_weight"], "position_cap": plan["position_cap"],
            "plan_hash": study["plan_hash"], "code_hash": study["code_hash"]}
        return study, package

    def run_phase(self, identifier, phase, as_of=None):
        if phase not in PHASES:
            raise ValueError("未知研究阶段")
        as_of = _day(as_of or now().astimezone(market_zone("cn")).date().isoformat())
        if as_of > now().astimezone(market_zone("cn")).date().isoformat():
            raise ValueError("评价截止日期不得在未来")
        study, package = self._package(identifier, phase, as_of)
        previous = self.attempts(identifier)
        prerequisites = PHASES[:PHASES.index(phase)]
        latest = {(row["phase"], row["variant_id"]): row for row in previous}
        required = {study["plan"]["baseline"], study["plan"]["candidate"]}
        ready = all(latest.get((prior, variant), {}).get("status") == "complete" for prior in prerequisites for variant in required)
        cache, output = {}, []
        variants = sorted(study["plan"]["variants"], key=lambda row: row["id"] != study["plan"]["baseline"])
        for variant in variants:
            attempt_id = "attempt_" + uuid4().hex
            with self.db() as conn:
                conn.execute("INSERT INTO research_attempts VALUES(?,?,?,?,?,?,?)", (
                    attempt_id, identifier, phase, variant["id"], now().isoformat(), "running", "{}"))
            try:
                if package is None:
                    result = {"status": "no_data", "reason": "尚未冻结输入"}
                elif not ready:
                    result = {"status": "pending", "reason": "训练/验证的固定基线与候选尚未完成，未进入下一阶段"}
                else:
                    result = evaluate_variant(study, package, variant, cache)
                if package is not None:
                    result["package"] = package
                    result["dataset_fingerprint"] = canonical_hash(package)
                registry = record_experiment("research:" + identifier + ":" + phase, "cn",
                    {"fingerprint": result.get("dataset_fingerprint", "missing"), "plan_hash": study["plan_hash"],
                     "phase": phase, "source": "frozen_research_package"}, variant,
                    result.get("metrics", {}), status="complete" if result["status"] == "complete" else
                    "failed" if result["status"] == "failed" else "no_data", db_path=self.path,
                    notes=result.get("reason", "同一阶段冻结输入与费用口径；不是实盘成绩"))
                result["registry_run_id"] = registry["run_id"]
            except Exception as exc:
                from agent.errors import safe_error
                result = {"status": "failed", "reason": safe_error(exc)}
                if package is not None:
                    result.update(package=package, dataset_fingerprint=canonical_hash(package))
                registry = record_experiment("research:" + identifier + ":" + phase, "cn",
                    {"fingerprint": result.get("dataset_fingerprint", "missing"), "plan_hash": study["plan_hash"], "phase": phase},
                    variant, {}, status="failed", notes=result["reason"], db_path=self.path)
                result["registry_run_id"] = registry["run_id"]
            with self.db() as conn:
                conn.execute("UPDATE research_attempts SET status=?,result_json=? WHERE id=?",
                             (result["status"], _json(result), attempt_id))
            output.append({"attempt_id": attempt_id, "variant_id": variant["id"], **result})
        return output

    def replay(self, identifier, attempt_id):
        study = self.get(identifier)
        run = next((row for row in self.attempts(identifier) if row["attempt_id"] == attempt_id), None)
        if run is None or run["status"] != "complete" or run["phase"] not in PHASES:
            raise ValueError("只能重放该研究已完成的运行")
        if study["code_hash"] != code_identity() or canonical_hash(run["package"]) != run["dataset_fingerprint"]:
            raise ValueError("代码或保存输入已变化，不能声称原样重放")
        variant = next(row for row in study["plan"]["variants"] if row["id"] == run["variant_id"])
        replayed = evaluate_variant(study, run["package"], variant, {})
        expected = {key: run[key] for key in ("status", "metrics", "scored_rows", "portfolio")}
        actual = {key: replayed[key] for key in expected}
        return {"attempt_id": attempt_id, "reproduced": canonical_hash(expected) == canonical_hash(actual),
                "result_fingerprint": canonical_hash(actual)}

    def review(self, identifier):
        study = self.get(identifier)
        runs = self.attempts(identifier)
        latest = {(row["phase"], row["variant_id"]): row for row in runs}
        report = review_results(study, latest)
        with self.db() as conn:
            members = [row[0] for row in conn.execute("SELECT id FROM research_plans WHERE family_id=?", (study["family_id"],))]
            count = conn.execute("SELECT COUNT(*) FROM research_attempts WHERE study_id IN (" + ",".join("?" for _ in members) + ")", members).fetchone()[0]
            report.update(study_id=identifier, plan=study["plan"], plan_hash=study["plan_hash"], code_hash=study["code_hash"],
                family_study_count=len(members), family_attempt_count=count, attempt_count=len(runs),
                attempts=[{key: row.get(key) for key in ("attempt_id", "phase", "variant_id", "status", "reason", "metrics", "registry_run_id", "dataset_fingerprint")} for row in runs],
                generated_at=now().isoformat(), auto_promoted=False)
            report["next_hypothesis"] = next_proposal(study, report)
            conn.execute("INSERT INTO research_reviews VALUES(?,?,?,?)", (
                "review_" + uuid4().hex, identifier, now().isoformat(), _json(report)))
        return report

    def summaries(self, limit=20):
        if not self.path.exists():
            return []
        with self.db() as conn:
            plans = conn.execute("SELECT id,created_at,plan_json FROM research_plans ORDER BY rowid DESC LIMIT ?", (limit,)).fetchall()
            output = []
            for row in plans:
                review = conn.execute("SELECT report_json FROM research_reviews WHERE study_id=? ORDER BY rowid DESC LIMIT 1", (row["id"],)).fetchone()
                plan = json.loads(row["plan_json"])
                parsed = json.loads(review[0]) if review else {}
                count = conn.execute("SELECT COUNT(*) FROM research_attempts WHERE study_id=?", (row["id"],)).fetchone()[0]
                output.append({"id": row["id"], "name": plan["name"], "hypothesis": plan["hypothesis"],
                    "created_at": row["created_at"], "verdict": ("awaiting_review" if count != parsed.get("attempt_count", 0) else parsed.get("verdict", "registered")),
                    "attempt_count": count, "next_hypothesis": parsed.get("next_hypothesis")})
        return output


def evaluate_variant(study, package, variant, signal_cache):
    plan = study["plan"]
    histories = {symbol: pd.DataFrame(rows) for symbol, rows in package["histories"].items()}
    if any(frame.empty for frame in histories.values()):
        return {"status": "pending", "reason": "尚无覆盖该阶段的价格"}
    sessions = sorted(set.intersection(*(set(frame["date"]) for frame in histories.values())))
    days = [day for day in sessions if package["start"] <= day <= package["end"]]
    if package["cutoff"] < package["end"] or not days or days[-1] < package["end"]:
        return {"status": "pending", "reason": "阶段尚未结束或价格尚未覆盖截止交易日",
                "progress": {"observed_dates": len(days), "matured_cases": max(0, len(days) - plan["horizon_days"]) * len(plan["symbols"]),
                             "required_end": package["end"]}}
    if len(days) <= plan["horizon_days"] or sessions.index(days[0]) < 26:
        return {"status": "no_data", "reason": "缺少指标预热或足够的5日到期窗口"}
    scores, usable_events = {}, []
    for event in package["news_events"]:
        available = _available(event, plan["visibility_policy"])
        if available is None:
            continue
        used = False
        for day in days:
            if (event_time(event["published_at"], "cn").date() >= date.fromisoformat(day) - timedelta(days=3)
                    and available <= market_close(day, "cn")):
                key = (day, event["symbol"])
                scores[key] = scores.get(key, 0.) + event["sentiment_score"]
                used = True
        if used:
            usable_events.append(event)
    if variant["kind"] == "news_gate" and not usable_events:
        return {"status": "no_data", "reason": "该阶段没有可用的冻结新闻，没有降级为合成事件"}
    model_logs = [row for row in package["logs"] if row["variant_id"] == variant["id"]]
    logs_by_day = {row["day"]: row for row in model_logs}
    signals, evidence_gaps = {}, 0
    if variant["kind"] == "agent":
        missing = [day for day in days[:-1] if day not in logs_by_day]
        if missing:
            return {"status": "no_data", "reason": f"缺少{len(missing)}个交易日的预存模型输出；没有调用替身或收费模型补写历史"}
        for day, attached in logs_by_day.items():
            if day not in days:
                continue
            log = attached["log"]
            if plan["mode"] == "prospective" and day != days[-1]:
                next_day = sessions[sessions.index(day) + 1]
                entry = datetime.combine(date.fromisoformat(next_day), time(9, 30), tzinfo=market_zone("cn"))
                if max(event_time(log["generated_at"], "cn"), event_time(attached["captured_at"], "cn")) >= entry:
                    return {"status": "no_data", "reason": "模型输出未在次日开盘前生成并登记，不能用于向前成交重放"}
            for prediction in log["predictions"]:
                supported = prediction.get("evidence", {}).get("status") == "supported"
                evidence_gaps += int(not supported)
                signals[(day, prediction["symbol"])] = {**prediction, "eligible": supported,
                                                       "direction": prediction["direction"] if supported else "flat"}
    scored = []
    for day in days[:-plan["horizon_days"]]:
        for symbol in plan["symbols"]:
            key = (symbol, day)
            if key not in signal_cache:
                signal_cache[key] = _technical_signal(histories[symbol], day)
            technical = signal_cache[key]
            if technical is None:
                return {"status": "no_data", "reason": "指标预热不足"}
            if variant["kind"] == "agent":
                predicted = signals[(day, symbol)]
            elif variant["kind"] == "buy_hold":
                predicted = {"direction": "up", "confidence": .5, "eligible": True}
            else:
                direction = "up" if technical["bullish"] else "down" if not technical["above_ma20"] else "flat"
                if variant["kind"] == "news_gate":
                    score = scores.get((day, symbol), 0.)
                    direction = "down" if score <= -1.5 else "flat" if score < 0 else direction
                predicted = {"direction": direction, "confidence": .5, "eligible": True}
            actual = actual_direction_from_frame(histories[symbol], day, plan["horizon_days"], plan["label_threshold"])
            index = histories[symbol]["date"].tolist().index(day)
            outcome_prices = histories[symbol].iloc[index:index + plan["horizon_days"] + 1][["date", "close"]].to_dict("records")
            scored.append({"date": day, "symbol": symbol, "actual": actual, "pred": predicted["direction"],
                "confidence": predicted["confidence"], "eligible": predicted["eligible"],
                "correct": actual == predicted["direction"], "horizon_days": plan["horizon_days"],
                "target_date": outcome_prices[-1]["date"], "outcome_fingerprint": canonical_hash(outcome_prices)})
    portfolio = walk_forward_backtest(histories, "cn", 60, variant["kind"] == "news_gate", scores,
        start_date=package["start"], end_date=package["end"],
        strategy="agent_predictions" if variant["kind"] == "agent" else "buy_hold" if variant["kind"] == "buy_hold" else "technical",
        prediction_signals=signals, config_override=plan["cost_model"], target_weight=plan["target_weight"],
        position_cap=plan["position_cap"], signal_cache=signal_cache)
    valid = [row for row in scored if row["eligible"]]
    calibration = calibration_metrics(valid)
    semantics = [row["semantic"] for row in model_logs if row["semantic"]]
    usage = [row["log"].get("agent_sdk", {}).get("usage", {}) for row in model_logs]
    gaps = sum(not record["eligible"] for row in model_logs for record in row["log"]["evidence_audit"]["catalog"]["records"])
    metrics = {"sample_count": len(valid), "expected_samples": len(scored), "coverage": len(valid) / len(scored),
        "accuracy_pct": sum(row["correct"] for row in valid) / len(valid) * 100 if valid else None,
        "brier_score": calibration["brier_score"], "ece": calibration["expected_calibration_error"],
        "net_return_pct": portfolio["metrics"]["total_return_pct"],
        "max_drawdown_pct": portfolio["metrics"]["max_drawdown_pct"],
        "total_fees": portfolio["total_fees"], "total_slippage_cost": portfolio["total_slippage_cost"],
        "total_cost_pct": (portfolio["total_fees"] + portfolio["total_slippage_cost"]) / plan["cost_model"]["initial_cash"] * 100,
        "trade_count": len(portfolio["trades"]), "evidence_insufficient_predictions": evidence_gaps,
        "unavailable_evidence_records": gaps, "semantic_reviewed_logs": len(semantics),
        "semantic_missing_logs": len(model_logs) - len(semantics),
        "semantic_incomplete_logs": sum(row["status"] != "completed" for row in semantics),
        "semantic_critical_unreviewed": sum(len(row["summary"].get("unselected_critical_ids", [])) +
                                           len(row["summary"].get("selected_not_reviewed_ids", [])) for row in semantics),
        "semantic_role_errors": sum(row["summary"]["rule_findings"].get("role_scope_error", 0) for row in semantics),
        "model_requests": sum(row.get("requests", 0) for row in usage),
        "model_tokens": sum(row.get("total_tokens", 0) for row in usage),
        "model_duration_ms": sum(row["log"].get("agent_sdk", {}).get("duration_ms", 0) for row in model_logs)}
    synthetic = "synthetic" in package["price_quality"] or any(row["data_quality"] == "synthetic" for row in usable_events)
    synthetic = synthetic or any(record["data_quality"] == "synthetic" for row in model_logs
                                 for record in row["log"]["evidence_audit"]["catalog"]["records"])
    return {"status": "complete", "metrics": metrics, "scored_rows": scored, "portfolio": portfolio,
        "synthetic_inputs": synthetic, "interpretation": "方向命中与组合净收益分别评价；重叠预测不是独立样本；无自动晋级"}


def review_results(study, latest):
    plan, criteria = study["plan"], study["plan"]["criteria"]
    baseline = latest.get(("test", plan["baseline"]))
    candidate = latest.get(("test", plan["candidate"]))
    base = {"version": VERSION, "verdict": "awaiting_data", "reason": "固定测试期的基线与候选尚未完成",
            "strategy_effect_verified": False, "comparison": None}
    if not baseline or not candidate or baseline["status"] != "complete" or candidate["status"] != "complete":
        return base
    if baseline["dataset_fingerprint"] != candidate["dataset_fingerprint"]:
        return {**base, "verdict": "not_comparable", "reason": "冻结输入、评价区间或费用口径不同"}
    left = {(row["date"], row["symbol"]): row for row in baseline["scored_rows"] if row["eligible"]}
    right = {(row["date"], row["symbol"]): row for row in candidate["scored_rows"] if row["eligible"]}
    keys = sorted(set(left) & set(right))
    paired = [right[key] for key in keys]
    days = sorted({key[0] for key in keys})
    sessions = sorted(set.intersection(*(set(row["date"] for row in rows) for rows in candidate["package"]["histories"].values())))
    nonoverlapping, prior = [], -plan["horizon_days"]
    for day in days:
        index = sessions.index(day)
        if index - prior >= plan["horizon_days"]:
            nonoverlapping.append(day)
            prior = index
    accuracy_gain = ((sum(right[key]["correct"] for key in keys) - sum(left[key]["correct"] for key in keys))
                     / len(keys) * 100) if keys else None
    gain = accuracy_gain if criteria["metric"] == "accuracy_pct" else candidate["metrics"]["net_return_pct"] - baseline["metrics"]["net_return_pct"]
    comparison = {"metric": criteria["metric"], "gain": gain, "paired_sample_count": len(keys),
        "paired_cases_fingerprint": canonical_hash(keys), "nonoverlapping_signal_dates": len(nonoverlapping),
        "candidate": candidate["metrics"], "baseline": baseline["metrics"],
        "paired_candidate_brier": calibration_metrics(paired)["brier_score"]}
    if len(keys) < criteria["min_matured_predictions"] or len(nonoverlapping) < criteria["min_nonoverlapping_dates"]:
        return {**base, "verdict": "insufficient_data", "reason": "到期配对样本或不重叠预测日期不足", "comparison": comparison}
    if criteria["require_semantic_review"] and (candidate["metrics"]["semantic_missing_logs"] or
        candidate["metrics"]["semantic_incomplete_logs"] or candidate["metrics"]["semantic_critical_unreviewed"]):
        return {**base, "verdict": "awaiting_review", "reason": "预注册要求的语义复核尚未覆盖关键结论", "comparison": comparison}
    passed = (gain is not None and gain >= criteria["min_gain"] and candidate["metrics"]["coverage"] >= criteria["min_coverage"]
              and abs(candidate["metrics"]["max_drawdown_pct"]) <= criteria["max_drawdown_pct"]
              and candidate["metrics"]["total_cost_pct"] <= criteria["max_total_cost_pct"]
              and (not criteria["require_semantic_review"] or candidate["metrics"]["semantic_role_errors"] == 0))
    if plan["mode"] == "engineering" or baseline["synthetic_inputs"] or candidate["synthetic_inputs"]:
        return {**base, "verdict": "unverified_engineering", "reason": "工程或合成数据验收，不能认定真实策略有效", "criteria_passed": passed, "comparison": comparison}
    verdict = "retain_for_forward_validation" if passed else "reject"
    if passed and plan["mode"] == "prospective":
        verdict = "retain_for_paper_validation"
    return {**base, "verdict": verdict, "criteria_passed": passed,
            "reason": "按预注册标准保留为下一轮模拟候选" if passed else "未达到预注册增益、覆盖率、费用或回撤标准",
            "comparison": comparison}


def next_proposal(study, report):
    reasons = " ".join(str(row.get("reason", "")) for row in report.get("attempts", []))
    if "新闻" in reasons and report["verdict"] == "awaiting_data":
        hypothesis, action = "仅使用有发布时间和首次采集记录的A股事件，新闻门控能否改善技术基线？", "先积累真实新闻，再注册独立未来测试期"
    elif report["verdict"] in {"awaiting_data", "insufficient_data", "awaiting_review"}:
        hypothesis, action = study["plan"]["hypothesis"], "补齐冻结输入、预存模型输出、到期样本或语义复核，不改写当前验收标准"
    elif report["verdict"] == "unverified_engineering":
        hypothesis, action = study["plan"]["hypothesis"], "保持固定版本，以真实资料注册新的向前模拟研究"
    elif report["verdict"] == "reject":
        hypothesis = f"在独立未来区间，{study['plan']['candidate']}相对{study['plan']['baseline']}的判断增益能否达到预先标准？"
        action = "保留本轮否决结果，只提出待验证假设，不在旧测试集继续挑参数"
    else:
        hypothesis, action = "原配置的增益能否在新的独立未来区间保持？", "注册关联父任务的新研究并继续模拟验证"
    return {"parent_id": study["id"], "hypothesis": hypothesis, "recommended_action": action,
            "proposal_only": True, "changes_to_risk_rules": False, "new_test_period_required": True}


def render_review(report):
    plan = report["plan"]
    def fmt(value):
        return "—" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
    lines = ["# A股研究复盘", "", plan["name"], "", "假设：" + plan["hypothesis"], "",
        "结论：" + report["verdict"], "", report["reason"], "",
        f"研究：{report['study_id']}；模式：{plan['mode']}；股票：{', '.join(plan['symbols'])}。", "",
        f"固定基线：{plan['baseline']}；固定候选：{plan['candidate']}；预测期限：5个交易日。", "",
        "时间区间：" + "；".join(f"{phase} {plan['splits'][phase][0]} 至 {plan['splits'][phase][1]}" for phase in PHASES) + "。", "",
        f"主要指标：{plan['criteria']['metric']}；最小增益：{plan['criteria']['min_gain']}个百分点；"
        f"最少配对预测：{plan['criteria']['min_matured_predictions']}；最少不重叠日期：{plan['criteria']['min_nonoverlapping_dates']}。", "",
        f"计划指纹：`{report['plan_hash']}`；代码指纹：`{report['code_hash']}`。", "",
        f"本研究运行 {report['attempt_count']} 次；同假设家族共 {report['family_attempt_count']} 次，未隐藏失败或无数据运行。", "",
        "| 阶段 | 变体 | 状态 | 命中率% | 净收益% | 最大回撤% | 费用+滑点元 |", "|---|---|---|---|---|---|---|"]
    reasons = []
    for row in report["attempts"]:
        metrics = row.get("metrics") or {}
        lines.append("| " + " | ".join(fmt(value) for value in (row["phase"], row["variant_id"], row["status"],
            metrics.get("accuracy_pct"), metrics.get("net_return_pct"), metrics.get("max_drawdown_pct"),
            (metrics.get("total_fees", 0) + metrics.get("total_slippage_cost", 0)) if metrics else None)) + " |")
        if row.get("reason"):
            reasons.append(f"{row['phase']} / {row['variant_id']}：{row['reason']}")
    if reasons:
        lines += ["", "缺口和失败记录：", ""] + ["- " + reason for reason in reasons]
    comparison = report.get("comparison")
    if comparison:
        lines += ["", f"测试期固定配对：{comparison['paired_sample_count']}条；不重叠信号日期：{comparison['nonoverlapping_signal_dates']}；"
                  f"主要指标增益：{fmt(comparison['gain'])}个百分点。", "",
                  "| 测试期 | 到期有效样本 | 覆盖率 | Brier | ECE | 模型请求 | tokens | 模型耗时ms |", "|---|---|---|---|---|---|---|---|"]
        for key in ("baseline", "candidate"):
            metrics = comparison[key]
            lines.append("| " + " | ".join([key] + [fmt(metrics.get(name)) for name in
                ("sample_count", "coverage", "brier_score", "ece", "model_requests", "model_tokens", "model_duration_ms")]) + " |")
    lines += ["", "未覆盖：缺失的新闻或模型输出不会补成替身成绩。Agent组合为固定仓位规则分配已保存预测的模拟结果，不是原始决策订单的收益。", "",
        "下一轮建议：" + report["next_hypothesis"]["hypothesis"], "",
        report["next_hypothesis"]["recommended_action"], "", "建议不会自动执行或改动风控。方向判断、组合收益和研究效率分别记录；实验通过不等于策略盈利能力已验证。", ""]
    return "\n".join(lines)
