"""长周期投资协议、持久研究底稿与宿主校验；不会发起交易。"""
from contextlib import closing, contextmanager
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit

from config import DATA_DIR, MARKETS, MAX_POSITION_PCT
from evals.experiments import canonical_hash
from servers.data_quality import event_time, market_close

VERSION = "long-term-investment-v1"
PROMPT_VERSION = "manager-long-v3"
POLICY = {"version": VERSION, "thesis_horizon_years": 3, "evaluation_horizon_days": 252,
          "review_calendar_days": 90, "news_role": "thesis_change_monitor", "force_investment": False}
CATEGORIES = {"business", "financials", "industry"}
RESEARCH_DB = DATA_DIR / "company_research.db"


def is_long(version):
    return version in {"manager-long-v1", "manager-long-v2", PROMPT_VERSION}


def case_variant(log):
    sdk = log.get("agent_sdk", {})
    return canonical_hash({"template": sdk.get("prompt", {}).get("template_sha256"),
        "roles": sdk.get("role_models", {}), "settings": {key: sdk.get("budget", {}).get(key) for key in ("reasoning_effort", "max_output_tokens")}, "policy": POLICY})


def manager_prompt(market, day, symbols):
    from agent.sdk_runner import RESULT_SCHEMA
    m = MARKETS[market]
    return (f"你是{m['name']}长期模拟投资经理，分析日{day}，固定研究池{','.join(symbols)}。\n" + LONG_INSTRUCTIONS +
        f"\n硬约束：仅受控股票；股数为{m['lot_size']}整数倍，单股仓位不超过{MAX_POSITION_PCT:.0%}，遵守现金和T+1约束。"
        "只提出计划，由宿主交独立风控审批；不得声称尚未发生的审批或成交。"
        "业务/财务原文只是来源记录，估值是带假设的模型推断，不能承诺收益。"
        "每只股票提供investment_cases；一年252交易日预测用于阶段观察，不作为每日交易开关。"
        "只返回JSON：" + json.dumps(result_schema(RESULT_SCHEMA), ensure_ascii=False, separators=(",", ":")))


def controlled_url(value):
    host = (urlsplit(str(value)).hostname or "").lower()
    return urlsplit(str(value)).scheme == "https" and any(host == domain or host.endswith("." + domain)
        for domain in ("cninfo.com.cn", "sse.com.cn", "szse.cn"))


def select_materials(rows, limit=4):
    selected = []
    for category in ("business", "financials", "industry"):
        candidates = [item for item in rows if category in item["categories"]]
        if category == "financials":
            usable = [item for item in candidates if item.get("unit_context_present") is not False]
            candidates = usable or candidates
            candidates.sort(key=lambda item: not bool(item.get('numeric_facts')))
        row = next((item for item in candidates if category == "financials" and item.get("document_type") == "annual"), None) or next(iter(candidates), None)
        if row is not None and row not in selected:
            selected.append(row)
        if category == "financials" and candidates and candidates[0] not in selected:
            selected.append(candidates[0])
        if category == "business":
            targeted = max(candidates, key=lambda item: len(item.get("task_matches", [])), default=None)
            if targeted and targeted.get("task_matches") and targeted not in selected:
                selected.append(targeted)
    for row in rows:
        if row not in selected:
            selected.append(row)
    return selected[:limit]


LONG_INSTRUCTIONS = (
    "本轮为长期投资：形成约三年投资论证，252交易日预测仅作一年阶段观察，按季度复核，期限届满不自动卖出。"
    "业务、现金流/负债/盈利质量、竞争和估值为主；价格历史只辅助买入节奏与风险。"
    "新闻只判断原论证是否改变，没有新新闻不代表没有投资价值。"
    "将受控候选和已有持仓按长期收益依据、估值、风险及证据充分性比较，同时说明继续持有现金的理由。"
    "可以分批配置多家公司，仓位表达判断强弱；不能为凑持股数量强迫买入。核心事实缺失与正常的未来不确定性须区分。"
    "必须区分研究未完成(research_pending)、论证可用但价格不合适(wait_price)、建仓(build)、"
    "继续持有(hold)、减仓(reduce)、退出(exit)。辅助数据缺失记non_core_gaps，不得一概否决。"
    "核心业务/财务/估值缺口需明确what_to_fetch及重新评估条件，不能靠含糊HOLD无限拖延。"
    "缺口不得编造补齐；有据的风险退出不要求先补齐全部增长材料。不要为了增加交易而强迫买入。"
    "旧投资论证是模型观点，不是外部证据；结合当时可用资料更新，只有实质变化才修改论证。"
    "只表述自己实际角色的结论，分析师不输出经理或风控已作出决定的说法。"
)


def result_schema(base):
    value = deepcopy(base)
    value["properties"]["investment_policy"] = {"type": "string", "enum": [VERSION]}
    value["properties"]["predictions"]["items"]["properties"]["horizon_days"] = {"type": "integer", "enum": [252]}
    from agent.evidence import LINK_SCHEMA
    text = {"type": "string"}
    number = {"type": ["number", "null"]}
    gap = {"type": "object", "additionalProperties": False, "properties": {
        "category": {"type": "string", "enum": ["business", "financials", "valuation"]},
        "reason": text, "what_to_fetch": text}, "required": ["category", "reason", "what_to_fetch"]}
    valuation = {"type": "object", "additionalProperties": False, "properties": {
        "method": text, "bear_price": number, "base_price": number, "bull_price": number, "assumptions": text},
        "required": ["method", "bear_price", "base_price", "bull_price", "assumptions"]}
    props = {"symbol": text, "research_status": {"type": "string", "enum": ["pending", "ready"]},
        "position_intent": {"type": "string", "enum": ["research_pending", "wait_price", "build", "hold", "reduce", "exit"]},
        "thesis": text, "thesis_horizon_years": {"type": "integer", "enum": [3]},
        "thesis_change": {"type": "string", "enum": ["new", "unchanged", "updated", "invalidated"]},
        "target_weight": {"type": "number"}, "valuation": valuation, "entry_condition": text,
        "exit_condition": text, "next_review_date": text, "core_gaps": {"type": "array", "items": gap},
        "non_core_gaps": {"type": "array", "items": text}, "evidence": deepcopy(LINK_SCHEMA)}
    value["properties"]["investment_cases"] = {"type": "array", "items": {"type": "object", "additionalProperties": False,
        "properties": props, "required": list(props)}}
    value["required"] += ["investment_policy", "investment_cases"]
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + "不能为空")


def validate_cases(result, symbols, *, catalog=None, as_of=None, account=None, prices=None):
    if result.get("investment_policy") != VERSION:
        raise ValueError("长期输出缺少正确策略协议")
    cases = result.get("investment_cases")
    if not isinstance(cases, list) or len(cases) != len(symbols) or {row.get("symbol") for row in cases} != set(symbols):
        raise ValueError("长期投资论证必须逐只覆盖股票且不能重复")
    proposed = result.get("risk_review", {}).get("proposed_decisions") or result["decisions"]
    plans = {row["symbol"]: row for row in proposed}
    for row in cases:
        if row.get("research_status") not in {"pending", "ready"} or row.get("position_intent") not in {"research_pending", "wait_price", "build", "hold", "reduce", "exit"}:
            raise ValueError("长期研究状态或持仓意图非法")
        if row.get("thesis_horizon_years") != 3:
            raise ValueError("长期论证须声明三年观察框架")
        for key in ("thesis", "entry_condition", "exit_condition"):
            _text(row.get(key), key)
        if row.get("thesis_change") not in {"new", "unchanged", "updated", "invalidated"}:
            raise ValueError("缺少投资论证变化状态")
        weight = row.get("target_weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= MAX_POSITION_PCT:
            raise ValueError("长期目标仓位超出宿主上限")
        review = date.fromisoformat(row.get("next_review_date", ""))
        if as_of and not date.fromisoformat(as_of) < review <= date.fromisoformat(as_of) + timedelta(days=90):
            raise ValueError("长期复核日期必须在未来且不超过90天")
        gaps = row.get("core_gaps")
        if not isinstance(gaps, list) or not isinstance(row.get("non_core_gaps"), list):
            raise ValueError("核心/辅助缺口必须分开记录")
        for gap in gaps:
            if gap.get("category") not in {"business", "financials", "valuation"}:
                raise ValueError("未知核心补证类别")
            _text(gap.get("reason"), "缺口原因")
            _text(gap.get("what_to_fetch"), "具体补证任务")
        if row["research_status"] == "pending" and not gaps:
            raise ValueError("研究未完成必须有具体核心补证任务")
        if row["research_status"] == "pending" and as_of and review > date.fromisoformat(as_of) + timedelta(days=7):
            raise ValueError("核心补证须在7天内复核，不能按季度无限等待")
        if row["research_status"] == "ready" and gaps:
            raise ValueError("核心缺口未解决不能声称研究完成")
        plan, intent = plans[row["symbol"]], row["position_intent"]
        allowed = {"research_pending": "HOLD", "wait_price": "HOLD", "hold": "HOLD", "build": "BUY", "reduce": "SELL", "exit": "SELL"}
        if plan["action"] != allowed[intent]:
            raise ValueError("长期持仓意图与送审计划不一致")
        valuation = row.get("valuation", {})
        _text(valuation.get("method"), "估值方法或不可估值原因")
        _text(valuation.get("assumptions"), "估值假设或资料缺口")
        values = [valuation.get(key) for key in ("bear_price", "base_price", "bull_price")]
        if any(value is not None for value in values):
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in values) or values != sorted(values):
                raise ValueError("悲观/基准/乐观估值必须有限、正数并有序")
        if intent in {"build", "wait_price"} and (row["research_status"] != "ready" or any(value is None for value in values)):
            raise ValueError("建仓/等待价格必须已有完整研究与估值依据")
        if intent == "build" and weight <= 0:
            raise ValueError("建仓必须有正目标仓位")
        if intent == "exit" and weight != 0:
            raise ValueError("退出目标仓位必须为0")
        if prices and row["symbol"] in prices and abs(plan["price"] - prices[row["symbol"]]) > 1e-6:
            raise ValueError("送审价格须为冻结行情价，不能把估值目标当成交价")
        if account and plan["action"] == "BUY":
            held = next((item["shares"] for item in account.get("positions", []) if item["symbol"] == row["symbol"]), 0)
            exposure = (held + plan["shares"]) * plan["price"] / account["total_assets"]
            if exposure > weight + 1e-8:
                raise ValueError("送审买入股数超过长期目标仓位")
        if catalog:
            from agent.evidence import validate_links
            validate_links(row.get("evidence"), catalog, row["symbol"])
            if intent in {"build", "wait_price"}:
                records = {item["evidence_id"]: item for item in catalog["records"]}
                categories = {category for ref in row["evidence"]["supporting_ids"] for category in
                    records[ref]["payload"].get("categories", []) if records[ref]["kind"] == "company_document"}
                if not {"business", "financials"} <= categories:
                    raise ValueError("建仓/估值等待缺少业务与财务原始披露证据")


class ResearchBook:
    def __init__(self, path=RESEARCH_DB):
        self.path = Path(path)

    @contextmanager
    def db(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS materials(id TEXT PRIMARY KEY,symbol TEXT,published_at TEXT,first_seen_at TEXT,expires_at TEXT,payload TEXT);
            CREATE TABLE IF NOT EXISTS cases(id TEXT PRIMARY KEY,symbol TEXT,day TEXT,variant TEXT,log_hash TEXT,payload TEXT,UNIQUE(symbol,day,variant));
            CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,symbol TEXT,category TEXT,opened_at TEXT,updated_at TEXT,status TEXT,payload TEXT);
            """)
            yield conn
            conn.commit()

    def add_material(self, row):
        value = deepcopy(row)
        if value.get("symbol") not in {item["symbol"] for item in MARKETS["cn"]["candidate_universe"]}:
            raise ValueError("研究材料必须属于受控A股")
        if not controlled_url(value.get("url")) or not set(value.get("categories", [])) <= CATEGORIES or not value.get("categories"):
            raise ValueError("只接受受控披露来源和明确材料类别")
        for key in ("title", "excerpt", "locator", "published_at"):
            _text(value.get(key), key)
        if len(value["excerpt"]) > 8000:
            raise ValueError("只保存相关片段，每份不得超过8000字")
        published = event_time(value["published_at"], "cn")
        now = datetime.now(timezone.utc)
        if published > now:
            raise ValueError("披露时间不能在未来")
        expires = value.get("expires_at") or (published.date() + timedelta(days=400)).isoformat()
        if not published.date() < date.fromisoformat(expires) <= published.date() + timedelta(days=550):
            raise ValueError("材料须有有限有效期，不允许永久有效")
        value.update(expires_at=expires, data_source="受控公司披露片段(导入)", data_quality="imported",
                     source_truth_verified=False, first_seen_at=now.isoformat())
        identifier = "material_" + canonical_hash({key: val for key, val in value.items() if key != "first_seen_at"})[:24]
        value["material_id"] = identifier
        with self.db() as conn:
            existing = conn.execute("SELECT payload FROM materials WHERE id=?", (identifier,)).fetchone()
            if existing:
                return json.loads(existing[0])
            conn.execute("INSERT INTO materials VALUES(?,?,?,?,?,?)", (identifier, value["symbol"], value["published_at"], value["first_seen_at"], expires, json.dumps(value, ensure_ascii=False)))
        return value

    def snapshot(self, symbols, day, variant=None, information_cutoff_at=None):
        cutoff = event_time(information_cutoff_at, "cn") if information_cutoff_at else market_close(day, "cn")
        materials, previous, tasks = [], [], []
        with self.db() as conn:
            for symbol in symbols:
                rows = conn.execute("SELECT payload FROM materials WHERE symbol=? ORDER BY published_at DESC,id", (symbol,)).fetchall()
                available = [json.loads(row[0]) for row in rows]
                available = [row for row in available if event_time(row["published_at"], "cn") <= cutoff and row["expires_at"] >= day]
                if information_cutoff_at:
                    available = [row for row in available if event_time(row["first_seen_at"], "cn") <= cutoff]
                    from agent.disclosures import VERSION as CAPTURE_VERSION
                    available = [row for row in available if not row.get("pdf_sha256") or row.get("capture_protocol") == CAPTURE_VERSION]
                available = select_materials(available, 8)
                materials += available[:8]
                query = "SELECT payload,day,log_hash FROM cases WHERE symbol=? AND day<?"
                params = [symbol, day]
                if variant is not None:
                    query += " AND variant=?"
                    params.append(variant)
                old = conn.execute(query + " ORDER BY day DESC,rowid DESC LIMIT 1", params).fetchone()
                if old:
                    previous.append({"case": json.loads(old[0]), "as_of_date": old[1], "source_log_hash": old[2], "kind": "past_model_opinion_not_external_fact"})
                categories = {category for row in available[:8] for category in row["categories"]}
                for category in ("business", "financials"):
                    task_id = "task_" + canonical_hash([symbol, category])[:24]
                    status = "source_available" if category in categories else "open"
                    previous_task = conn.execute("SELECT opened_at FROM tasks WHERE id=?", (task_id,)).fetchone()
                    opened = previous_task[0] if previous_task else day
                    due = (date.fromisoformat(opened) + timedelta(days=7)).isoformat()
                    if status == "open" and day > due:
                        status = "overdue"
                    task = {"task_id": task_id, "symbol": symbol, "category": category, "status": status,
                        "what_to_fetch": ("获取业务模式、竞争优势及主要风险的年报原文片段" if category == "business" else "获取多年度营收/利润/现金流/负债及最新季报片段"),
                        "required_metadata": ["披露时间", "报告期", "原始链接", "页码或段落", "相关文字"],
                        "review_by": due}
                    conn.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at,status=excluded.status,payload=excluded.payload",
                        (task_id, symbol, category, day, day, status, json.dumps(task, ensure_ascii=False)))
                    tasks.append(task)
                for old_task in conn.execute("SELECT payload,status FROM tasks WHERE symbol=? AND id LIKE 'gap_%' AND status!='resolved' AND updated_at<=?", (symbol, day)):
                    task = json.loads(old_task[0])
                    task["status"] = "overdue" if day > task["review_by"] else old_task[1]
                    tasks.append(task)
        return {"policy": POLICY, "as_of_date": day, "materials": materials, "previous_cases": previous, "tasks": tasks,
                "visibility_policy": "published_and_first_seen_by_information_cutoff; imports_not_independently_verified" if information_cutoff_at else "published_by_close; first_seen_retained; imports_not_independently_verified"}

    def save(self, log, variant):
        from agent.contracts import validate_decision_result
        from agent.evidence import validate_evidence_audit
        valid_mode = log.get("execution_mode") in {"shadow", "paper"}
        if log.get("execution_mode") == "pending" and log.get("timing", {}).get("version") == "after-close-next-session-v1":
            from agent.order_queue import OrderQueue
            valid_mode = any(row["id"] == log.get("queue_plan_hash") for row in OrderQueue().rows())
        if log.get("market") != "cn" or not valid_mode:
            raise ValueError("研究论证仅接收A股原始shadow/模拟盘日志")
        validate_decision_result(log, "cn", expected_symbols={row["symbol"] for row in log["decisions"]})
        validate_evidence_audit(log)
        if log["evidence_audit"]["catalog"]["as_of_date"] != log["as_of_date"]:
            raise ValueError("论证日期与冻结证据日期不同")
        if not isinstance(log.get("input_dataset"), dict) or canonical_hash(log["input_dataset"]) != log.get("dataset_fingerprint"):
            raise ValueError("长期论证缺少一致的冻结输入")
        validate_cases(log, {row["symbol"] for row in log["decisions"]}, catalog=log["evidence_audit"]["catalog"], as_of=log["as_of_date"])
        with self.db() as conn:
            for row in log["investment_cases"]:
                prior = conn.execute("SELECT log_hash FROM cases WHERE symbol=? AND day=? AND variant=?", (row["symbol"], log["as_of_date"], variant)).fetchone()
                if prior and prior[0] != canonical_hash(log):
                    raise ValueError("长期论证同版本同日只保留首次输出")
                conn.execute("INSERT OR IGNORE INTO cases VALUES(?,?,?,?,?,?)", ("case_" + canonical_hash([row["symbol"], log["as_of_date"], variant])[:24], row["symbol"], log["as_of_date"], variant, canonical_hash(log), json.dumps(row, ensure_ascii=False)))
                old_tasks = {item["id"]: dict(item) for item in conn.execute("SELECT * FROM tasks WHERE symbol=? AND id LIKE 'gap_%'", (row["symbol"],))}
                conn.execute("UPDATE tasks SET status='resolved',updated_at=? WHERE symbol=? AND id LIKE 'gap_%'", (log["as_of_date"], row["symbol"]))
                for gap in row["core_gaps"]:
                    task_id = "gap_" + canonical_hash([row["symbol"], gap["category"]])[:24]
                    payload = {**gap, "task_id": task_id, "symbol": row["symbol"], "status": "open",
                        "review_by": row["next_review_date"], "source_case_hash": canonical_hash(row)}
                    if task_id in old_tasks and old_tasks[task_id]["status"] != "resolved":
                        payload["review_by"] = min(payload["review_by"], json.loads(old_tasks[task_id]["payload"])["review_by"])
                    if log["as_of_date"] > payload["review_by"]:
                        payload["status"] = "overdue"
                    conn.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at,status=excluded.status,payload=excluded.payload",
                        (task_id, row["symbol"], gap["category"], log["as_of_date"], log["as_of_date"], payload["status"], json.dumps(payload, ensure_ascii=False)))

    def tasks(self):
        if not self.path.exists():
            return []
        with closing(sqlite3.connect(f"file:{self.path.resolve().as_posix()}?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            return [{**json.loads(row["payload"]), "opened_at": row["opened_at"], "updated_at": row["updated_at"], "status": row["status"]}
                for row in conn.execute("SELECT * FROM tasks ORDER BY symbol,category,id")]

    def review_order(self, symbols, day, logs=()):
        """Unreviewed companies first, then oldest review; no new model request."""
        last = {}
        if self.path.exists():
            with closing(sqlite3.connect(f"file:{self.path.resolve().as_posix()}?mode=ro", uri=True)) as conn:
                last = dict(conn.execute("SELECT symbol,MAX(day) FROM cases WHERE day<=? GROUP BY symbol", (day,)))
        from evals.metrics import uses_synthetic_data
        for log in logs:
            stamp = (log.get("timing", {}).get("decision_at") or log.get("generated_at") or log.get("as_of_date") or log.get("date", ""))[:10]
            if not stamp or stamp > day or uses_synthetic_data(log):
                continue
            for row in log.get("decisions", []):
                last[row["symbol"]] = max(last.get(row["symbol"], ""), stamp)
        allowed = {row["symbol"] for row in MARKETS["cn"]["candidate_universe"]}
        return sorted((set(symbols) | set(last)) & allowed, key=lambda symbol: (last.get(symbol, ""), symbol))


async def prepare_context(host, day, symbols, variant=None, book=None):
    """先固定资料，再调用模型；历史优先复用可核验本地缓存。"""
    from evals.price_data import normalize_history
    import pandas as pd
    research = (book or ResearchBook()).snapshot(symbols, day, variant, getattr(host, "timing", {}).get("information_cutoff_at"))
    histories, gaps = [], []
    cache_path = DATA_DIR / "market_views" / "cn-prices.json"
    cached = {}
    if cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        if raw.get("market") == "cn" and raw.get("data_quality") == "live_prices" and canonical_hash(raw.get("histories", {})) == raw.get("fingerprint"):
            cached = raw.get("histories", {})
    synthetic = bool(getattr(host, "news_screen", {}).get("offline"))
    if synthetic:
        cached = {}
    for symbol in symbols:
        rows = [row for row in cached.get(symbol, []) if row.get("date", "") <= day]
        try:
            if not rows or rows[-1]["date"] != day:
                import asyncio
                rows = await asyncio.wait_for(host.call("get_history", {"symbol": symbol, "days": 756}), timeout=30)
            frame = normalize_history(symbol, pd.DataFrame(rows))
            frame = frame[frame["date"] <= day].tail(756)
            if frame.empty or frame.iloc[-1]["date"] != day:
                raise ValueError("历史价格未覆盖分析日")
            histories.append({"symbol": symbol, "date": day, "data_source": "冻结模拟价格历史" if synthetic else "冻结真实价格历史", "data_quality": "synthetic" if synthetic else "provider_observation",
                "start_date": frame.iloc[0]["date"], "observations": len(frame),
                "monthly_observations": frame.iloc[::21][["date", "close", "volume"]].tail(36).to_dict("records"),
                "recent_daily": frame.tail(10).to_dict("records"), "history_hash": canonical_hash(frame.to_dict("records"))})
            if len(frame) < 504:
                gaps.append({"symbol": symbol, "kind": "price_history", "severity": "auxiliary", "reason": f"当前仅{len(frame)}个交易日，未覆盖两年以上"})
        except Exception as exc:
            from agent.errors import safe_error
            gaps.append({"symbol": symbol, "kind": "price_history", "severity": "auxiliary", "reason": safe_error(exc)})
    research.update(price_history=histories, auxiliary_gaps=gaps)
    return research
