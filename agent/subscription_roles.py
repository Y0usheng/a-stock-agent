"""订阅四角色输入：公开市场字段 + 本项目虚拟账本的固定数值字段。"""
from __future__ import annotations

from copy import deepcopy
import json
import re

from agent.sdk_runner import RESULT_SCHEMA, build_agent_definitions
from agent.subscription import INSTRUCTIONS, PROTOCOL_VERSION, SubscriptionClient
from config import MARKETS
from evals.experiments import canonical_hash
from agent.evidence import (CLAIM_SCHEMA, EVIDENCE_VERSION, build_catalog, cited_schema,
                            validate_claims, validate_result_evidence, coherent_link_schema)
from agent.investment import LONG_INSTRUCTIONS, is_long, result_schema, validate_cases, select_materials

EVIDENCE_INSTRUCTIONS = (
    "宿主已提供本轮可用数据，不调用工具。缺失字段注明不足。分析摘要每只股票不超过两句；"
    "经理与风控理由简短。每条专家结论、决策、预测及风控评审必须带 evidence："
    "supporting_ids/contradicting_ids 只引用本次提供且同股票的 evidence_id，"
    "不得把新闻情绪分、历史模型预测或自己的分析当作外部事实。"
    "有支持依据用 supported；没有则用 insufficient、空支持列表并说明缺口，不能编 ID。"
    "没有反证可留空，但必须给出具体 invalidation_condition。"
    "专家 claims 逐只覆盖股票池；证据不足时经理只能 HOLD。"
    "公告标题/摘要不代表已阅读全文，财报报告期不代表披露时间。只输出指定 JSON。"
)

SUMMARY_SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}},
                  "required": ["summary"], "additionalProperties": False}
SUMMARY_SCHEMA["properties"]["claims"] = {"type": "array", "items": deepcopy(CLAIM_SCHEMA)}
SUMMARY_SCHEMA["required"].append("claims")
RISK_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {"summary": {"type": "string"}, "assessments": {
        "type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": {"symbol": {"type": "string"},
                "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                "verdict": {"type": "string", "enum": ["APPROVE", "REJECT", "RESIZE"]},
                "max_shares": {"type": "integer"}, "reason": {"type": "string"}},
            "required": ["symbol", "action", "verdict", "max_shares", "reason"]}}},
    "required": ["summary", "assessments"],
}
RESULT_EVIDENCE_SCHEMA = cited_schema(RESULT_SCHEMA, ("decisions", "predictions"))
RISK_SCHEMA = cited_schema(RISK_SCHEMA, ("assessments",))
SNAPSHOT_FIELDS = ("symbol", "name", "date", "close", "open", "high", "low", "volume",
    "change_pct", "ma5", "ma20", "ma60", "rsi14", "macd_dif", "macd_dea", "above_ma20", "data_source")
FUNDAMENTAL_FIELDS = ("symbol", "name", "as_of_date", "report_date", "data_source",
    "revenue_growth_pct", "profit_growth_pct", "net_profit_growth_pct", "net_income_growth_pct", "roe_pct", "gross_margin_pct",
    "net_margin_pct", "debt_to_equity_pct", "pe_ttm", "price_to_book", "free_cash_flow", "market_cap")
EVENT_FIELDS = ("event_id", "symbol", "title", "summary", "source", "url", "published_at",
                "data_quality", "content_hash", "first_seen_at", "ingested_at", "collected_at",
                "event_occurred_at")
SNAPSHOT_FIELDS += ("collected_at",)
FUNDAMENTAL_FIELDS += ("published_at", "collected_at",)


def _pick(row: dict, keys: tuple, text_limit: int = 300) -> dict:
    return {key: (row[key][:text_limit] if isinstance(row[key], str) else deepcopy(row[key]))
            for key in keys if key in row and (row[key] is None or isinstance(row[key], (str, int, float, bool)))}


def export_paper_context(context: dict, market: str, long_term=False) -> dict:
    """禁止通用账户对象、自由文本记忆及无关字段进入订阅输入。"""
    account = context["account"]
    if (account.get("market") != market or account.get("currency") != MARKETS[market]["currency"]
            or account.get("initial_cash") != MARKETS[market]["initial_cash"]):
        raise ValueError("订阅输入仅支持本项目生成的虚拟模拟账本")
    paper = _pick(account, ("cash", "total_assets", "initial_cash"))
    paper["kind"] = "local_virtual_paper_ledger"
    paper["positions"] = [_pick(row, ("symbol", "shares", "avg_cost", "last_buy_date"))
                          for row in account.get("positions", [])]
    news = []
    for candidate in context.get("news", {}).get("candidates", []):
        if candidate.get("symbol") not in context["symbols"]:
            continue
        item = _pick(candidate, ("symbol", "article_count"))
        # 链接/哈希/时间戳不得按正文长度截断。
        item["top_events"] = [_pick(row, EVENT_FIELDS, text_limit=1000) for row in candidate.get("top_events", [])[:2]]
        news.append(item)
    # 仅保留以前模型的方向/概率，无历史理由、成交备注或用户自由文本。
    memory = context.get("memory", {})
    past_predictions = [_pick(row, ("date", "symbol", "direction", "confidence", "horizon_days"))
                        for row in memory.get("recent_predictions", [])[-12:]]
    result = {"date": context["date"], "symbols": list(context["symbols"]),
        "snapshot": [_pick(row, SNAPSHOT_FIELDS) for row in context["snapshot"]],
        "fundamentals": [_pick(row, FUNDAMENTAL_FIELDS) for row in context["fundamentals"]],
        "news": news, "paper_account": paper, "past_predictions": past_predictions}
    if long_term:
        research = context.get("investment_context", {})
        if context.get("timing"):
            result["timing"] = deepcopy(context["timing"])
        selected = []
        for symbol in context["symbols"]:
            material_rows = [row for row in research.get("materials", []) if row["symbol"] == symbol]
            for row in select_materials(material_rows, 4):
                text_limit = 1400 if context.get("timing") else 1200
                item = _pick(row, ("symbol", "material_id", "title", "url", "published_at", "first_seen_at", "expires_at", "data_source", "data_quality", "locator", "report_period", "excerpt", "pdf_sha256", "page_text_hash", "extraction_status", "page_excerpt_complete", "document_type", "capture_protocol", "unit_context_present"), text_limit=text_limit)
                item["categories"] = [category for category in row["categories"] if category in {"business", "financials", "industry"}]
                item["source_truth_verified"] = False
                item["excerpt_truncated"] = len(row.get("excerpt", "")) > text_limit
                selected.append(item)
        result.update(research_materials=selected, price_history=deepcopy(research.get("price_history", [])),
            previous_cases=deepcopy(research.get("previous_cases", [])), research_tasks=deepcopy(research.get("tasks", [])),
            auxiliary_gaps=deepcopy(research.get("auxiliary_gaps", [])), investment_policy=deepcopy(research.get("policy", {})))
        from agent.opportunities import compact_context
        result["opportunity_context"] = compact_context(context.get("news", {}).get("theme_discovery"), context["symbols"])
    return result


def subscription_manifest(manifest: dict) -> dict:
    value = deepcopy(manifest)
    value["template_sha256"] = canonical_hash({"original": value["template_sha256"],
        "protocol": PROTOCOL_VERSION, "instructions": INSTRUCTIONS,
        "fields": [SNAPSHOT_FIELDS, FUNDAMENTAL_FIELDS, EVENT_FIELDS],
        "evidence_version": EVIDENCE_VERSION,
        "evidence_instructions": EVIDENCE_INSTRUCTIONS,
        "input_projection": "compact-evidence-data-v1",
        "schemas": [SUMMARY_SCHEMA, RESULT_EVIDENCE_SCHEMA, RISK_SCHEMA]})
    value["subscription_protocol"] = PROTOCOL_VERSION
    if is_long(value.get("version")):
        value["template_sha256"] = canonical_hash({"base": value["template_sha256"],
            "long_instructions": LONG_INSTRUCTIONS, "schema": result_schema(RESULT_EVIDENCE_SCHEMA),
            "projection": "long-research-and-news-opportunities-v2"})
    if value.get("version") == "manager-long-v2":
        value["template_sha256"] = canonical_hash({"base": value["template_sha256"],
            "coherent_evidence_schemas": [coherent_link_schema(schema) for schema in (SUMMARY_SCHEMA, result_schema(RESULT_EVIDENCE_SCHEMA), RISK_SCHEMA)]})
    return value


class SubscriptionRoles:
    def __init__(self, context: dict, market: str, manager_prompt: str, model: str, *, allow_synthetic=False, long_term=False):
        self.client = SubscriptionClient(model)
        self.long_term = long_term
        self.context = export_paper_context(context, market, long_term)
        self.definitions = build_agent_definitions()
        self.manager_prompt = manager_prompt
        self.catalog = build_catalog(self.context, market, allow_synthetic=allow_synthetic)
        self.claims_by_role = {}
        self.raw_outputs = {}

    def audit(self):
        return {"catalog": deepcopy(self.catalog), "analyst_claims": deepcopy(self.claims_by_role),
                "validation": "id_content_scope_symbol_and_publication_cutoff",
                "semantic_support_verified": False}

    async def call(self, name: str, input: str) -> str:
        context = self.context
        kinds = ({"market_snapshot"} if name == "technical-analyst" else
                 {"fundamentals", "news"} if name == "fundamental-analyst" else
                 {"market_snapshot", "fundamentals", "news"})
        if self.long_term:
            kinds |= ({"price_history"} if name == "technical-analyst" else {"company_document", "market_snapshot"} if name == "fundamental-analyst" else {"company_document", "price_history"})
        records = [record for record in self.catalog["records"] if record["kind"] in kinds]
        payload = {"date": context["date"], "symbols": context["symbols"], "input": input,
            "evidence": [{"evidence_id": record["evidence_id"], "kind": record["kind"],
                "source_tier": record["source_tier"], "content_scope": record["content_scope"],
                "data_quality": record["data_quality"], "issues": record["issues"],
                "data": {key: value for key, value in record["payload"].items()
                         if key not in {"ingested_at", "collected_at", "content_hash"}}}
                for record in records if record["eligible"]],
            "unavailable_evidence": [{key: record[key] for key in ("evidence_id", "symbol", "kind", "issues")}
                                      for record in records if not record["eligible"]]}
        if name in {"investment-manager", "risk-manager"}:
            payload["paper_account"] = context["paper_account"]
            payload["past_predictions"] = context["past_predictions"]
        if self.long_term:
            payload.update(previous_cases=context["previous_cases"], research_tasks=context["research_tasks"],
                           auxiliary_gaps=context["auxiliary_gaps"], investment_policy=context["investment_policy"])
            payload["opportunity_context"] = context["opportunity_context"]
            payload["opportunity_boundary"] = "主题仅为前序模型提出的待核验假设；不可作为公司事实或证据ID引用，不能替代公司披露、估值、反证及建仓条件。无新新闻不等于论证失效。"
            if context.get("timing"):
                payload["timing"] = context["timing"]
                payload["timing_boundary"] = "行情日不等于决策时间。只使用information_cutoff_at前已观察资料；本轮只提出未来交易日收盘的限价模拟计划，参考价仍填冻结行情价。未确认、超时或超限不成交；估值和研究完成不等于已买入。"
        prompt = self.manager_prompt if name == "investment-manager" else self.definitions[name].prompt
        instructions = EVIDENCE_INSTRUCTIONS
        if self.long_term:
            instructions = instructions.replace("证据不足时经理只能 HOLD。", "关键交易依据不足时不得新增仓位；辅助缺口单独披露，不一概否决。") + LONG_INSTRUCTIONS
            if name in {"technical-analyst", "fundamental-analyst"}:
                instructions += "你只输出本角色研究结论，不使用经理：、风控：等角色归属前缀，不宣称任何审批状态。"
        prompt += ("\n" + instructions + "\n" +
                   json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        schema = RESULT_EVIDENCE_SCHEMA if name == "investment-manager" else RISK_SCHEMA if name == "risk-manager" else SUMMARY_SCHEMA
        if self.long_term and name == "investment-manager":
            schema = result_schema(schema)
        if context.get("timing"):
            schema = coherent_link_schema(schema)
        result = await self.client.complete(prompt, schema, role=name)
        self.raw_outputs[name] = deepcopy(result)
        if name in {"technical-analyst", "fundamental-analyst"}:
            summary = result.get("summary")
            if not isinstance(summary, str) or not summary.strip() or len(summary) > 6000:
                raise RuntimeError("订阅分析摘要缺失或过长")
            validate_claims(result.get("claims"), self.catalog, context["symbols"], kinds)
            if self.long_term and any(re.search(r"(?:^|[。；\n])\s*(?:经理(?:决策)?|投资经理|风控(?:评审)?)\s*[：:]", row["claim"]) for row in result["claims"]):
                raise RuntimeError("长期分析师输出冒用经理/风控身份，停止采用该结论")
            self.claims_by_role[name] = deepcopy(result["claims"])
            return json.dumps(result, ensure_ascii=False)
        validate_result_evidence(result, self.catalog, context["symbols"], risk=name == "risk-manager")
        if self.long_term and name == "investment-manager":
            validate_cases(result, set(context["symbols"]), catalog=self.catalog, as_of=context["date"],
                account=context["paper_account"], prices={row["symbol"]: row["close"] for row in context["snapshot"]})
        return json.dumps(result, ensure_ascii=False)
