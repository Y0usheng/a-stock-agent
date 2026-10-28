"""冻结证据、逐条引用与宿主校验；不把引用存在等同于结论正确。"""
from copy import deepcopy
from datetime import date
from urllib.parse import urlsplit

from evals.experiments import canonical_hash
from servers.data_quality import event_time, market_close

EVIDENCE_VERSION = "frozen-evidence-v1"
LINK_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "status": {"type": "string", "enum": ["supported", "insufficient"]},
    "supporting_ids": {"type": "array", "items": {"type": "string"}},
    "contradicting_ids": {"type": "array", "items": {"type": "string"}},
    "invalidation_condition": {"type": "string"}},
    "required": ["status", "supporting_ids", "contradicting_ids", "invalidation_condition"]}
CLAIM_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "symbol": {"type": "string"}, "claim": {"type": "string"}, "evidence": LINK_SCHEMA},
    "required": ["symbol", "claim", "evidence"]}


def cited_schema(schema, collections):
    schema = deepcopy(schema)
    for name in collections:
        item = schema["properties"][name]["items"]
        item["properties"]["evidence"] = deepcopy(LINK_SCHEMA)
        item["required"].append("evidence")
    return schema


def _quality(row):
    source = str(row.get("source") or row.get("data_source") or "")
    if row.get("data_quality") == "synthetic" or any(
            word in source.lower() for word in ("mock", "synthetic", "离线", "模拟")):
        return "synthetic"
    if not source.strip() or any(word in source.lower() for word in ("unavailable", "unknown", "不可用")):
        return "unavailable"
    return row.get("data_quality", "provider_observation")


def _record(kind, row, market, day, allow_synthetic):
    payload = deepcopy(row)
    source = str(row.get("source") or row.get("data_source") or "").strip()
    url = str(row.get("url") or "")
    quality = _quality(row)
    issues, blockers = [], []
    published = row.get("published_at")
    if quality == "unavailable":
        blockers.append("missing_or_unavailable_source")
    if quality == "synthetic":
        issues.append("synthetic_test_data")
        if not allow_synthetic:
            blockers.append("synthetic_in_live_context")
    timestamp = published if kind in {"news", "fundamentals", "company_document"} else row.get("date")
    if kind in {"market_snapshot", "price_history"}:
        try:
            available = market_close(str(timestamp), market)
        except (ValueError, TypeError):
            available = None
    elif timestamp:
        try:
            available = event_time(timestamp, market)
            if len(timestamp) == 10:
                issues.append("date_only_usable_after_publication_day")
        except (ValueError, TypeError):
            available = None
    else:
        available = None
    if available is None and not (quality == "synthetic" and allow_synthetic):
        blockers.append("unknown_publication_time" if kind != "market_snapshot" else "unknown_quote_date")
    if available is not None and available > market_close(day, market):
        blockers.append("after_as_of_close")
    if kind in {"news", "company_document"} and quality != "synthetic":
        link = urlsplit(url)
        if link.scheme not in {"http", "https"} or not link.hostname:
            blockers.append("missing_or_invalid_source_url")
        if kind == "company_document":
            from agent.investment import controlled_url
            if not controlled_url(url):
                blockers.append("uncontrolled_document_source")
            if row.get("excerpt_truncated"):
                issues.append("excerpt_truncated_check_units_and_context")
    # 报告期/抓取日期都不能替代真实财报披露时间。
    if kind == "fundamentals" and not published and quality != "synthetic":
        issues.append("report_date_is_not_publication_time")
    first_seen = row.get("first_seen_at") or row.get("ingested_at")
    if kind == "news" and not first_seen:
        issues.append("first_seen_time_unknown")
    if first_seen:
        try:
            if event_time(first_seen, market) > market_close(day, market):
                issues.append("observed_after_as_of_close")
        except (ValueError, TypeError):
            issues.append("invalid_first_seen_time")
    host = (urlsplit(url).hostname or "").lower()
    disclosure = (host == "cninfo.com.cn" or host.endswith(".cninfo.com.cn") or
                  host == "sse.com.cn" or host.endswith(".sse.com.cn") or
                  host == "szse.cn" or host.endswith(".szse.cn") or
                  host == "sec.gov" or host.endswith(".sec.gov"))
    # 获取时间保留用于审计，但不使相同内容的证据 ID 随抓取次数改变。
    identity_payload = {key: value for key, value in payload.items()
                        if key not in {"collected_at", "first_seen_at", "ingested_at"}}
    content_hash = canonical_hash({"kind": kind, "market": market, "payload": identity_payload})
    return {"evidence_id": "ev_" + content_hash[:24], "kind": kind, "market": market,
        "symbol": row.get("symbol"), "as_of_date": day, "source": source, "url": url,
        "source_tier": "disclosure_reference" if disclosure and kind in {"news", "company_document"} else
                       "media_report" if kind == "news" else "provider_values",
        "content_scope": "title_and_excerpt" if kind == "news" else "frozen_fields",
        "published_at": published, "event_occurred_at": row.get("event_occurred_at"),
        "collected_at": row.get("collected_at"), "first_seen_at": first_seen,
        "available_at": available.isoformat() if available else None,
        "source_content_hash": row.get("content_hash"), "content_hash": content_hash,
        "data_quality": quality, "eligible": not blockers,
        "issues": sorted(set(issues + blockers)), "payload": payload}


def build_catalog(context, market, *, allow_synthetic=False):
    day = date.fromisoformat(context["date"]).isoformat()
    symbols = set(context["symbols"])
    rows = []
    for kind, entries in (("market_snapshot", context["snapshot"]),
                          ("fundamentals", context["fundamentals"]),
                          ("company_document", context.get("research_materials", [])),
                          ("price_history", context.get("price_history", [])),
                          ("news", [event for candidate in context.get("news", [])
                                    for event in candidate.get("top_events", [])])):
        for row in entries:
            if row.get("symbol") in symbols:
                rows.append(_record(kind, row, market, day, allow_synthetic))
    rows = sorted({row["evidence_id"]: row for row in rows}.values(), key=lambda row: row["evidence_id"])
    return {"version": EVIDENCE_VERSION, "market": market, "as_of_date": day,
        "visibility_policy": "publication_by_close; later_collection_is_explicitly_flagged",
        "semantic_support_verified": False, "allow_synthetic": allow_synthetic,
        "records": rows, "fingerprint": canonical_hash(rows)}


def validate_catalog(catalog):
    if catalog.get("version") != EVIDENCE_VERSION or canonical_hash(catalog["records"]) != catalog.get("fingerprint"):
        raise ValueError("证据目录版本或指纹不一致")
    ids = set()
    for record in catalog["records"]:
        expected = _record(record["kind"], record["payload"], catalog["market"],
                           catalog["as_of_date"], catalog["allow_synthetic"])
        if record != expected or record["evidence_id"] in ids:
            raise ValueError("证据内容、时间、可用性或 ID 被改写")
        ids.add(record["evidence_id"])


def validate_links(annotation, catalog, symbol, allowed_kinds=None):
    if not isinstance(annotation, dict) or annotation.get("status") not in {"supported", "insufficient"}:
        raise ValueError("结论缺少证据状态")
    if not isinstance(annotation.get("invalidation_condition"), str) or not annotation["invalidation_condition"].strip():
        raise ValueError("结论缺少失效条件")
    by_id = {record["evidence_id"]: record for record in catalog["records"]}
    groups = []
    for name in ("supporting_ids", "contradicting_ids"):
        refs = annotation.get(name)
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs) or len(set(refs)) != len(refs):
            raise ValueError("证据引用必须是无重复 ID 数组")
        for ref in refs:
            record = by_id.get(ref)
            if (record is None or not record["eligible"] or record["symbol"] != symbol or
                    (allowed_kinds is not None and record["kind"] not in allowed_kinds)):
                raise ValueError("引用不属于本轮股票的可用证据")
        groups.append(set(refs))
    if groups[0] & groups[1]:
        raise ValueError("同一结论不能把同一证据同时列作支持和反证")
    if annotation["status"] == "supported" and not groups[0]:
        raise ValueError("有据结论必须至少引用一条支持证据")
    if annotation["status"] == "insufficient" and groups[0]:
        raise ValueError("证据不足状态不能同时声称存在支持依据")


def validate_claims(claims, catalog, symbols, allowed_kinds=None):
    validate_catalog(catalog)
    if not isinstance(claims, list) or {row.get("symbol") for row in claims if isinstance(row, dict)} != set(symbols):
        raise ValueError("专家证据结论必须覆盖本轮股票池")
    for row in claims:
        if not isinstance(row, dict) or not isinstance(row.get("claim"), str) or not row["claim"].strip():
            raise ValueError("专家结论缺少正文")
        validate_links(row.get("evidence"), catalog, row["symbol"], allowed_kinds)


def validate_result_evidence(result, catalog, symbols, *, risk=False):
    validate_catalog(catalog)
    for collection in (("assessments",) if risk else ("decisions", "predictions")):
        rows = result.get(collection)
        if not isinstance(rows, list) or {row.get("symbol") for row in rows if isinstance(row, dict)} != set(symbols):
            raise ValueError("证据结论未覆盖本轮股票池")
        for row in rows:
            validate_links(row.get("evidence"), catalog, row["symbol"])
            if collection == "decisions" and row.get("action") != "HOLD" and row["evidence"]["status"] != "supported":
                raise ValueError("证据不足的计划不得产生买卖")
            if (risk and row.get("action") != "HOLD" and row.get("verdict") != "REJECT" and
                    row["evidence"]["status"] != "supported"):
                raise ValueError("证据不足的风控不得批准买卖")


def validate_evidence_audit(result):
    audit = result.get("evidence_audit")
    if not audit:
        raise ValueError("缺少宿主证据审计")
    catalog = audit["catalog"]
    symbols = {row["symbol"] for row in result["decisions"]}
    validate_result_evidence(result, catalog, symbols)
    validate_result_evidence({"assessments": result["risk_review"]["assessments"]}, catalog, symbols, risk=True)
    for role in ("technical-analyst", "fundamental-analyst"):
        kinds = {"market_snapshot"} if role == "technical-analyst" else {"fundamentals", "news"}
        if result.get("investment_policy"):
            kinds |= {"price_history"} if role == "technical-analyst" else {"company_document", "market_snapshot"}
        validate_claims(audit["analyst_claims"][role], catalog, symbols, kinds)
    if result.get("investment_policy"):
        from agent.investment import validate_cases
        validate_cases(result, symbols, catalog=catalog, as_of=catalog["as_of_date"])
