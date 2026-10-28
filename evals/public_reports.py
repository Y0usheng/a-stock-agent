"""Public, immutable daily reports. Only an explicit projection leaves the local workspace."""
from copy import deepcopy
from datetime import date, datetime
from html import escape
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

from agent.opportunities import OpportunityBook
from config import MARKETS, LOG_DIR
from evals.experiments import canonical_hash
from evals.trading_view import PRICE_CACHE, _journal, analysis_logs, build_view, read_ledger, render_html
from evals.metrics import uses_synthetic_data
from scripts.check_repository_secrets import scan_content
from servers.data_quality import market_zone

VERSION = "public-paper-report-v1"
REPORT_FILE = re.compile(r"\d{4}-\d{2}-\d{2}\.[0-9a-f]{16}\.json")
MAX_BYTES = 5_000_000


def saved_logs():
    rows = analysis_logs()
    for path in (Path(LOG_DIR) / "pending").glob("*.cn.*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("market") == "cn" and not uses_synthetic_data(row):
                rows.append(row)
        except (ValueError, OSError):
            continue
    return rows


def pick(row, keys):
    return {key: deepcopy(row[key]) for key in keys.split() if key in row}


def public_url(value):
    parsed = urlsplit(value or "")
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443):
        return ""
    hosts = ("gov.cn", "eastmoney.com", "cls.cn", "cninfo.com.cn", "sse.com.cn", "szse.cn")
    if not any(parsed.hostname == host or (parsed.hostname or "").endswith("." + host) for host in hosts):
        return ""
    if re.search(r"(?i)(token|key|secret|password|auth)=", parsed.query):
        return ""
    return value


def public_case(row):
    if not row:
        return None
    result = pick(row, "symbol research_status position_intent thesis thesis_horizon_years thesis_change target_weight entry_condition exit_condition next_review_date")
    result["valuation"] = pick(row.get("valuation", {}), "method bear_price base_price bull_price assumptions")
    return result


def sanitize_view(view):
    result = pick(view, "status market as_of_date fetched_at source price_basis fingerprint tracking_start analysis_date analysis_policy message warnings limitations")
    result["portfolio"] = pick(view.get("portfolio", {}), "initial_cash cash total_assets net_pnl net_return_pct realized_pnl unrealized_pnl total_fees total_slippage trade_count position_count journal_reconciled")
    result["portfolio_curve"] = [pick(row, "date total cumulative_return_pct") for row in view.get("portfolio_curve", [])]
    result["positions"] = [pick(row, "symbol name shares avg_cost last_buy_date last_price market_value unrealized_pnl") for row in view.get("positions", [])]
    def trade(row):
        return pick(row, "symbol name trade_date action shares price fee slippage_cost")
    result["trades"] = [trade(row) for row in view.get("trades", [])]
    result["stocks"] = []
    for row in view.get("stocks", []):
        item = pick(row, "symbol name last_close daily_change_pct")
        item["series"] = [pick(point, "date open high low close volume ma20 ma60") for point in row.get("series", [])]
        item["tracking_benchmark"] = [pick(point, "date return_pct") for point in row.get("tracking_benchmark", [])]
        item["trades"] = [trade(t) for t in row.get("trades", [])]
        item["decision"] = pick(row["decision"], "symbol name action shares price market_price reason") if row.get("decision") else None
        item["prediction"] = pick(row["prediction"], "symbol direction horizon_days confidence rationale") if row.get("prediction") else None
        item["investment_case"] = public_case(row.get("investment_case"))
        result["stocks"].append(item)
    result["research_tasks"] = [pick(row, "symbol status what_to_fetch review_by") for row in view.get("research_tasks", [])]
    opportunity = view.get("opportunities", {})
    public = pick(opportunity, "status as_of_date message selected_symbols new_candidate_limit selection_mode limitations")
    public["candidates"] = [pick(row, "symbol name status reason theme_ids") for row in opportunity.get("candidates", [])]
    public["excluded_events"] = [pick(row, "event_id reason") for row in opportunity.get("excluded_events", [])]
    public["omitted_theme_ids"] = deepcopy(opportunity.get("omitted_theme_ids", []))
    public["forward_observations"] = [pick(row, "symbol selected_on observation_start price_as_of observed_return_pct status basis") for row in opportunity.get("forward_observations", [])]
    public["themes"] = []
    for row in opportunity.get("themes", []):
        theme = pick(row, "theme_id name possible_symbols status mapping_status")
        theme["judgment"] = pick(row["judgment"], "assessment durability horizon_months priority mechanism counterargument validation_needed invalidation_condition beneficiary_symbols") if row.get("judgment") else None
        theme["events"] = [{**pick(event, "event_id title source_kind published_at"), "url": public_url(event.get("url"))} for event in row.get("events", [])]
        public["themes"].append(theme)
    result["opportunities"] = public
    return result


def ledger_as_of(ledger, day):
    initial = MARKETS["cn"]["initial_cash"]
    current_cash, quantities, _ = _journal(ledger, initial)
    if abs(current_cash - ledger["cash"]) > .02 or quantities != {row["symbol"]: row["shares"] for row in ledger["positions"]}:
        raise ValueError("账本未对账，停止公开收益")
    trades = [row for row in ledger["trades"] if row["trade_date"] <= day]
    holdings = {}
    for row in trades:
        symbol, quantity = row["symbol"], row["shares"]
        held = holdings.setdefault(symbol, {"symbol": symbol, "name": row.get("name", symbol), "shares": 0, "basis": 0., "last_buy_date": ""})
        if row["action"] == "BUY":
            held["basis"] += quantity * row["price"] + row["fee"]
            held["shares"] += quantity
            held["last_buy_date"] = row["trade_date"]
        else:
            held["basis"] -= held["basis"] / held["shares"] * quantity
            held["shares"] -= quantity
    cash, _, _ = _journal({"trades": trades}, initial)
    positions = [{**pick(row, "symbol name shares last_buy_date"), "avg_cost": row["basis"] / row["shares"]} for row in holdings.values() if row["shares"]]
    return {"cash": cash, "positions": positions, "trades": trades, "exists": ledger.get("exists", False)}


def report_days(logs=None, book=None):
    logs = saved_logs() if logs is None else logs
    book = book or OpportunityBook()
    snapshots = book._read("SELECT payload FROM snapshots ORDER BY rowid")
    return sorted({row.get("as_of_date") or row["date"] for row in logs} | {row["as_of_date"] for row in snapshots})


def make_report(day, *, cache=PRICE_CACHE, ledger=None, logs=None, book=None):
    if date.fromisoformat(day) > datetime.now(market_zone("cn")).date():
        raise ValueError("不能公开未来日期报告")
    payload = json.loads(Path(cache).read_text(encoding="utf-8"))
    if canonical_hash(payload["histories"]) != payload.get("fingerprint"):
        raise ValueError("行情缓存指纹不符")
    if payload.get("data_quality") != "live_prices":
        raise ValueError("公开报告只接受真实行情")
    payload["histories"] = {symbol: [row for row in rows if row["date"] <= day] for symbol, rows in payload["histories"].items()}
    payload["histories"] = {symbol: rows for symbol, rows in payload["histories"].items() if rows}
    payload["fingerprint"] = canonical_hash(payload["histories"])
    logs = saved_logs() if logs is None else logs
    logs = [row for row in logs if not uses_synthetic_data(row) and (row.get("as_of_date") or row.get("date", "")) <= day]
    logs.sort(key=lambda row: (row.get("as_of_date") or row.get("date", ""), {"paper": 2, "pending": 1}.get(row.get("execution_mode"), 0), row.get("generated_at", "")))
    latest = logs[-1] if logs else {}
    ledger = ledger_as_of(read_ledger() if ledger is None else ledger, day)
    view = build_view(payload, ledger, logs)
    book = book or OpportunityBook()
    snapshots = [row for row in book._read("SELECT payload FROM snapshots ORDER BY rowid") if row["as_of_date"] <= day]
    if snapshots:
        # Share the existing forward-observation implementation with an as-of-only source.
        class AsOfBook:
            def latest(self):
                return snapshots[-1]
            def _read(self, *args):
                return snapshots
        view["opportunities"] = OpportunityBook.view(AsOfBook(), payload["histories"])
    view["research_tasks"] = []  # Current task status is not a historical snapshot.
    view = sanitize_view(view)
    mode = latest.get("execution_mode", "no_analysis")
    base = {"schema": VERSION, "report_date": day, "analysis_mode": mode,
        "analysis_date": view.get("analysis_date"), "price_date": view.get("as_of_date"),
        "reconstruction": "saved_inputs_and_dated_simulated_trades_not_contemporaneous_publication",
        "source_dataset_hash": latest.get("dataset_fingerprint"), "view": view}
    result = {**base, "content_hash": canonical_hash(base),
              "published_at": datetime.now(market_zone("cn")).isoformat()}
    validate_report(result)
    return result


def validate_report(report):
    if set(report) != {"schema", "report_date", "analysis_mode", "analysis_date", "price_date", "reconstruction", "source_dataset_hash", "view", "content_hash", "published_at"}:
        raise ValueError("公开报告字段不符合白名单")
    if report["schema"] != VERSION or report["view"] != sanitize_view(report["view"]):
        raise ValueError("公开视图包含非公开字段")
    if report["analysis_mode"] not in {"shadow", "paper", "pending", "no_analysis"}:
        raise ValueError("未知分析模式")
    if report["source_dataset_hash"] is not None and (not isinstance(report["source_dataset_hash"], str) or not re.fullmatch(r"[0-9a-f]{64}", report["source_dataset_hash"])):
        raise ValueError("来源身份字段无效")
    if report["price_date"] != report["view"].get("as_of_date") or report["analysis_date"] != report["view"].get("analysis_date"):
        raise ValueError("报告与视图时点不一致")
    day = date.fromisoformat(report["report_date"])
    if day > datetime.now(market_zone("cn")).date():
        raise ValueError("报告日期在未来")
    for field in ("analysis_date", "price_date"):
        if report[field] and date.fromisoformat(report[field]) > day:
            raise ValueError("报告含未来分析或价格")
    datetime.fromisoformat(report["published_at"])
    numeric = set("initial_cash cash total_assets net_pnl net_return_pct realized_pnl unrealized_pnl total_fees total_slippage trade_count position_count shares avg_cost last_price market_value price fee slippage_cost open high low close volume ma20 ma60 total cumulative_return_pct return_pct daily_change_pct last_close market_price horizon_days confidence target_weight thesis_horizon_years bear_price base_price bull_price new_candidate_limit horizon_months priority observed_return_pct".split())
    containers = {"", "portfolio", "opportunities", "investment_case", "valuation", "decision", "prediction", "judgment"}
    string_arrays = {"warnings", "limitations", "selected_symbols", "possible_symbols", "theme_ids", "omitted_theme_ids", "beneficiary_symbols"}
    def check(value, key="", depth=0):
        if depth > 12:
            raise ValueError("公开报告嵌套超限")
        if key in numeric and value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError("公开数值字段无效")
        if isinstance(value, dict):
            if key not in containers:
                raise ValueError("公开标量字段不能包含任意对象")
            for name, child in value.items():
                check(child, name, depth + 1)
        elif isinstance(value, list):
            if key in string_arrays and not all(isinstance(child, str) for child in value):
                raise ValueError("公开文本列表无效")
            for child in value:
                check(child, "", depth + 1)
        elif isinstance(value, str) and len(value) > 3000:
            raise ValueError("公开文本超限")
        if key in {"date", "trade_date", "last_buy_date", "as_of_date", "selected_on", "observation_start", "price_as_of"} and value and date.fromisoformat(value) > day:
            raise ValueError("公开视图包含未来记录")
    check(report["view"])
    base = {key: value for key, value in report.items() if key not in {"content_hash", "published_at"}}
    if report["content_hash"] != canonical_hash(base):
        raise ValueError("公开报告内容指纹不符")
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_BYTES or scan_content("public-report.json", encoded) or re.search(rb"[A-Za-z]:\\|/(?:Users|home)/", encoded):
        raise ValueError("公开报告超限或包含疑似凭据/本机路径，停止发布")
    return encoded


def filename(report):
    return report["report_date"] + "." + report["content_hash"][:16] + ".json"


def write_report(report, root):
    data = validate_report(report)
    path = Path(root) / "reports" / filename(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        validate_report(previous)
        if previous["content_hash"] != report["content_hash"]:
            raise ValueError("不可覆盖既有报告")
        return path
    path.write_bytes(data)
    return path


def build_site(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("站点输出须与公开输入目录分开")
    if (source / "reports").is_symlink():
        raise ValueError("公开输入不能使用符号链接")
    if output.exists():
        for existing in output.rglob("*"):
            relative = existing.relative_to(output).as_posix()
            generated = relative in {"index.html", "manifest.json", ".nojekyll"} or re.fullmatch(r"reports/\d{4}-\d{2}-\d{2}\.html", relative)
            if existing.is_symlink() or (existing.is_file() and not generated):
                raise ValueError("输出目录包含非站点文件，拒绝混入部署包")
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for path in sorted((source / "reports").glob("*.json")):
        if path.is_symlink() or not REPORT_FILE.fullmatch(path.name) or path.stat().st_size > MAX_BYTES:
            raise ValueError("报告路径/大小无效")
        report = json.loads(path.read_text(encoding="utf-8"))
        validate_report(report)
        if path.name != filename(report):
            raise ValueError("文件名与内容身份不符")
        reports.append(report)
    latest = {}
    for report in sorted(reports, key=lambda row: (row["published_at"], row["content_hash"])):
        latest[report["report_date"]] = report
    entries = []
    for day, report in sorted(latest.items(), reverse=True):
        view = report["view"]
        html = render_html(view).replace('<a href="/">账户总览 ↗</a>', '<span class="muted">公开只读快照</span>')
        label = {"shadow": "只分析 · 未成交建议", "paper": "模拟账本结果", "pending": "待本机确认 · 尚未成交", "no_analysis": "暂无公司分析"}.get(report["analysis_mode"], "研究结果")
        banner = f'<div class="note">报告日期 {day} · 行情截至 {report["price_date"] or "未知"} · 最近分析 {report["analysis_date"] or "无"} · {label}<br>发布于 {escape(report["published_at"])}；历史报告由已保存输入重建，未补造新判断或成交。</div>'
        html = html.replace('<div id="error">', banner + '<div id="error">')
        target = output / "reports" / (day + ".html")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        entries.append({"date": day, "href": "reports/" + day + ".html", "price_date": report["price_date"], "analysis_date": report["analysis_date"], "mode": label})
    manifest = {"schema": VERSION, "reports": entries, "revision_count": len(reports)}
    encoded = json.dumps(manifest, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).resolve().parent.parent / "dashboard" / "templates" / "archive.html"
    (output / "index.html").write_text(template.read_text(encoding="utf-8").replace("__ARCHIVE_DATA__", encoded), encoding="utf-8")
    (output / "manifest.json").write_text(encoded, encoding="utf-8")
    (output / ".nojekyll").write_text("", encoding="utf-8")
    return manifest
