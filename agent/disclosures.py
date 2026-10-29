"""Bounded controlled-source PDF acquisition; captured text is not semantic verification."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from time import monotonic
from urllib.parse import urljoin, urlsplit

from agent.investment import ResearchBook
from config import DATA_DIR, MARKETS, ROOT
from evals.experiments import canonical_hash
from servers.data_quality import event_time

VERSION = "controlled-pdf-capture-v4"
HOME = DATA_DIR / "disclosures"
MAX_PDF = 40_000_000
UNITS = re.compile(r"单位\s*[：:][^\n]{1,45}|营业收入\s*[（(](?:元|千元|万元|百万元)[）)]")
TERMS = {
    "business": ("主要业务", "主营业务", "经营模式", "核心竞争力", "主要风险", "Business overview"),
    "financials": ("主要会计数据", "营业收入", "经营活动产生的现金流量净额", "资产总计", "负债合计", "合并现金流量表", "Revenue", "Cash flow"),
}


def financial_unit_visible(text):
    # A unit from a later, unrelated table cannot label the first financial table.
    inline = re.search(r"营业收入\s*[（(](?:元|千元|万元|百万元)[）)]", text)
    first_value = re.search(r"营业收入|利润总额|净利润|资产总计|总资产|负债合计|经营活动产生", text)
    units = UNITS.search(text)
    return bool(first_value and ((inline and inline.start() == first_value.start()) or (units and units.start() < first_value.start())))


def controlled_pdf(url):
    parsed = urlsplit(url)
    return (parsed.scheme == "https" and not parsed.username and not parsed.password and parsed.port in (None, 443)
            and parsed.hostname in {"static.cninfo.com.cn", "www.cninfo.com.cn", "www.sse.com.cn", "static.sse.com.cn", "disc.static.szse.cn"})


def query_reports(symbol, cutoff, session):
    response = session.post("https://www.cninfo.com.cn/new/information/topSearch/query", data={"keyWord": symbol, "maxNum": 10}, timeout=(5, 12))
    response.raise_for_status()
    company = next(row for row in response.json() if row.get("code") == symbol)
    year = event_time(cutoff, "cn").year
    params = {"pageNum": "1", "pageSize": "30", "column": "szse", "tabName": "fulltext", "stock": symbol + "," + company["orgId"],
        "searchkey": "", "category": "category_ndbg_szsh;category_sjdbg_szsh;category_bndbg_szsh;category_yjdbg_szsh",
        "seDate": f"{year-3}-01-01~{str(cutoff)[:10]}", "sortName": "time", "sortType": "desc", "isHLtitle": "true"}
    response = session.post("https://www.cninfo.com.cn/new/hisAnnouncement/query", data=params, timeout=(5, 12))
    response.raise_for_status()
    return select_reports(response.json().get("announcements") or [], symbol, cutoff)


def select_reports(rows, symbol, cutoff):
    clean = []
    for row in rows:
        title = re.sub(r"<[^>]*>", "", row.get("announcementTitle", ""))
        path = str(row.get("adjunctUrl", ""))
        if row.get("secCode") != symbol or any(word in title for word in ("摘要", "英文", "取消", "更正", "修订说明")):
            continue
        if not re.fullmatch(r"finalpage/\d{4}-\d{2}-\d{2}/\d+\.[Pp][Dd][Ff]", path):
            continue
        published = datetime.fromtimestamp(float(row["announcementTime"]) / 1000, timezone.utc).astimezone(event_time(cutoff, "cn").tzinfo)
        # The index generally provides a disclosure date, not a verified release time.
        stamp = published.date().isoformat() if published.hour == 0 and published.minute == 0 else published.isoformat()
        if event_time(stamp, "cn") > event_time(cutoff, "cn"):
            continue
        kind = "annual" if "年度报告" in title and "半年度" not in title else "interim"
        clean.append({"symbol": symbol, "title": title, "published_at": stamp, "url": "https://static.cninfo.com.cn/" + path,
                      "announcement_id": Path(path).stem, "document_type": kind})
    clean.sort(key=lambda row: row["published_at"], reverse=True)
    annual = next((row for row in clean if row["document_type"] == "annual"), None)
    interim = next((row for row in clean if row["document_type"] == "interim" and (not annual or row["published_at"] > annual["published_at"])), None)
    return [row for row in (annual, interim) if row]


def download_pdf(url, session, deadline):
    for _ in range(3):
        if not controlled_pdf(url) or monotonic() >= deadline:
            raise ValueError("PDF链接或采集预算无效")
        response = session.get(url, timeout=(5, max(1, min(15, deadline-monotonic()))), stream=True, allow_redirects=False)
        if response.status_code in {301, 302, 303, 307, 308}:
            url = urljoin(url, response.headers.get("Location", ""))
            response.close()
            continue
        response.raise_for_status()
        parts, size = [], 0
        try:
            for part in response.iter_content(65536):
                size += len(part)
                if size > MAX_PDF or monotonic() >= deadline:
                    raise ValueError("PDF超出大小或总时限")
                parts.append(part)
        finally:
            response.close()
        data = b"".join(parts)
        if not data.startswith(b"%PDF-"):
            raise ValueError("来源没有返回PDF")
        return data
    raise ValueError("PDF重定向过多")


def extract_pages(path):
    from pypdf import PdfReader
    reader = PdfReader(path, strict=False)
    if reader.is_encrypted or not 1 <= len(reader.pages) <= 700:
        raise ValueError("加密或页数超限，需人工处理")
    pages = []
    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = "\n".join(line.strip() for line in text.splitlines())[:50000]
        pages.append({"pdf_page": index + 1, "text": text, "text_hash": canonical_hash(text)})
    if sum(len(page["text"].strip()) for page in pages) < 500:
        raise ValueError("扫描件或文本不足，需OCR核验")
    return {"pages": pages, "page_count": len(pages), "extractor": "pypdf", "semantic_verified": False}


def select_passages(document, extracted):
    identity_text = "".join(page["text"] for page in extracted["pages"][:3])
    if document["symbol"] not in re.sub(r"\s+", "", identity_text):
        raise ValueError("PDF前页未找到对应股票代码，不能自动绑定公司")
    selected = []
    for category, terms in TERMS.items():
        if category == "business" and document["document_type"] != "annual":
            continue
        scored = []
        for page in extracted["pages"]:
            text = page["text"]
            if "目录" in text[:180] or text.count("\ufffd") > max(2, len(text) * .01):
                continue
            hits = [text.find(term) for term in terms if term in text]
            if not hits:
                continue
            # Prefer a compact page with several relevant labels, not isolated keyword density.
            score = len(hits) + (10 if category == "financials" and "主要会计数据" in text else 0)
            start = max(0, min(hits) - 220)
            business_heading = re.search(r"(?:^|\n)[一二三四五六七八九十]+、\s*报告期内公司[^\n]{0,12}业务", text)
            if category == "business" and business_heading:
                score += 100
                start = business_heading.start()
            elif category == "financials" and "主要会计数据" in text:
                start = text.rfind("\n", 0, text.find("主要会计数据")) + 1
            start = text.rfind("\n", 0, start) + 1
            end = text.rfind("\n", start, min(len(text), start + 1100))
            end = end if end > start + 100 else min(len(text), start + 1100)
            excerpt = text[start:end].strip()
            if len(excerpt) < 100:
                continue
            scored.append((score, -page["pdf_page"], page, start, end, excerpt))
        for _, _, page, start, end, excerpt in sorted(scored, key=lambda row: (row[0], row[1]), reverse=True)[:1]:
            spans = [{"pdf_page": page["pdf_page"], "start": start, "end": end, "page_text_hash": page["text_hash"]}]
            anchor = ""
            if category == "financials" and not financial_unit_visible(excerpt) and page["pdf_page"] > 1:
                previous = extracted["pages"][page["pdf_page"] - 2]
                tail = previous["text"][-240:]
                # Narrow continuation rule: the prior page ends with this table's heading + unit,
                # and the current page begins with the same table heading. Never borrow arbitrary units.
                unit = UNITS.search(tail)
                if unit and "主要会计数据" in tail[:unit.start()] and not tail[unit.end():].strip() and "主要会计数据" in page["text"][:180]:
                    anchor_start = len(previous["text"]) - len(tail) + tail.rfind("主要会计数据", 0, unit.start())
                    anchor = previous["text"][anchor_start:].strip()
                    spans.insert(0, {"pdf_page": previous["pdf_page"], "start": anchor_start, "end": len(previous["text"]), "page_text_hash": previous["text_hash"]})
                    excerpt = f"[PDF第{previous['pdf_page']}页表头与单位]\n{anchor}\n[PDF第{page['pdf_page']}页续表]\n{excerpt}"
            line_start = page["text"].count("\n", 0, start) + 1
            line_end = page["text"].count("\n", 0, end) + 1
            selected.append({**document, "categories": [category], "excerpt": excerpt,
                "locator": (f"PDF第{page['pdf_page']-1}页末表头单位及" if anchor else "") + f"PDF第{page['pdf_page']}页，文本行{line_start}-{line_end}（PDF页序，非印刷页码）",
                "pdf_page": page["pdf_page"], "text_start": start, "text_end": end, "page_text_hash": page["text_hash"],
                "data_source": "巨潮官方PDF自动采集", "data_quality": "imported",
                "source_truth_verified": False, "capture_protocol": VERSION, "source_spans": spans,
                "unit_context_present": financial_unit_visible(excerpt) if category == "financials" else None,
                "extraction_status": "source_spans_units_checked_semantics_unverified",
                "page_excerpt_complete": start == 0 and end == len(page["text"]), "report_period": None})
    return selected


def collect(symbols, *, book=None, root=HOME, budget_seconds=90, cutoff=None):
    import requests
    from agent.trading_clock import now
    symbols = list(dict.fromkeys(symbols))
    if not symbols or len(symbols) > 3 or not set(symbols) <= {row["symbol"] for row in MARKETS["cn"]["candidate_universe"]}:
        raise ValueError("每次最多采集3家受控公司")
    book, root = book or ResearchBook(), Path(root)
    root.mkdir(parents=True, exist_ok=True)
    deadline = monotonic() + budget_seconds
    cutoff = cutoff or now().isoformat()
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"})
    receipt = {"protocol": VERSION, "requested_symbols": symbols, "documents": [], "errors": [], "model_requests": 0}
    for symbol in symbols:
        if monotonic() >= deadline:
            receipt["errors"].append({"symbol": symbol, "reason": "total_budget_exhausted"})
            continue
        try:
            metadata = root / (symbol + ".index.json")
            cached = json.loads(metadata.read_text(encoding="utf-8")) if metadata.exists() else None
            if cached and (now() - event_time(cached["fetched_at"], "cn")).total_seconds() < 86400:
                documents = cached["documents"]
            else:
                documents = query_reports(symbol, cutoff, session)
                from agent.analysis import write_log
                write_log(metadata, {"fetched_at": now().isoformat(), "documents": documents})
            if not documents:
                raise ValueError("没有符合时点的年报或季报")
            for document in documents:
                if deadline - monotonic() < 3:
                    receipt["errors"].append({"symbol": symbol, "reason": "total_budget_exhausted"})
                    break
                reference = root / (document["announcement_id"] + ".capture.json")
                if reference.exists():
                    capture = json.loads(reference.read_text(encoding="utf-8"))
                    digest = capture["pdf_sha256"]
                    pdf_path = root / (digest + ".pdf")
                    if hashlib.sha256(pdf_path.read_bytes()).hexdigest() != digest:
                        raise ValueError("PDF原件已被改写")
                else:
                    data = download_pdf(document["url"], session, deadline)
                    digest = hashlib.sha256(data).hexdigest()
                    pdf_path = root / (digest + ".pdf")
                    if not pdf_path.exists():
                        pdf_path.write_bytes(data)
                    capture = {**document, "pdf_sha256": digest, "first_downloaded_at": now().isoformat(), "byte_count": len(data)}
                    from agent.analysis import write_log
                    write_log(reference, capture)
                text_path = root / (digest + ".pages.json")
                if not text_path.exists() or not capture.get("pages_sha256"):
                    subprocess.run([sys.executable, str(ROOT / "scripts" / "collect_disclosures.py"), "--extract-worker", str(pdf_path), str(text_path)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=max(1, min(35, deadline-monotonic())))
                    capture["pages_sha256"] = hashlib.sha256(text_path.read_bytes()).hexdigest()
                    from agent.analysis import write_log
                    write_log(reference, capture)
                if hashlib.sha256(text_path.read_bytes()).hexdigest() != capture["pages_sha256"]:
                    raise ValueError("PDF提取文本缓存已被改写")
                extracted = json.loads(text_path.read_text(encoding="utf-8"))
                passages = select_passages(document, extracted)
                identifiers = [book.add_material({**passage, "pdf_sha256": digest, "first_downloaded_at": capture["first_downloaded_at"]})["material_id"] for passage in passages]
                receipt["documents"].append({**document, "pdf_sha256": digest, "page_count": extracted["page_count"], "material_ids": identifiers,
                                             "captured_categories": sorted({category for p in passages for category in p["categories"]})})
        except Exception as exc:
            receipt["errors"].append({"symbol": symbol, "reason": type(exc).__name__ + ": controlled_source_capture_failed"})
    session.close()
    receipt["elapsed_seconds"] = round(budget_seconds - max(0, deadline-monotonic()), 3)
    from agent.analysis import write_log
    write_log(root / "latest-receipt.json", receipt)
    return receipt
