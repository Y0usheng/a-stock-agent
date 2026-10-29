# -*- coding: utf-8 -*-
"""可追溯历史新闻数据集。

实时新闻在抓取时追加到 SQLite；第三方历史导出可用 CLI 导入。查询必须给出
``as_of_date``，数据库层会同时限制窗口起点和终点，防止回测读取未来新闻。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import sqlite3
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import NEWS_DB
from servers.data_quality import event_time, event_available_date, market_close


SCHEMA_VERSION = "news-events-v1"
QUALITY_VALUES = {"live", "imported", "synthetic"}


def _connect(db_path: Path | str = NEWS_DB) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS news_events (
        event_id TEXT PRIMARY KEY,
        market TEXT NOT NULL CHECK(market IN ('cn','us')),
        symbol TEXT NOT NULL,
        published_at TEXT NOT NULL,
        title TEXT NOT NULL,
        summary TEXT NOT NULL DEFAULT '',
        source TEXT NOT NULL,
        url TEXT NOT NULL DEFAULT '',
        sentiment_score REAL NOT NULL DEFAULT 0,
        event_tags_json TEXT NOT NULL DEFAULT '[]',
        data_quality TEXT NOT NULL CHECK(data_quality IN ('live','imported','synthetic')),
        content_hash TEXT NOT NULL,
        ingested_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_news_market_symbol_time
        ON news_events(market, symbol, published_at);
    """)
    return conn


@contextmanager
def _database(db_path: Path | str = NEWS_DB):
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _published(value: object, market: str) -> str:
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        raise ValueError("新闻缺少 published_at")
    try:
        parsed = event_time(text, market)
    except ValueError as exc:
        raise ValueError(f"无法解析 published_at={text!r}") from exc
    return text if len(text) == 10 else parsed.isoformat(timespec="seconds")


def _canonical_event(event: dict, market: str, data_quality: str) -> dict:
    if market not in {"cn", "us"}:
        raise ValueError(f"不支持的市场:{market}")
    if data_quality not in QUALITY_VALUES:
        raise ValueError(f"不支持的数据质量:{data_quality}")
    symbol = str(event.get("symbol") or "").strip().upper()
    title = " ".join(str(event.get("title") or "").split())[:180]
    source = " ".join(str(event.get("source") or "").split())[:100]
    if not symbol or not title or not source:
        raise ValueError("新闻必须包含 symbol、title、source")
    published_at = _published(event.get("published_at"), market)
    summary = " ".join(str(event.get("summary") or "").split())[:500]
    url = str(event.get("url") or "")[:1000]
    tags = sorted({str(tag)[:60] for tag in event.get("event_tags", []) if str(tag)})
    try:
        score = max(-3.0, min(3.0, float(event.get("sentiment_score") or 0)))
    except (TypeError, ValueError) as exc:
        raise ValueError("sentiment_score 必须是数字") from exc
    content = {
        "market": market, "symbol": symbol, "published_at": published_at,
        "title": title, "summary": summary, "source": source, "url": url,
        "sentiment_score": round(score, 6), "event_tags": tags,
    }
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    content_hash = hashlib.sha256(encoded).hexdigest()
    event_id = str(event.get("event_id") or content_hash[:24])[:64]
    return {**content, "event_id": event_id, "content_hash": content_hash,
            "data_quality": data_quality}


def ingest_events(events: Iterable[dict], market: str, data_quality: str = "live",
                  db_path: Path | str = NEWS_DB) -> dict:
    """幂等追加事件；同一 event_id 内容变化时拒绝，避免历史被静默改写。"""
    rows = [_canonical_event(event, market, data_quality) for event in events]
    inserted = 0
    duplicates = 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _database(db_path) as conn:
        for item in rows:
            existing = conn.execute(
                "SELECT content_hash FROM news_events WHERE event_id=?", (item["event_id"],)
            ).fetchone()
            if existing:
                if existing["content_hash"] != item["content_hash"]:
                    raise ValueError(f"event_id={item['event_id']} 已存在但内容不同")
                duplicates += 1
                continue
            conn.execute("""
                INSERT INTO news_events(
                    event_id,market,symbol,published_at,title,summary,source,url,
                    sentiment_score,event_tags_json,data_quality,content_hash,ingested_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                item["event_id"], item["market"], item["symbol"], item["published_at"],
                item["title"], item["summary"], item["source"], item["url"],
                item["sentiment_score"], json.dumps(item["event_tags"], ensure_ascii=False),
                item["data_quality"], item["content_hash"], now,
            ))
            inserted += 1
    return {"received": len(rows), "inserted": inserted, "duplicates": duplicates,
            "schema_version": SCHEMA_VERSION}


def query_events(market: str, symbol: str, as_of_date: str, days: int = 3,
                 limit: int = 20, db_path: Path | str = NEWS_DB,
                 include_synthetic: bool = False, information_cutoff_at: str | None = None) -> list[dict]:
    """按事件发布时间查询过去窗口，SQL 条件保证 published_at 不超过截面日。"""
    cutoff = event_time(information_cutoff_at, market) if information_cutoff_at else market_close(str(as_of_date)[:10], market)
    end = cutoff.date()
    start = end - timedelta(days=max(1, int(days)))
    quality_clause = "" if include_synthetic else "AND data_quality <> 'synthetic'"
    observed_clause = "AND julianday(ingested_at) <= julianday(?)" if information_cutoff_at else ""
    with _database(db_path) as conn:
        rows = conn.execute(f"""
            SELECT * FROM news_events
            WHERE market=? AND symbol=?
              AND ((length(published_at)=10 AND published_at >= ? AND published_at < ?)
                   OR (length(published_at)>10 AND julianday(published_at) >= julianday(?)
                       AND julianday(published_at) <= julianday(?)))
              {quality_clause}
              {observed_clause}
            ORDER BY published_at DESC, event_id ASC LIMIT ?
        """, (market, symbol.upper(), start.isoformat(), end.isoformat(),
              market_close(start.isoformat(), market).replace(hour=0, minute=0).isoformat(),
              cutoff.isoformat(), *([cutoff.isoformat()] if information_cutoff_at else []), max(1, int(limit)))).fetchall()
    return [{
        "event_id": row["event_id"], "symbol": row["symbol"],
        "title": row["title"], "summary": row["summary"],
        "published_at": row["published_at"], "source": row["source"],
        "url": row["url"], "sentiment_score": row["sentiment_score"],
        "event_tags": json.loads(row["event_tags_json"]),
        "data_quality": row["data_quality"], "untrusted_text": True,
        "content_hash": row["content_hash"], "ingested_at": row["ingested_at"],
        "first_seen_at": row["ingested_at"], "collected_at": row["ingested_at"],
    } for row in rows]


def get_stored_event(event_id: str, market: str, as_of_date: str,
                     db_path: Path | str = NEWS_DB) -> dict | None:
    """按 ID 返回已落库且截止日可见的保存片段；不下载或声称保存了原文全文。"""
    with _database(db_path) as conn:
        row = conn.execute("SELECT symbol,published_at FROM news_events WHERE event_id=? AND market=?",
                           (event_id, market)).fetchone()
    if not row or event_time(row["published_at"], market) > market_close(as_of_date, market):
        return None
    day = event_time(row["published_at"], market).date()
    days = max(1, (date.fromisoformat(as_of_date) - day).days + 1)
    return next((item for item in query_events(market, row["symbol"], as_of_date, days,
                 100000, db_path, include_synthetic=True) if item["event_id"] == event_id), None)


def load_news_scores(market: str, start_date: str, end_date: str,
                     db_path: Path | str = NEWS_DB,
                     include_synthetic: bool = False,
                     window_days: int = 3) -> tuple[dict[tuple[str, str], float], dict]:
    """加载回测日/股票分数；每个截面只聚合此前窗口事件，不读取未来。"""
    start_day = date.fromisoformat(start_date[:10])
    end_day = date.fromisoformat(end_date[:10])
    window_days = max(1, min(int(window_days), 30))
    query_start = start_day - timedelta(days=window_days + 1)
    quality_clause = "" if include_synthetic else "AND data_quality <> 'synthetic'"
    with _database(db_path) as conn:
        rows = conn.execute(f"""
            SELECT event_id,symbol,published_at,sentiment_score,content_hash,data_quality
            FROM news_events WHERE market=?
              AND substr(published_at,1,10) >= ? AND substr(published_at,1,10) <= ?
              {quality_clause}
            ORDER BY published_at,event_id
        """, (market, query_start.isoformat(), (end_day + timedelta(days=1)).isoformat())).fetchall()
    scores: dict[tuple[str, str], float] = {}
    attention: dict[str, dict[str, float]] = {}
    used_rows = []
    for row in rows:
        published_day = event_time(row["published_at"], market).date()
        first_signal_day = max(start_day, event_available_date(row["published_at"], market))
        last_signal_day = min(end_day, published_day + timedelta(days=window_days))
        if first_signal_day > last_signal_day:
            continue
        used_rows.append(row)
        signal_day = first_signal_day
        while signal_day <= last_signal_day:
            key = (signal_day.isoformat(), row["symbol"])
            scores[key] = round(scores.get(key, 0.0) + float(row["sentiment_score"]), 6)
            daily_attention = attention.setdefault(signal_day.isoformat(), {})
            daily_attention[row["symbol"]] = round(daily_attention.get(row["symbol"], 0.0)
                                                  + abs(float(row["sentiment_score"])) + .25, 6)
            signal_day += timedelta(days=1)
    rows = used_rows
    fingerprint_input = "close-cutoff-v2|" + "|".join(f"{row['event_id']}:{row['content_hash']}" for row in rows)
    fingerprint = hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()
    qualities = sorted({row["data_quality"] for row in rows})
    manifest = {
        "schema_version": SCHEMA_VERSION, "market": market,
        "start_date": start_date[:10], "end_date": end_date[:10],
        "signal_window_days": window_days,
        "timing_policy": "market-close-cutoff-v2; date-only usable next day",
        "attention_by_day": attention,
        "event_count": len(rows), "qualities": qualities,
        "fingerprint": fingerprint,
        "data_quality": "historical_news" if rows else "no_historical_news",
    }
    return scores, manifest


def dataset_summary(db_path: Path | str = NEWS_DB) -> dict:
    if not Path(db_path).exists():
        return {"schema_version": SCHEMA_VERSION, "event_count": 0, "markets": {}}
    with _database(db_path) as conn:
        rows = conn.execute("""
            SELECT market,data_quality,COUNT(*) AS n,MIN(substr(published_at,1,10)) AS first_day,
                   MAX(substr(published_at,1,10)) AS last_day
            FROM news_events GROUP BY market,data_quality ORDER BY market,data_quality
        """).fetchall()
    markets: dict[str, dict] = {}
    for row in rows:
        markets.setdefault(row["market"], {})[row["data_quality"]] = {
            "count": row["n"], "first_day": row["first_day"], "last_day": row["last_day"],
        }
    return {"schema_version": SCHEMA_VERSION,
            "event_count": sum(row["n"] for row in rows), "markets": markets}


def _read_import(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, dict):
        payload = payload.get("events", [])
    if not isinstance(payload, list):
        raise ValueError("导入文件必须是 JSON 数组、{events:[...]} 或 JSONL")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="导入/检查历史新闻数据集")
    parser.add_argument("--import-file", type=Path)
    parser.add_argument("--market", choices=["cn", "us"])
    parser.add_argument("--db", type=Path, default=NEWS_DB)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.import_file:
        if not args.market:
            parser.error("--import-file 需要 --market")
        print(json.dumps(ingest_events(_read_import(args.import_file), args.market,
                                       "imported", args.db), ensure_ascii=False, indent=2))
    if args.summary or not args.import_file:
        print(json.dumps(dataset_summary(args.db), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
