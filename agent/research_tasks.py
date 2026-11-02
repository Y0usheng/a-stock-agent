"""Individual research questions, source candidates and dated progress. No model calls."""
from copy import deepcopy
from contextlib import closing
from datetime import datetime, timezone
import json
import re
import sqlite3

from evals.experiments import canonical_hash
from servers.data_quality import event_time

PROTOCOL = 'question-tasks-v1'
TOPICS = ('客户集中', '客户资产', '零售客户', '存款结构', '存款成本', '负债成本', '净息差',
          '不良贷款', '拨备覆盖', '资本充足', '信用成本', '信用风险', '贷款行业', '房地产',
          '产品结构', '直销', '经销', '销量', '价格', '合同负债', '渠道库存', '受限资金',
          '产量', '单位成本', '矿种', '项目回报', '资本承诺', '债务到期', '现金流',
          '净利润', '净资产', '营业收入', '资本开支', '分红', '负债', '成本', '竞争')
FINAL = {'resolved', 'migrated', 'quarantined'}


def questions(gap):
    parts = [p.strip() for p in re.split(r'[；;。\n]+', gap['what_to_fetch']) if len(p.strip()) >= 5]
    return [{**gap, 'what_to_fetch': p} for p in (parts or [gap['what_to_fetch']])]


def question_id(symbol, gap):
    normalized = re.sub(r'[\s，,。；;、：:（）()]+', '', gap['what_to_fetch'])
    return 'gap_' + canonical_hash([PROTOCOL, symbol, gap['category'], normalized])[:24]


def query_terms(task):
    text = task.get('what_to_fetch', '')
    phrases = []
    for part in re.split(r'[、，,；;。\s]+', text):
        part = re.sub(r'^(?:请|获取|取得|核对|核验|补取|补齐|获取并核对)+', '', part)
        part = re.sub(r'(?:的完整披露|原文片段|资料|披露|情况|数据)$', '', part)
        if 3 <= len(part) <= 20:
            phrases.append(part)
    return tuple(dict.fromkeys(phrases + [word for word in TOPICS if word in text]))[:20]


def task_materials(rows, limit=3):
    candidates = sorted((r for r in rows if r.get('task_ids')), key=lambda r: str(r.get('published_at', '')), reverse=True)
    candidates.sort(key=lambda r: (r.get('task_rank', 999), -r.get('task_match_score', 0)))
    selected, covered = [], set()
    for row in candidates:
        if set(row['task_ids']) - covered:
            selected.append(row)
            covered.update(row['task_ids'])
        if len(selected) == limit:
            break
    return selected


class TaskTracker:
    def _put_task(self, conn, task, day, *, stamp=None):
        task = deepcopy(task)
        old = conn.execute('SELECT payload,opened_at FROM tasks WHERE id=?', (task['task_id'],)).fetchone()
        task['opened_at'] = old['opened_at'] if old else task.get('opened_at', day)
        task['updated_at'] = day
        encoded = json.dumps(task, ensure_ascii=False)
        if old and json.loads(old['payload']) == task:
            return
        conn.execute('INSERT INTO tasks VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at,status=excluded.status,payload=excluded.payload',
                     (task['task_id'], task['symbol'], task['category'], task['opened_at'], day, task['status'], encoded))
        self._task_event(conn, task, stamp)

    def _task_event(self, conn, task, stamp=None):
        stamp = stamp or datetime.now(timezone.utc).isoformat()
        payload = {**deepcopy(task), 'recorded_at': stamp}
        conn.execute('INSERT OR IGNORE INTO task_events VALUES(?,?,?,?,?)',
                     (canonical_hash(payload), task['task_id'], task['symbol'], stamp, json.dumps(payload, ensure_ascii=False)))

    def _migrate_tasks(self, conn):
        legacy = conn.execute("SELECT * FROM tasks WHERE id LIKE 'gap_%' AND status!='migrated'").fetchall()
        for old in legacy:
            payload = json.loads(old['payload'])
            if payload.get('task_protocol') == PROTOCOL:
                continue
            case_row = next((r for r in conn.execute('SELECT payload,log_hash FROM cases WHERE symbol=? ORDER BY day DESC,rowid DESC', (old['symbol'],))
                             if canonical_hash(json.loads(r['payload'])) == payload.get('source_case_hash')), None)
            gaps = json.loads(case_row['payload'])['core_gaps'] if case_row else [payload]
            for gap in gaps:
                if gap['category'] != old['category']:
                    continue
                for question in questions(gap):
                    task = {**payload, **question, 'task_id': question_id(old['symbol'], question),
                            'task_protocol': PROTOCOL, 'opened_at': old['opened_at'],
                            'status': 'open' if old['status']=='resolved' else old['status'], 'legacy_status': old['status'],
                            'source_log_hash': case_row['log_hash'] if case_row else None,
                            'evidence': [], 'attempt_count': 0, 'migration_from': old['id']}
                    if not conn.execute('SELECT 1 FROM tasks WHERE id=?', (task['task_id'],)).fetchone():
                        self._put_task(conn, task, old['updated_at'])
            conn.execute("UPDATE tasks SET status='migrated' WHERE id=?", (old['id'],))
        for old in conn.execute("SELECT * FROM tasks WHERE id LIKE 'task_%'").fetchall():
            payload = json.loads(old['payload'])
            identifier = question_id(old['symbol'], payload)
            if not conn.execute('SELECT 1 FROM tasks WHERE id=?', (identifier,)).fetchone():
                task = {**payload, 'task_id': identifier, 'task_protocol': PROTOCOL,
                        'origin': 'baseline_source_question', 'opened_at': old['opened_at'],
                        'status': 'open', 'evidence': [], 'attempt_count': 0}
                self._put_task(conn, task, old['updated_at'])

    def migrate_tasks(self):
        with self.db() as conn:
            self._migrate_tasks(conn)

    def _save_tasks(self, conn, log, case):
        self._migrate_tasks(conn)
        day = log['as_of_date']
        old = [json.loads(r['payload']) for r in conn.execute("SELECT payload FROM tasks WHERE symbol=? AND id LIKE 'gap_%' AND status!='migrated'", (case['symbol'],))]
        active = [t for t in old if t['status'] not in FINAL]
        for gap in case['core_gaps']:
            for question in questions(gap):
                identifier = question_id(case['symbol'], question)
                existing = next((t for t in old if t['task_id'] == identifier), {})
                same_category = [t for t in active if t['category'] == question['category']]
                due = min([case['next_review_date']] + [t['review_by'] for t in same_category])
                task = {**existing, **question, 'task_id': identifier, 'task_protocol': PROTOCOL,
                        'symbol': case['symbol'], 'review_by': due, 'source_case_hash': canonical_hash(case),
                        'source_log_hash': canonical_hash(log), 'evidence': existing.get('evidence', []),
                        'attempt_count': existing.get('attempt_count', 0),
                        'status': existing.get('status', 'open') if existing.get('status') not in FINAL else 'open'}
                if day > due:
                    task['status'] = 'overdue'
                self._put_task(conn, task, day)
        if case['research_status'] == 'ready':
            refs = set(case['evidence'].get('supporting_ids', []))
            materials = {r['payload'].get('material_id') for r in log['evidence_audit']['catalog']['records']
                         if r['evidence_id'] in refs and r['kind'] == 'company_document' and r['eligible']}
            blocked = self.quality.blocked('material')
            for task in active:
                if any(e['material_id'] in materials - blocked for e in task.get('evidence', [])):
                    task.update(status='resolved', closed_by_log_hash=canonical_hash(log), resolution='manager_ready_with_cited_task_source')
                    self._put_task(conn, task, day)

    def record_capture(self, tasks, materials, *, cutoff, had_error=False):
        stamp = datetime.now(timezone.utc).isoformat()
        day = event_time(stamp, 'cn').date().isoformat()
        blocked = self.quality.blocked('material')
        with self.db() as conn:
            for original in tasks:
                if original.get('task_protocol') != PROTOCOL or original['status'] in FINAL:
                    continue
                row = conn.execute('SELECT payload FROM tasks WHERE id=?', (original['task_id'],)).fetchone()
                if not row:
                    continue
                task = json.loads(row['payload'])
                refs = []
                for material in materials:
                    if task['task_id'] not in material.get('task_ids', []) or material['symbol'] != task['symbol']:
                        continue
                    stored = conn.execute('SELECT payload FROM materials WHERE id=?', (material['material_id'],)).fetchone()
                    if not stored or json.loads(stored['payload']) != material or material['material_id'] in blocked:
                        continue
                    if event_time(material['published_at'], 'cn') > event_time(cutoff, 'cn') or not material.get('source_spans'):
                        continue
                    if event_time(material['first_seen_at'], 'cn') > event_time(stamp, 'cn') or material['expires_at'] < day:
                        continue
                    refs.append({k: deepcopy(material[k]) for k in ('material_id', 'url', 'locator', 'published_at', 'first_seen_at', 'expires_at', 'source_spans')})
                task.update(evidence=refs, attempt_count=task.get('attempt_count', 0)+1, last_attempt_at=stamp,
                            capture_status='source_found' if refs else 'capture_error' if had_error else 'not_found')
                task['status'] = 'overdue' if day > task['review_by'] else task['capture_status']
                self._put_task(conn, task, day, stamp=stamp)

    def task_snapshot(self, cutoff):
        if not self.path.exists():
            return []
        with closing(sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro', uri=True)) as conn:
            if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_events'").fetchone():
                return []
            rows = conn.execute('SELECT payload FROM task_events ORDER BY recorded_at,rowid').fetchall()
        latest = {}
        for row in rows:
            task = json.loads(row[0])
            if event_time(task['recorded_at'], 'cn') <= event_time(cutoff, 'cn'):
                latest[task['task_id']] = task
        return self._visible_tasks(list(latest.values()), cutoff)

    def _visible_tasks(self, tasks, cutoff=None):
        bad_cases = self.quality.blocked('case', cutoff)
        bad_logs = self.quality.blocked(cutoff=cutoff)
        rows = []
        for task in tasks:
            if task['status'] == 'migrated':
                continue
            if task.get('source_case_hash') in bad_cases or task.get('source_log_hash') in bad_logs:
                task = {**task, 'status': 'quarantined'}
            elif task.get('closed_by_log_hash') in bad_logs:
                task = {**task, 'status': 'open', 'resolution': 'invalidated_by_input_fault'}
            rows.append(task)
        specific = {(t['symbol'], t['category']) for t in rows if t.get('source_case_hash') and t['status'] != 'quarantined'}
        rows = [t for t in rows if t.get('origin') != 'baseline_source_question' or (t['symbol'], t['category']) not in specific]
        granular = {t['symbol'] for t in rows if t.get('task_protocol') == PROTOCOL and t['status'] != 'quarantined'}
        return sorted((t for t in rows if t.get('task_protocol') == PROTOCOL or t['symbol'] not in granular),
                      key=lambda t: (t['symbol'], t.get('review_by', ''), t['task_id']))
