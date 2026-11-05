"""Register a prospective CSI 300 total-return reference and append real valuations."""
from datetime import timedelta
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.trading_clock import calendar, now
from evals.portfolio_comparison import ComparisonBook, SPEC, normalize_index, _write
from evals.trading_view import PRICE_CACHE, analysis_logs, build_view, read_ledger
from servers.data_quality import market_close


def main():
    import requests
    stamp = now()
    book = ComparisonBook()
    try:
        sessions = calendar()['sessions']
        policy = book.register(sessions, stamp.isoformat())
        completed = max(d for d in sessions if market_close(d,'cn') <= stamp)
        start = min((stamp-timedelta(days=30)).date().isoformat(), policy['start_session'])
        response = requests.get('https://www.csindex.com.cn/csindex-home/perf/index-perf',
            params={'indexCode':SPEC['index_code'], 'startDate':start.replace('-',''), 'endDate':completed.replace('-','')},
            timeout=(5,12), allow_redirects=False)
        response.raise_for_status()
        if response.status_code != 200 or len(response.content) > 2_000_000:
            raise ValueError('官方指数响应无效')
        rows = normalize_index(response.json().get('data') or [], stamp.isoformat())
        if rows[-1]['date'] != completed:
            raise ValueError('官方基准尚未更新至最近已收盘交易日')
        _write(book.root/'index.json', {'index_code':SPEC['index_code'], 'source':'中证指数官方历史收盘',
                                      'fetched_at':stamp.isoformat(), 'rows':rows})
        if not PRICE_CACHE.exists():
            state = 'waiting_for_start' if completed < policy['start_session'] else 'waiting_for_portfolio_prices'
        else:
            ledger = read_ledger()
            view = build_view(json.loads(PRICE_CACHE.read_text(encoding='utf-8')), ledger, analysis_logs())
            if not view['portfolio']['journal_reconciled']:
                raise ValueError('模拟账本未对账')
            state = book.observe(policy, rows, view['portfolio_curve'], ledger, sessions, stamp.isoformat())
            if completed >= policy['start_session'] and view['as_of_date'] != completed:
                state = 'waiting_for_portfolio_prices'
        book.status(stamp.isoformat(), state, rows[-1]['date'])
        print(json.dumps({'status':state, 'registered_at':policy['registered_at'], 'start_session':policy['start_session'],
                          'source_date':rows[-1]['date'], 'model_requests':0, 'orders_submitted':0},ensure_ascii=False))
        return 2 if state == 'waiting_for_portfolio_prices' else 0
    except Exception as exc:
        # Never print raw network responses, local paths or authentication output.
        try:
            book.status(stamp.isoformat(), 'refresh_failed')
        except (OSError, ValueError):
            pass  # Corrupt/unwritable records remain frozen; never print their contents.
        print(json.dumps({'status':'failed','error_type':type(exc).__name__,
                          'message':'组合基准刷新或估值核对未完成；保留原登记和已有估值，未重跑投资研究。'},ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
