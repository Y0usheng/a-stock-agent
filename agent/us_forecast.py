"""Freeze a US forecast before the target session and settle it once prices arrive."""
from datetime import date, timedelta
import json

from agent.analysis import write_log
from config import DATA_DIR, LOG_DIR
from evals.calibration import apply_calibrator, calibration_metrics, fit_histogram_calibrator
from evals.experiments import canonical_hash
from servers.data_quality import event_time, market_zone
from servers.us_research_sources import PRICE_SOURCE

VERSION = 'us-next-session-forecast-v2'
OUTCOMES = DATA_DIR / 'us_forecasts' / 'outcomes'


def forecast_window(now):
    import pandas_market_calendars as calendars
    zone = market_zone('us')
    local = now.astimezone(zone)
    today = local.date()
    schedule = calendars.get_calendar('NASDAQ').schedule(start_date=today-timedelta(days=12),
                                                         end_date=today+timedelta(days=12))
    sessions = {day.date().isoformat(): {k: row[k].to_pydatetime().astimezone(zone)
                                        for k in ('market_open', 'market_close')}
                for day, row in schedule.iterrows()}
    current = sessions.get(today.isoformat())
    if not current:
        return None, 'us_exchange_holiday'
    if current['market_open'] <= local < current['market_close']:
        return None, 'us_regular_session_still_open'
    completed = {day: times['market_close'] for day, times in sessions.items() if times['market_close'] <= local}
    upcoming = [(day, times) for day, times in sessions.items() if times['market_open'] > local]
    if len(completed) < 2 or not upcoming:
        raise ValueError('美股交易日历没有覆盖参考日或目标日')
    target, times = upcoming[0]
    return {'completed': completed, 'target_session': target,
            'target_open_at': times['market_open'].isoformat(),
            'target_close_at': times['market_close'].isoformat()}, None


def reference_session(histories, completed):
    common = set.intersection(*({r['date'] for r in rows} for rows in histories.values()))
    eligible = sorted(completed)[-2:]
    return next((day for day in reversed(eligible) if day in common), None)


def forecast_path(target, log_dir=LOG_DIR):
    return log_dir / 'shadow' / f'{target}.us-forecast-v2.us.json'


def direction(change):
    return 'up' if change > 0 else 'down' if change < 0 else 'flat'


def validate_lock(forecast, locked_at):
    observed = event_time(forecast['information_cutoff_at'], 'us')
    saved = event_time(locked_at, 'us')
    opens = event_time(forecast['target_open_at'], 'us')
    closes = event_time(forecast['target_close_at'], 'us')
    target = date.fromisoformat(forecast['target_session'])
    reference = date.fromisoformat(forecast['reference_session'])
    if (not observed <= saved < opens < closes or opens.date() != target or closes.date() != target
            or not reference <= observed.date() or reference >= target or forecast['horizon_days'] not in {1, 2}):
        raise ValueError('美股预测必须在目标日开盘前冻结')


def settle_forecasts(histories, observed_at, *, log_dir=LOG_DIR, outcomes=OUTCOMES):
    now = event_time(observed_at, 'us')
    actual = {s: {row['date']: row['close'] for row in rows} for s, rows in histories.items()}
    for p in sorted((log_dir / 'shadow').glob('*.us-forecast-v2.us.json')):
        d = json.loads(p.read_text(encoding='utf-8'))
        f = d['us_forecast']
        day = f['target_session']
        dataset = d['input_dataset']
        if (p != forecast_path(day, log_dir) or d['market'] != 'us' or d['execution_mode'] != 'shadow'
                or d['execution']['orders_submitted'] != 0 or f['protocol'] != VERSION
                or d['data_sources'] != [PRICE_SOURCE] or d['dataset_fingerprint'] != canonical_hash(dataset)
                or dataset['news_screen']['forecast'] != {key: value for key, value in f.items() if key != 'locked_at'}
                or {row['symbol']: row['close'] for row in dataset['snapshot']}
                != {symbol: row['close'] for symbol, row in f['references'].items()}):
            raise ValueError('美股预测记录未通过身份核验')
        validate_lock(f, f['locked_at'])
        output = outcomes / f'{day}.json'
        if output.exists() or now < event_time(f['target_close_at'], 'us'):
            continue
        if not all(day in actual.get(s, {}) for s in f['references']):
            continue
        predictions = {row['symbol']: row for row in d['predictions']}
        if set(predictions) != set(f['references']):
            raise ValueError('预测与参考价格的标的不一致')
        scored = []
        for symbol, ref in f['references'].items():
            end = actual[symbol][day]
            realized = direction(end - ref['close'])
            predicted = predictions[symbol]['direction']
            scored.append({'symbol': symbol, 'reference_close': ref['close'], 'target_close': end,
                           'actual': realized, 'predicted': predicted, 'confidence': predictions[symbol]['confidence'],
                           'correct': predicted == realized, 'baseline_correct': ref['baseline_direction'] == realized,
                           'return_from_reference_pct': round((end/ref['close']-1)*100, 6)})
        body = {'protocol': VERSION, 'target_session': day, 'forecast_fingerprint': canonical_hash(d),
                'reference_session': f['reference_session'], 'horizon_days': f['horizon_days'], 'locked_at': f['locked_at'],
                'observed_at': now.isoformat(), 'price_source': PRICE_SOURCE, 'rows': scored}
        write_log(output, {**body, 'fingerprint': canonical_hash(body)})


def feedback(cutoff, horizon_days, *, outcomes=OUTCOMES):
    ready = []
    for p in sorted(outcomes.glob('*.json')):
        outcome = json.loads(p.read_text(encoding='utf-8'))
        identity = outcome.pop('fingerprint')
        if identity != canonical_hash(outcome) or outcome['protocol'] != VERSION:
            raise ValueError('美股评分回执指纹不一致')
        if outcome['horizon_days'] == horizon_days and event_time(outcome['observed_at'], 'us') <= event_time(cutoff, 'us'):
            ready.append(outcome)
    rows = [row for receipt in ready for row in receipt['rows']]
    metrics = calibration_metrics(rows)
    metrics.pop('bins')
    report = {'status': 'insufficient_data', 'train_count': 0, 'test_count': 0}
    if len(ready) >= 2:
        index = max(1, min(len(ready)-1, int(len(ready)*.8)))
        holdout = ready[index:]
        first_prediction = event_time(holdout[0]['locked_at'], 'us')
        train = [row for r in ready[:index] if event_time(r['observed_at'], 'us') < first_prediction
                 for row in r['rows']]
        test = [row for r in holdout for row in r['rows']]
        report.update(train_count=len(train), test_count=len(test))
        if len(train) >= 30 and len(test) >= 10:
            raw = calibration_metrics(test)['brier_score']
            calibrated = calibration_metrics(apply_calibrator(test, fit_histogram_calibrator(train)))['brier_score']
            report.update(status='comparison_complete', raw_brier=raw, candidate_brier=calibrated)
    return {**metrics, 'accuracy': sum(r['correct'] for r in rows)/len(rows) if rows else None,
            'baseline_accuracy': sum(r['baseline_correct'] for r in rows)/len(rows) if rows else None,
            'baseline': 'direction_of_last_observed_daily_change', 'calibration_review': report,
            'recent': [{'target_session': r['target_session'], 'rows': r['rows']} for r in ready[-3:]]}
