"""只读模拟账本与真实股价的展示数据；不生成交易或回填虚构收益。"""
from contextlib import closing
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import sqlite3

import pandas as pd

from config import DATA_DIR, LOG_DIR, MARKETS
from evals.experiments import canonical_hash
from evals.metrics import uses_synthetic_data
from evals.performance import indexed_curve, performance_metrics
from evals.price_data import normalize_history

PRICE_CACHE = DATA_DIR / "market_views" / "cn-prices.json"


def analysis_logs(root=LOG_DIR):
    rows = []
    for path in [*Path(root).glob("*.cn.json"), *(Path(root) / "shadow").glob("*.cn.json"), *(Path(root) / "after_close").glob("*.cn.json")]:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("market") == "cn" and not uses_synthetic_data(row):
                rows.append(row)
        except (ValueError, OSError):
            continue
    return sorted(rows, key=lambda row: (row.get("as_of_date") or row.get("date", ""), row.get("generated_at", "")))


def read_ledger(path=MARKETS["cn"]["db"]):
    if not Path(path).exists():
        return {"cash": MARKETS["cn"]["initial_cash"], "positions": [], "trades": [], "exists": False}
    with closing(sqlite3.connect(f"file:{Path(path).resolve().as_posix()}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        result = {"cash": float(conn.execute("SELECT value FROM meta WHERE key='cash'").fetchone()[0]),
                "positions": [dict(row) for row in conn.execute("SELECT * FROM positions WHERE shares>0")],
                "trades": [dict(row) for row in conn.execute("SELECT * FROM trades ORDER BY trade_date,id")], "exists": True}
        from agent.corporate_actions import read_actions
        result['corporate_actions']=read_actions(conn)
        tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for key in ('corporate_settlements','corporate_refreshes'):
            result[key]=[dict(r) for r in conn.execute('SELECT * FROM '+key+' ORDER BY id')] if key in tables else []
        if result['corporate_settlements']:
            result['as_of_date']=max([result['corporate_settlements'][-1]['through_date']]+[r['trade_date'] for r in result['trades']])
        return result


def tracking_symbols(logs, ledger, opportunity=None):
    """Keep previously researched companies visible when today's focus changes."""
    opportunity = opportunity or {}
    allowed = {row["symbol"] for row in MARKETS["cn"]["candidate_universe"]}
    symbols = set(MARKETS["cn"].get("investment_watchlist", []))
    for log in logs:
        if uses_synthetic_data(log):
            continue
        symbols.update(row["symbol"] for row in log.get("decisions", []))
    symbols.update(opportunity.get("active_symbols", []))
    symbols.update(opportunity.get("research_watchlist", []))
    symbols.update(row["symbol"] for row in opportunity.get("forward_observations", []))
    symbols.update(row["symbol"] for row in ledger.get("positions", []) + ledger.get("trades", []))
    return sorted(symbols & allowed)


def _journal(ledger, initial):
    from agent.corporate_actions import replay
    state=replay(ledger,initial)
    return state['cash'],{p['symbol']:p['shares'] for p in state['positions']},state['realized_pnl']


def build_view(payload, ledger, logs, initial=1_000_000):
    if payload.get("market") != "cn" or payload.get("data_quality") != "live_prices":
        raise ValueError("交易展示需要明确标注的真实A股价格缓存")
    if canonical_hash(payload["histories"]) != payload.get("fingerprint"):
        raise ValueError("价格缓存指纹不符")
    histories = {symbol: normalize_history(symbol, pd.DataFrame(rows)).to_dict("records")
                 for symbol, rows in payload["histories"].items()}
    if not histories or any(not rows for rows in histories.values()):
        raise ValueError("价格缓存为空")
    end = min(rows[-1]["date"] for rows in histories.values())
    if end > date.today().isoformat():
        raise ValueError("价格缓存包含未来日期")
    prices = {symbol: {row["date"]: row["close"] for row in rows} for symbol, rows in histories.items()}
    raw=payload.get('valuation_histories',{})
    if raw:
        if payload.get('valuation_price_basis')!='unadjusted' or canonical_hash(raw)!=payload.get('valuation_fingerprint'):
            raise ValueError('未复权估值缓存口径或指纹不符')
        raw={s:normalize_history(s,pd.DataFrame(r)).to_dict('records') for s,r in raw.items()}
    elif payload.get('price_basis')=='unadjusted':raw=histories
    if raw:
        if set(raw)!=set(histories) or any(raw[s][-1]['date']!=histories[s][-1]['date'] for s in histories):
            raise ValueError('观察行情与未复权行情的股票或日期不一致')
        histories=raw  # Display actual market prices so historical fill markers keep their price units.
        prices={s:{r['date']:r['close'] for r in rows} for s,rows in histories.items()}
    valuation_prices={s:{r['date']:r['close'] for r in rows} for s,rows in raw.items()}
    from agent.corporate_actions import replay,receivable_value,VERSION as EQUITY_VERSION
    from agent.corporate_settlement import coverage_ok
    valid_logs = [row for row in logs if not uses_synthetic_data(row) and (row.get("as_of_date") or row.get("date", "")) <= end]
    latest = valid_logs[-1] if valid_logs else {}
    observations = [row.get("as_of_date") or row.get("date") for row in valid_logs]
    warnings = []
    if latest.get("timing") and latest.get("execution_mode") == "shadow" and not latest.get("news_screen", {}).get("theme_discovery"):
        warnings.append("本轮为固定受控公司样本验收，未由新闻发现流程自动入选；只分析，不代表建仓。")
    journal_ok = True
    actual_quantities = {row["symbol"]: row["shares"] for row in ledger["positions"]}
    try:
        cash, quantities, realized = _journal(ledger, initial)
        if abs(cash - ledger["cash"]) > .02 or quantities != actual_quantities:
            raise ValueError("成交记录无法与当前现金/持仓对账")
    except ValueError as exc:
        journal_ok, realized = False, None
        warnings.append(str(exc) + "；只展示当前估值，不回填历史组合曲线")
    trade_dates = [row["trade_date"] for row in ledger["trades"]]
    start = min(observations + trade_dates) if observations or trade_dates else end
    if not journal_ok:
        start = end
    if start < min(rows[0]["date"] for rows in histories.values()):
        warnings.append("价格窗口未覆盖全部跟踪期，曲线与回撤仅代表有完整价格的区间")
    days = sorted({row["date"] for rows in histories.values() for row in rows if start <= row["date"] <= end})
    points, missing = [], set()
    for day in days:
        if journal_ok:
            # 只有实际账本成交会改变持仓。HOLD/预测均不会成为买入。
            state = replay(ledger,initial,day)
            daily_cash, held = state['cash'],{p['symbol']:p['shares'] for p in state['positions']}
        else:
            daily_cash, held = ledger["cash"], actual_quantities
            state={'rights':[],'cash_receivable':0}
        pending={r['symbol'] for r in state['rights'] if r['bonus_shares']}
        missing_today = {symbol for symbol in set(held)|pending if day not in valuation_prices.get(symbol, {})}
        if missing_today:
            missing |= missing_today
            continue
        day_prices={s:r[day] for s,r in valuation_prices.items() if day in r}
        value = daily_cash + sum(quantity * day_prices[symbol] for symbol, quantity in held.items())+receivable_value(state,day_prices)
        points.append({"date": day, "total": round(value, 2),
                       'equity_accounting_complete':bool(raw) and coverage_ok(ledger,day)})
    if missing:
        warnings.append("持仓缺少对应日期真实价格，跳过不完整估值：" + ", ".join(sorted(missing)))
    curve = indexed_curve(points, initial) if journal_ok else []
    flags={p['date']:p['equity_accounting_complete'] for p in points}
    for p in curve:p['equity_accounting_complete']=flags[p['date']]
    current_complete = bool(points and points[-1]["date"] == end)
    if ledger.get('as_of_date','')>end:
        current_complete=False
        warnings.append('权益结算晚于缓存行情日，当前净盈亏等待对应未复权收盘价')
    if (trade_dates and max(trade_dates) > end) or any(str(row.get("last_buy_date") or "") > end for row in ledger["positions"]):
        current_complete = False
        warnings.append("存在晚于缓存价格的成交，当前盈亏尚不能估值")
    positions = []
    for row in ledger["positions"]:
        close = valuation_prices.get(row["symbol"], {}).get(end) if end >= str(row.get("last_buy_date") or "") else None
        value = close * row["shares"] if close is not None else None
        cost = row["avg_cost"] * row["shares"]
        positions.append({**row, "last_price": close, "market_value": round(value, 2) if value is not None else None,
                          "unrealized_pnl": round(value - cost, 2) if value is not None else None})
    names = {row["symbol"]: row["name"] for row in MARKETS["cn"]["candidate_universe"]}
    ending=replay(ledger,initial,end) if journal_ok else {'rights':[],'cash_receivable':None,'dividend_income':None}
    stocks = []
    decisions, predictions, cases, studied = {}, {}, {}, {}
    # Each company keeps its own last judgment. Shadow research cannot replace a
    # formal paper judgment on the same date; the actual ledger remains authoritative.
    for log in sorted(valid_logs, key=lambda r: (r.get("as_of_date") or r.get("date", ""),
                      {"paper": 2, "pending": 1}.get(r.get("execution_mode"), 0), r.get("generated_at", ""))):
        for decision in log.get("decisions", []):
            symbol = decision["symbol"]
            decisions[symbol] = decision
            predictions[symbol] = next((r for r in log.get("predictions", []) if r["symbol"] == symbol), None)
            cases[symbol] = next((r for r in log.get("investment_cases", []) if r["symbol"] == symbol), None)
            studied[symbol] = {"analysis_date": log.get("as_of_date") or log.get("date"),
                               "decision_at": log.get("timing", {}).get("decision_at") or log.get("generated_at"),
                               "research_mode": log.get("execution_mode", "unknown")}
    for symbol, rows in histories.items():
        frame = pd.DataFrame(rows)
        ma20, ma60 = frame["close"].rolling(20).mean(), frame["close"].rolling(60).mean()
        series = [{**row, "ma20": round(float(ma20.iloc[index]), 4) if pd.notna(ma20.iloc[index]) else None,
                   "ma60": round(float(ma60.iloc[index]), 4) if pd.notna(ma60.iloc[index]) else None}
                  for index, row in enumerate(rows) if row["date"] <= end]
        last, previous = series[-1], series[-2] if len(series) > 1 else series[-1]
        base = prices[symbol].get(start)
        benchmark = [{"date": row["date"], "return_pct": round((row["close"] / base - 1) * 100, 4)}
                     for row in series if base is not None and row["date"] >= start]
        if base is None:
            warnings.append(symbol + "缺少跟踪起点价格，不补造买入持有对照")
        stocks.append({"symbol": symbol, "name": names.get(symbol, symbol), "series": series,
            **studied.get(symbol, {"analysis_date": None, "decision_at": None, "research_mode": "unreviewed"}),
            "tracking_status": "held" if symbol in actual_quantities else 'research_rejected' if (cases.get(symbol) or {}).get('research_status')=='rejected' else "research_pending" if cases.get(symbol, {}) and cases[symbol].get("research_status") == "pending" else "observing" if symbol in decisions else "unreviewed",
            "actual_weight": round((sum(p["market_value"] or 0 for p in positions if p["symbol"] == symbol)+sum(r['bonus_shares']*valuation_prices[symbol][end] for r in ending['rights'] if r['symbol']==symbol and r['bonus_shares'])) / points[-1]["total"], 6) if current_complete and points[-1]["total"] > 0 else None,
            "last_close": last["close"], "daily_change_pct": round((last["close"] / previous["close"] - 1) * 100, 3),
            "decision": decisions.get(symbol), "prediction": predictions.get(symbol),
            "investment_case": cases.get(symbol),
            "tracking_benchmark": benchmark,
            "trades": [row for row in ledger["trades"] if row["symbol"] == symbol and row["trade_date"] in prices[symbol]]})
    fees = sum(float(row.get("fee") or 0) for row in ledger["trades"])
    slippage = sum(float(row.get("slippage_cost") or 0) for row in ledger["trades"])
    total = points[-1]["total"] if current_complete else None
    pending_pnl=[]
    for right in ending['rights']:
        if right['bonus_shares']:
            px=valuation_prices.get(right['symbol'],{}).get(end)
            pending_pnl.append(right['bonus_shares']*px-right['basis'] if px is not None else None)
    unrealized=[p['unrealized_pnl'] for p in positions]+pending_pnl
    if not raw and ledger['trades']:warnings.append('缺少未复权估值价格，不用前复权行情回填持仓盈亏')
    equity_ready=bool(raw) and coverage_ok(ledger,end) and current_complete
    equity={'version':EQUITY_VERSION,'status':'ready' if equity_ready else 'awaiting_verified_data',
            'valuation_price_basis':'unadjusted' if raw else 'unavailable','dividend_tax_basis':'gross_before_personal_dividend_tax',
            'cash_receivable':ending['cash_receivable'],'dividend_income':ending['dividend_income'],
            'pending_shares':sum(r['bonus_shares'] for r in ending['rights']),
            'settled_through':(ledger.get('corporate_settlements') or [{}])[-1].get('through_date'),
            'message':'未复权价格加已核验的应收/到账权益；股息税前，成交费用已扣。' if equity_ready else
                      '权益来源覆盖或未复权估值尚不齐备，持仓区间暂停严格全收益评分。'}
    from agent.investment import POLICY, ResearchBook
    try:
        research_tasks = ResearchBook().tasks()
    except sqlite3.Error:
        research_tasks = []
    return {"status": "ready", "market": "cn", "as_of_date": end, "fetched_at": payload.get("fetched_at"),
        "source": payload["source"], "price_basis": 'unadjusted' if raw else payload["price_basis"], "fingerprint": payload.get('valuation_fingerprint',payload["fingerprint"]) if raw else payload["fingerprint"],
        "tracking_start": start, "analysis_date": latest.get("as_of_date") or latest.get("date"),
        "investment_policy": POLICY, "analysis_policy": latest.get("investment_policy", "legacy-short-term"),
        "research_tasks": research_tasks,
        "stocks": stocks, "positions": positions, "trades": ledger["trades"], "portfolio_curve": curve,
        'equity_accounting':equity,
        "portfolio": {"initial_cash": initial, "cash": ledger["cash"], "total_assets": total,
            'dividend_income':ending['dividend_income'],'cash_receivable':ending['cash_receivable'],
            "net_pnl": round(total - initial, 2) if total is not None and journal_ok else None,
            "net_return_pct": round((total / initial - 1) * 100, 4) if total is not None and journal_ok else None,
            "realized_pnl": round(realized, 2) if realized is not None else None,
            "unrealized_pnl": round(sum(unrealized),2) if all(v is not None for v in unrealized) and current_complete else None,
            "total_fees": round(fees, 2), "total_slippage": round(slippage, 2), "trade_count": len(ledger["trades"]),
            "position_count": len(positions), "journal_reconciled": journal_ok},
        "metrics": performance_metrics(points if journal_ok else [], initial), "warnings": sorted(set(warnings)),
        "limitations": ["组合只按实际模拟成交重建，历史股价不等于Agent历史收益；未成交建议不算买入。",
            "成交价已包含滑点，收益已扣实际费用，不重复减滑点。",
            "观察股票对照为不计费的价格涨跌，不代表模拟成交或可实现收益。",
            "股价观察使用所示复权口径；组合估值仅接受未复权价及独立权益记录，不重复计算分红。",
            "股息按税前模拟，暂不模拟个人持仓期限差别税；单日记录不足以评价策略表现。"]}


def load_view(cache=PRICE_CACHE, log_dir=LOG_DIR, db=None):
    from agent.opportunities import OpportunityBook
    if not Path(cache).exists():
        return {"status": "no_data", "message": "尚未缓存真实股价，请先运行 scripts/refresh_trading_view.py", "opportunities": OpportunityBook().view()}
    payload = json.loads(Path(cache).read_text(encoding="utf-8"))
    logs = analysis_logs(log_dir)
    result = build_view(payload, read_ledger(db or MARKETS["cn"]["db"]), logs)
    result["opportunities"] = OpportunityBook().view(payload.get("histories", {}))
    if logs and logs[-1].get("timing"):
        from agent.order_queue import OrderQueue
        from agent.trading_clock import now
        result["timing"] = deepcopy(logs[-1]["timing"])
        result["pending_orders"] = OrderQueue().public_state(now().date().isoformat())
        result["analysis_date"] = (logs[-1]["timing"].get("decision_at") or logs[-1]["timing"]["information_cutoff_at"])[:10]
    if logs and logs[-1].get("execution_policy"):
        result["execution_policy"] = logs[-1]["execution_policy"]
    if logs and logs[-1].get("learning"):
        from agent.learning import public_summary
        result["learning"] = public_summary(logs[-1]["learning"])
    from agent.research_quality import QualityBook
    QualityBook().annotate(result, logs)
    from agent.trading_clock import now
    from evals.portfolio_comparison import ComparisonBook, build_comparison
    try:
        result['comparison'] = build_comparison(ComparisonBook().context(now().isoformat()), result['as_of_date'],
                                                result['portfolio']['journal_reconciled'])
    except (ValueError, OSError, KeyError):
        result['comparison'] = {'status':'record_invalid','message':'组合基准记录未通过核对，暂不显示相对表现。','series':[]}
    return result


def render_html(view):
    template = Path(__file__).resolve().parent.parent / "dashboard" / "templates" / "trading.html"
    encoded = json.dumps(view, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return template.read_text(encoding="utf-8").replace("__VIEW_DATA__", encoded)
