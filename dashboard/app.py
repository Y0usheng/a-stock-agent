# -*- coding: utf-8 -*-
"""
Web 仪表盘
==========
本地网站查看模拟盘结果:资产曲线、持仓、每日决策、预测命中率。
数据直接读账本 SQLite + logs/decisions/ 日志(仪表盘是"只读视图",
不走 MCP——查看数据无需 agent 参与,这也是架构上的合理分工)。

用法:
    python dashboard/app.py
    浏览器打开 http://127.0.0.1:5000
"""
import sys
import json
import sqlite3
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EXPERIMENT_DB, MARKETS, LOG_DIR, NEWS_DB
from evals.calibration import variant_calibration
from evals.experiments import comparable_experiments, list_experiments, decision_log_variant
from evals.metrics import (EVALUATION_HISTORY_DAYS,
                           actual_direction_from_frame,
                           prediction_start_date,
                           uses_synthetic_data)
from evals.performance import indexed_curve, load_valuation_snapshots, performance_report
from evals.news_store import dataset_summary
from agent.operations import operation_summary

from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/api/opportunities")
def opportunities_data():
    from agent.opportunities import OpportunityBook
    try:
        return jsonify(OpportunityBook().view())
    except (ValueError, OSError, sqlite3.Error) as exc:
        from agent.errors import safe_error
        return jsonify({"status": "error", "message": safe_error(exc)}), 503


@app.route("/api/trading-view")
def trading_view_data():
    from evals.trading_view import load_view
    try:
        return jsonify(load_view())
    except (ValueError, OSError, sqlite3.Error) as exc:
        from agent.errors import safe_error
        return jsonify({"status": "error", "message": safe_error(exc)}), 503


@app.route("/trading")
def trading_view_page():
    from evals.trading_view import load_view, render_html
    try:
        return render_html(load_view())
    except (ValueError, OSError, sqlite3.Error) as exc:
        from agent.errors import safe_error
        return render_html({"status": "error", "message": safe_error(exc)}), 503


@app.route("/api/operations")
def operations():
    return jsonify(operation_summary())


def _positions(market: str) -> tuple[list, float]:
    db = MARKETS[market]["db"]
    if not db.exists():
        return [], MARKETS[market]["initial_cash"]
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    pos = [dict(r) for r in conn.execute("SELECT * FROM positions WHERE shares>0")]
    cash = conn.execute("SELECT value FROM meta WHERE key='cash'").fetchone()[0]
    conn.close()
    return pos, cash


def _trades(market: str, limit: int = 50) -> list:
    db = MARKETS[market]["db"]
    if not db.exists():
        return []
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))]
    conn.close()
    return rows


def _logs(market: str, include_shadow=False) -> list:
    out = []
    paths = [*LOG_DIR.glob(f"*.{market}.json")]
    if include_shadow:
        paths += list((LOG_DIR / "shadow").glob(f"*.{market}.json"))
    for f in sorted(paths):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            if d.get("market") == market:
                out.append(d)
        except Exception:
            pass
    out.sort(key=lambda item: (str(item.get("as_of_date") or item.get("date") or ""), item.get("generated_at", "")))
    return out


@app.route("/api/overview")
def overview():
    data = {}
    for mkt, m in MARKETS.items():
        pos, cash = _positions(mkt)
        logs = _logs(mkt, include_shadow=True)
        trades = _trades(mkt, 30)
        last_log = logs[-1] if logs else None
        price_as_of = str((last_log or {}).get("as_of_date") or
                          (last_log or {}).get("date") or "")
        # 只允许行情截面不早于持仓买入日，避免用旧行情给新持仓估值。
        last_prices = {}
        if last_log:
            for d in last_log.get("decisions", []):
                if d.get("symbol") and d.get("price"):
                    last_prices[d["symbol"]] = d["price"]
            for p in last_log.get("account_after", {}).get("positions", []):
                if p.get("shares") and p.get("market_value") is not None:
                    last_prices[p["symbol"]] = p["market_value"] / p["shares"]
        mv = 0.0
        invested_cost = 0.0
        stale_symbols = []
        for p in pos:
            can_use_snapshot = bool(price_as_of and
                                    price_as_of >= str(p.get("last_buy_date") or ""))
            px = last_prices.get(p["symbol"]) if can_use_snapshot else None
            if px is None:
                px = p["avg_cost"]
                stale_symbols.append(p["symbol"])
                p["price_source"] = "cost_fallback"
            else:
                p["price_source"] = "latest_log"
            p["last_price"] = px
            p["market_value"] = round(px * p["shares"], 2)
            p["cost_value"] = round(p["avg_cost"] * p["shares"], 2)
            p["pnl_amount"] = round(p["market_value"] - p["cost_value"], 2)
            p["pnl_pct"] = round((px / p["avg_cost"] - 1) * 100, 2)
            mv += p["market_value"]
            invested_cost += p["cost_value"]
        total = round(cash + mv, 2)
        for p in pos:
            p["weight_pct"] = round(p["market_value"] / total * 100, 2) if total else 0.0

        # 优先使用 SQLite 持久化每日估值；旧数据才回退到 JSON 日志。
        curve_by_date = {}
        valuation_snapshots = load_valuation_snapshots(m["db"])
        if valuation_snapshots:
            for item in valuation_snapshots:
                day = str(item.get("snapshot_date") or "")
                value = item.get("total_assets")
                if day and value:
                    curve_by_date[day] = {"date": day, "total": value, "estimated": False}
        else:
            for item in logs:
                day = str(item.get("as_of_date") or item.get("date") or "")
                value = item.get("account_after", {}).get("total_assets")
                if day and value:
                    curve_by_date[day] = {"date": day, "total": value, "estimated": False}
        ledger_dates = [str(t.get("trade_date") or "") for t in trades if t.get("trade_date")]
        valuation_date = max(ledger_dates + ([price_as_of] if price_as_of else []) or
                             [date.today().isoformat()])
        if not curve_by_date or valuation_date > max(curve_by_date):
            curve_by_date[valuation_date] = {
                "date": valuation_date, "total": total, "estimated": bool(stale_symbols),
            }
        curve = indexed_curve(list(curve_by_date.values()), m["initial_cash"])
        if curve:
            curve[-1]["estimated"] = bool(curve_by_date[curve[-1]["date"]].get("estimated"))
        previous_total = curve[-2]["total"] if len(curve) > 1 else m["initial_cash"]
        latest_change = round(total - previous_total, 2)
        latest_change_pct = round((total / previous_total - 1) * 100, 2) if previous_total else None
        total_fees = round(sum(float(t.get("fee") or 0) for t in trades), 2)
        unrealized_pnl = round(sum(p["pnl_amount"] for p in pos), 2)
        warning = None
        if stale_symbols:
            warning = "部分持仓缺少买入日之后的行情，暂按成本估值: " + ", ".join(stale_symbols)
        data[mkt] = {
            "name": m["name"], "currency": m["currency"],
            "cash": round(cash, 2), "positions": pos,
            "total_assets": total, "initial_cash": m["initial_cash"],
            "return_pct": round((total / m["initial_cash"] - 1) * 100, 2),
            "profit_amount": round(total - m["initial_cash"], 2),
            "latest_change": latest_change,
            "latest_change_pct": latest_change_pct,
            "market_value": round(mv, 2),
            "invested_cost": round(invested_cost, 2),
            "cash_pct": round(cash / total * 100, 2) if total else 0.0,
            "unrealized_pnl": unrealized_pnl,
            "fees": total_fees,
            "curve": curve,
            "trades": trades,
            "last_log": last_log,
            "price_as_of": price_as_of or None,
            "valuation_date": valuation_date,
            "data_warning": warning,
            "valuation_source": "sqlite_snapshots" if valuation_snapshots else "decision_logs",
        }
    return jsonify(data)


@app.route("/api/score")
def score():
    """预测命中率(需要拉历史行情,较慢,前端按钮触发)"""
    sys.path.insert(0, str(Path(__file__).parent.parent / "servers"))
    import market_server, us_market_server
    DATA = {"cn": market_server, "us": us_market_server}
    stats = {"cn": {"total": 0, "hit": 0, "pending": 0},
             "us": {"total": 0, "hit": 0, "pending": 0}}
    rows = []
    for mkt in MARKETS:
        for d in _logs(mkt):
            for p in d.get("predictions", []):
                try:
                    fetch = DATA[mkt]._mock_history if uses_synthetic_data(d) else DATA[mkt]._fetch_history
                    df = fetch(p["symbol"], EVALUATION_HISTORY_DAYS)
                except Exception:
                    continue
                start = prediction_start_date(d)
                if not start:
                    continue
                act = actual_direction_from_frame(df, start, p.get("horizon_days", 5))
                if act is None:
                    stats[mkt]["pending"] += 1
                    continue
                ok = act == p["direction"]
                stats[mkt]["total"] += 1
                stats[mkt]["hit"] += ok
                sdk = d.get("agent_sdk") or {}
                rows.append({"date": start, "market": mkt, "symbol": p["symbol"],
                             "pred": p["direction"], "actual": act, "hit": ok,
                             "correct": ok, "confidence": p.get("confidence"),
                             "variant": decision_log_variant(d),
                             "prompt_version": (sdk.get("prompt") or {}).get("version", "unversioned")})
    return jsonify({"stats": stats, "rows": rows[-100:], **variant_calibration(rows)})


@app.route("/api/experiments")
def experiments():
    """只读返回实验注册表和真实历史新闻覆盖情况。"""
    from evals.research import ResearchBook
    return jsonify({"runs": list_experiments(EXPERIMENT_DB, 30),
                    "comparisons": comparable_experiments(EXPERIMENT_DB),
                    "news_dataset": dataset_summary(NEWS_DB), "research": ResearchBook(EXPERIMENT_DB).summaries()})


@app.route("/api/performance")
def performance():
    """收益风险指标与无未来泄漏时变候选基准；真实行情可能较慢。"""
    sys.path.insert(0, str(Path(__file__).parent.parent / "servers"))
    import market_server, us_market_server
    sources = {"cn": market_server, "us": us_market_server}
    reports = {}
    for market, config in MARKETS.items():
        logs = _logs(market)
        reports[market] = performance_report(
            logs, sources[market], config.get("candidate_universe", config["watchlist"]),
            valuation_snapshots=load_valuation_snapshots(config["db"]),
            trades=_trades(market, 10000),
        )
    return jsonify(reports)


PAGE = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>模拟盘仪表盘</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
:root { --bg:#0f1117; --card:#181c25; --border:#262b38; --text:#e6e9f0;
        --muted:#8b93a7; --green:#2ecc71; --red:#e74c3c; --accent:#4f8ef7; }
* { box-sizing:border-box; margin:0; }
body { background:var(--bg); color:var(--text);
       font:14px/1.6 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif; padding:24px; }
h1 { font-size:20px; margin-bottom:4px; }
.sub { color:var(--muted); font-size:12px; margin-bottom:20px; }
.toolbar { display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }
.toolbar-actions { display:flex; gap:10px; align-items:center; }
.grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media (max-width:900px){ .grid{ grid-template-columns:1fr; } }
.card { background:var(--card); border:1px solid var(--border);
        border-radius:12px; padding:18px; }
.card h2 { font-size:15px; margin-bottom:12px; display:flex;
           justify-content:space-between; align-items:baseline; }
.big { font-size:26px; font-weight:700; }
.pos { color:var(--green); } .neg { color:var(--red); }
.muted { color:var(--muted); font-size:12px; font-weight:400; }
.kpis { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin:12px 0; }
.kpi { background:#121620; border:1px solid var(--border); border-radius:9px; padding:9px 10px; }
.kpi b { display:block; font-size:16px; margin-top:2px; }
.warning { color:#f3bd63; background:#332816; border:1px solid #59431e;
           padding:7px 10px; border-radius:8px; margin:9px 0; font-size:12px; }
.views { display:grid; gap:7px; }
.view { padding:8px 10px; border-left:3px solid var(--accent); background:#121620; }
.view b { margin-right:8px; }
.newslist { display:grid; gap:7px; margin:8px 0 12px; }
.newsitem { padding:8px 10px; border:1px solid var(--border); border-radius:8px; background:#121620; }
.score { display:inline-block; min-width:52px; font-weight:700; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; color:var(--muted); font-weight:500; padding:6px 8px;
     border-bottom:1px solid var(--border); }
td { padding:6px 8px; border-bottom:1px solid var(--border); }
tr:last-child td { border-bottom:none; }
.tag { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; }
.tag.BUY { background:#173527; color:var(--green); }
.tag.SELL { background:#3a1d1d; color:var(--red); }
.tag.HOLD { background:#232a3a; color:var(--muted); }
button { background:var(--accent); color:#fff; border:none; border-radius:8px;
         padding:8px 16px; cursor:pointer; font-size:13px; }
button:disabled { opacity:.5; }
canvas { max-height:220px; }
.full { grid-column:1 / -1; }
@media (max-width:620px){ body{padding:14px}.kpis{grid-template-columns:1fr 1fr}.toolbar{display:block}.toolbar-actions{margin:10px 0} }
</style>
</head>
<body>
<div class="toolbar"><div><h1>📈 双市场投资收益仪表盘</h1>
<div class="sub">A股 + 美股 AI Agent 模拟交易 · 聚焦收益、变动、仓位与风险 · 仅为学习项目,不构成投资建议</div></div>
<div class="toolbar-actions"><a href="/trading" style="color:#70a3ff">模拟交易与真实股价 ↗</a><span id="updated" class="muted">等待刷新</span><button id="refreshBtn" onclick="load()">刷新数据</button></div></div>
<div class="card" style="margin-bottom:16px"><h2>运行状态</h2><div id="operations">等待刷新</div></div>
<div class="grid" id="root"><div class="card">加载中…</div></div>

<template id="tpl">
<div class="card">
  <h2><span class="mname"></span><span class="ret big"></span></h2>
  <div class="muted assets"></div>
  <div class="warning datawarning" style="display:none"></div>
  <div class="kpis">
    <div class="kpi"><span class="muted">累计盈亏</span><b class="profit"></b></div>
    <div class="kpi"><span class="muted">最新变动</span><b class="change"></b></div>
    <div class="kpi"><span class="muted">现金占比</span><b class="cashpct"></b></div>
    <div class="kpi"><span class="muted">持仓市值</span><b class="marketvalue"></b></div>
    <div class="kpi"><span class="muted">浮动盈亏</span><b class="unrealized"></b></div>
    <div class="kpi"><span class="muted">累计费用</span><b class="fees"></b></div>
  </div>
  <h2>资产与区间盈亏 <span class="muted valuationdate"></span></h2>
  <canvas class="curve"></canvas>
  <h2 style="margin-top:14px">新闻动态候选池 <span class="muted newsprovider"></span></h2>
  <div class="newslist"></div>
  <h2 style="margin-top:14px">持仓</h2>
  <table class="postable"><thead><tr>
    <th>名称</th><th>股数</th><th>成本</th><th>现价</th><th>市值</th><th>权重</th><th>浮盈亏</th>
  </tr></thead><tbody></tbody></table>
  <h2 style="margin-top:14px">最近成交</h2>
  <table class="tradetable"><thead><tr>
    <th>日期</th><th>操作</th><th>标的</th><th>股数</th><th>价格</th><th>费用</th>
  </tr></thead><tbody></tbody></table>
  <h2 style="margin-top:14px">最近决策 <span class="muted logdate"></span></h2>
  <table class="dectable"><thead><tr>
    <th>操作</th><th>标的</th><th>股数</th><th>价格</th><th>理由</th>
  </tr></thead><tbody></tbody></table>
  <h2 style="margin-top:14px">多角色分析</h2>
  <div class="views"></div>
</div>
</template>

<div class="card full" style="margin-top:16px">
  <h2>🎯 预测命中率(evals)
    <button id="scoreBtn" onclick="loadScore()">计算命中率</button></h2>
  <div id="scoreBox" class="muted">点击按钮计算(需拉取行情,稍慢)</div>
</div>

<div class="card full" style="margin-top:16px">
  <h2>⚖️ 收益风险 Evals
    <button id="perfBtn" onclick="loadPerformance()">计算收益风险</button></h2>
  <div id="perfBox" class="muted">累计收益、时变候选基准、波动、回撤、Sharpe、Sortino 与换手率</div>
  <div class="grid" style="margin-top:14px"><div><h2>组合 vs 时变候选基准</h2><canvas id="perfCurve"></canvas></div>
  <div><h2>组合回撤</h2><canvas id="drawdownCurve"></canvas></div></div>
</div>

<div class="card full" style="margin-top:16px">
  <h2>🧪 第三阶段实验台
    <button id="expBtn" onclick="loadExperiments()">刷新实验</button></h2>
  <div id="experimentBox" class="muted">加载历史新闻覆盖、Prompt/模型版本和可复现实验记录…</div>
</div>

<script>
const fmt = (n, c) => (c === "USD" ? "$" : "¥") + Number(n).toLocaleString();
const esc = (v) => String(v ?? "").replace(/[&<>"']/g,
  ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[ch]);
const signed = (n, suffix="") => `${Number(n) >= 0 ? "+" : ""}${Number(n).toLocaleString()}${suffix}`;
const tone = (n) => Number(n) >= 0 ? "pos" : "neg";
const marketCharts = {};

async function load() {
  loadOperations();
  const btn = document.getElementById("refreshBtn");
  const root = document.getElementById("root");
  btn.disabled = true; btn.textContent = "刷新中…";
  try {
    const response = await fetch("/api/overview");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    Object.values(marketCharts).forEach(chart => chart.destroy());
    root.innerHTML = "";
    for (const mkt of ["cn", "us"]) {
      const d = data[mkt];
      const node = document.getElementById("tpl").content.cloneNode(true);
      node.querySelector(".mname").textContent = d.name + " (" + mkt + ")";
      const ret = node.querySelector(".ret");
      ret.textContent = signed(d.return_pct, "%");
      ret.className = "ret big " + tone(d.return_pct);
      node.querySelector(".assets").textContent =
        "总资产 " + fmt(d.total_assets, d.currency) + " · 现金 " + fmt(d.cash, d.currency)
        + " · 初始 " + fmt(d.initial_cash, d.currency);
      node.querySelector(".profit").textContent = signed(d.profit_amount);
      node.querySelector(".profit").className = "profit " + tone(d.profit_amount);
      node.querySelector(".change").textContent = signed(d.latest_change) +
        (d.latest_change_pct == null ? "" : ` (${signed(d.latest_change_pct, "%")})`);
      node.querySelector(".change").className = "change " + tone(d.latest_change);
      node.querySelector(".cashpct").textContent = d.cash_pct.toFixed(2) + "%";
      node.querySelector(".marketvalue").textContent = fmt(d.market_value, d.currency);
      node.querySelector(".unrealized").textContent = signed(d.unrealized_pnl);
      node.querySelector(".unrealized").className = "unrealized " + tone(d.unrealized_pnl);
      node.querySelector(".fees").textContent = fmt(d.fees, d.currency);
      node.querySelector(".valuationdate").textContent = "估值日 " + d.valuation_date +
        (d.price_as_of ? " · 行情 " + d.price_as_of : "");
      if (d.data_warning) {
        const warning = node.querySelector(".datawarning");
        warning.textContent = "⚠ " + d.data_warning; warning.style.display = "block";
      }

      const ptb = node.querySelector(".postable tbody");
      if (!d.positions.length) ptb.innerHTML = "<tr><td colspan=7 class=muted>空仓</td></tr>";
      for (const p of d.positions) {
        ptb.insertAdjacentHTML("beforeend",
          `<tr><td>${esc(p.name)}<span class=muted> ${esc(p.symbol)}</span></td><td>${p.shares}</td>` +
          `<td>${p.avg_cost.toFixed(2)}</td><td>${p.last_price.toFixed(2)}${p.price_source === "cost_fallback" ? "*" : ""}</td>` +
          `<td>${p.market_value.toLocaleString()}</td><td>${p.weight_pct.toFixed(2)}%</td>` +
          `<td class=${tone(p.pnl_amount)}>${signed(p.pnl_amount)}<div class=muted>${signed(p.pnl_pct, "%")}</div></td></tr>`);
      }

      const ttb = node.querySelector(".tradetable tbody");
      if (!d.trades.length) ttb.innerHTML = "<tr><td colspan=6 class=muted>暂无成交</td></tr>";
      for (const trade of d.trades.slice(0, 8)) {
        const action = trade.action === "SELL" ? "SELL" : "BUY";
        ttb.insertAdjacentHTML("beforeend", `<tr><td>${esc(trade.trade_date)}</td>` +
          `<td><span class="tag ${action}">${action}</span></td><td>${esc(trade.name)}<span class=muted> ${esc(trade.symbol)}</span></td>` +
          `<td>${trade.shares}</td><td>${Number(trade.price).toFixed(2)}</td><td>${Number(trade.fee || 0).toFixed(2)}</td></tr>`);
      }

      const dtb = node.querySelector(".dectable tbody");
      const log = d.last_log;
      if (log) {
        node.querySelector(".logdate").textContent = (log.as_of_date || log.date) + " · " + (log.mode || "");
        for (const dec of log.decisions || []) {
          const action = ["BUY", "SELL", "HOLD"].includes(dec.action) ? dec.action : "HOLD";
          dtb.insertAdjacentHTML("beforeend",
            `<tr><td><span class="tag ${action}">${action}</span></td>` +
            `<td>${esc(dec.name)}<span class=muted> ${esc(dec.symbol)}</span></td>` +
            `<td>${dec.shares || "-"}</td><td>${dec.price}</td><td class=muted>${esc(dec.reason)}</td></tr>`);
        }
        const roleNames = {technical:"技术", fundamental:"基本面", risk:"风险"};
        const views = node.querySelector(".views");
        for (const [role, label] of Object.entries(roleNames)) {
          const content = log.analyst_views?.[role];
          if (content) views.insertAdjacentHTML("beforeend", `<div class=view><b>${label}</b>${esc(content)}</div>`);
        }
        if (!views.children.length) views.innerHTML = "<div class=muted>本次日志没有多角色分析</div>";
        const screen = log.news_screen || {};
        node.querySelector(".newsprovider").textContent = screen.provider_strategy || "";
        const newsList = node.querySelector(".newslist");
        for (const candidate of screen.candidates || []) {
          const headline = candidate.top_events?.[0]?.title || "暂无事件标题";
          newsList.insertAdjacentHTML("beforeend", `<div class=newsitem>` +
            `<span class="score ${tone(candidate.news_score)}">${signed(candidate.news_score)}</span>` +
            `<b>${esc(candidate.name)}</b><span class=muted> ${esc(candidate.symbol)} · ${candidate.article_count}条</span>` +
            `<div class=muted>${esc(headline)}</div></div>`);
        }
        if (!newsList.children.length) newsList.innerHTML = "<div class=muted>本次日志没有新闻候选结果</div>";
      } else {
        dtb.innerHTML = "<tr><td colspan=5 class=muted>暂无决策日志,先运行 run_daily.py</td></tr>";
        node.querySelector(".views").innerHTML = "<div class=muted>暂无分析</div>";
        node.querySelector(".newslist").innerHTML = "<div class=muted>暂无新闻候选</div>";
      }
      root.appendChild(node);
      const curve = d.curve;
      const canvas = root.querySelectorAll(".curve")[root.querySelectorAll(".curve").length-1];
      if (curve.length) {
        marketCharts[mkt] = new Chart(canvas, {
          data: { labels: curve.map(x => x.date), datasets: [
            { type:"line", label:"总资产", data:curve.map(x => x.total), borderColor:"#4f8ef7",
              backgroundColor:"rgba(79,142,247,.10)", fill:true, tension:.3, pointRadius:2, yAxisID:"assets" },
            { type:"bar", label:"区间盈亏", data:curve.map(x => x.period_change),
              backgroundColor:curve.map(x => x.period_change >= 0 ? "rgba(46,204,113,.45)" : "rgba(231,76,60,.45)"), yAxisID:"change" }
          ]},
          options: { interaction:{mode:"index",intersect:false}, plugins:{legend:{labels:{color:"#8b93a7"}}},
            scales:{x:{ticks:{color:"#8b93a7"},grid:{color:"#262b38"}},
              assets:{position:"left",ticks:{color:"#8b93a7"},grid:{color:"#262b38"}},
              change:{position:"right",ticks:{color:"#8b93a7"},grid:{display:false}}}}
        });
      }
    }
    document.getElementById("updated").textContent = "更新于 " + new Date().toLocaleTimeString();
  } catch (error) {
    root.innerHTML = `<div class="card full neg">加载失败: ${esc(error.message)}</div>`;
    document.getElementById("updated").textContent = "刷新失败";
  } finally {
    btn.disabled = false; btn.textContent = "刷新数据";
  }
}

async function loadOperations() {
  const box = document.getElementById("operations");
  try {
    const data = await (await fetch("/api/operations")).json();
    const labels = {complete:"完成", skipped:"已跳过重复运行", failed:"失败", running:"运行中",
                    interrupted:"已中断", awaiting_confirmation:"计划等待人工确认"};
    const latest = Object.entries(data.latest || {});
    box.innerHTML = latest.length ? latest.map(([market, run]) =>
      `<div><b>${market === "cn" ? "A股" : "美股"}</b> · ${esc(labels[run.status] || run.status)} · ${esc(run.finished_at || run.started_at)}` +
      (run.error ? `<div class="warning">${esc(run.error)}</div>` : "") + "</div>").join("") : "还没有运行记录";
  } catch (error) { box.textContent = "运行状态暂时无法读取"; }
}

async function loadScore() {
  const btn = document.getElementById("scoreBtn");
  btn.disabled = true; btn.textContent = "计算中…";
  const s = await (await fetch("/api/score")).json();
  let html = "";
  for (const mkt of ["cn", "us"]) {
    const st = s.stats[mkt];
    html += `<div style="margin-bottom:6px"><b>${mkt === "cn" ? "A股" : "美股"}</b>:` +
      (st.total ? `已到期 ${st.total} 条,命中 ${st.hit},命中率 <b>${(st.hit/st.total*100).toFixed(1)}%</b>`
                : "暂无已到期预测") +
      (st.pending ? ` <span class=muted>(未到期 ${st.pending})</span>` : "") + "</div>";
  }
  if (s.rows.length) {
    html += "<table><thead><tr><th>日期</th><th>市场</th><th>代码</th><th>预测</th><th>实际</th><th>命中</th></tr></thead><tbody>";
    for (const r of s.rows.reverse()) {
      html += `<tr><td>${esc(r.date)}</td><td>${esc(r.market)}</td><td>${esc(r.symbol)}</td>` +
        `<td>${esc(r.pred)}</td><td>${esc(r.actual)}</td>` +
        `<td class=${r.hit ? "pos" : "neg"}>${r.hit ? "✓" : "✗"}</td></tr>`;
    }
    html += "</tbody></table>";
  }
  const cal = s.calibration || {};
  html += `<div style="margin-top:10px"><b>置信度校准</b>：${cal.status === "complete"
    ? `未来验证 ${cal.test_count} 条 · 原始 Brier ${metric(cal.raw.brier_score)} / ECE ${metric(cal.raw.expected_calibration_error)} · 校准后 Brier ${metric(cal.calibrated.brier_score)} / ECE ${metric(cal.calibrated.expected_calibration_error)}`
    : esc(cal.message || "样本不足")}</div>`;
  if (s.calibration_by_variant?.length > 1) {
    for (const entry of s.calibration_by_variant) {
      const c = entry.calibration;
      html += `<div class=muted>${esc(entry.market)} / ${esc(entry.variant.prompt_version || "unversioned")}：` +
        (c.status === "complete" ? `验证 ${c.test_count}条 · Brier ${metric(c.calibrated.brier_score)} / ECE ${metric(c.calibrated.expected_calibration_error)}` : esc(c.message)) + "</div>";
    }
  }
  document.getElementById("scoreBox").innerHTML = html;
  btn.disabled = false; btn.textContent = "重新计算";
}

const metric = (value, suffix="") => value == null ? "样本不足" : Number(value).toFixed(2) + suffix;
const perfCharts = {};
async function loadPerformance() {
  const btn = document.getElementById("perfBtn");
  btn.disabled = true; btn.textContent = "计算中…";
  const data = await (await fetch("/api/performance")).json();
  let html = "<table><thead><tr><th>市场</th><th>累计收益</th><th>时变候选基准</th><th>超额</th><th>年化波动</th><th>最大回撤</th><th>Sharpe</th><th>Sortino</th><th>换手</th></tr></thead><tbody>";
  for (const mkt of ["cn", "us"]) {
    const r = data[mkt], p = r.portfolio, b = r.benchmark;
    html += `<tr><td>${mkt === "cn" ? "A股" : "美股"}<div class=muted>${esc(r.data_quality.message)}</div></td>` +
      `<td>${metric(p.total_return_pct,"%")}</td><td>${metric(b.total_return_pct,"%")}</td>` +
      `<td>${metric(r.excess_return_pct,"%")}</td><td>${metric(p.annualized_volatility_pct,"%")}</td>` +
      `<td>${metric(p.max_drawdown_pct,"%")}</td><td>${metric(p.sharpe_ratio)}</td>` +
      `<td>${metric(p.sortino_ratio)}</td><td>${metric(r.turnover_pct,"%")}</td></tr>`;
  }
  html += "</tbody></table>";
  document.getElementById("perfBox").innerHTML = html;
  Object.values(perfCharts).forEach(chart => chart.destroy());
  const labels = [...new Set(["cn", "us"].flatMap(mkt =>
    (data[mkt].portfolio_curve || []).map(point => point.date)))].sort();
  const values = (curve, field) => {
    const byDate = Object.fromEntries((curve || []).map(point => [point.date, point[field]]));
    return labels.map(day => byDate[day] ?? null);
  };
  perfCharts.returns = new Chart(document.getElementById("perfCurve"), {
    type:"line", data:{labels, datasets:[
      {label:"A股组合",data:values(data.cn.portfolio_curve,"index"),borderColor:"#4f8ef7",tension:.25},
      {label:"A股时变候选",data:values(data.cn.benchmark_curve,"index"),borderColor:"#89b4fa",borderDash:[5,4],tension:.25},
      {label:"美股组合",data:values(data.us.portfolio_curve,"index"),borderColor:"#2ecc71",tension:.25},
      {label:"美股时变候选",data:values(data.us.benchmark_curve,"index"),borderColor:"#95e6b7",borderDash:[5,4],tension:.25}
    ]}, options:{plugins:{legend:{labels:{color:"#8b93a7"}}},scales:{x:{ticks:{color:"#8b93a7"},grid:{color:"#262b38"}},y:{ticks:{color:"#8b93a7"},grid:{color:"#262b38"}}}}
  });
  perfCharts.drawdown = new Chart(document.getElementById("drawdownCurve"), {
    type:"line", data:{labels,datasets:[
      {label:"A股回撤",data:values(data.cn.portfolio_curve,"drawdown_pct"),borderColor:"#e74c3c",backgroundColor:"rgba(231,76,60,.12)",fill:true,tension:.25},
      {label:"美股回撤",data:values(data.us.portfolio_curve,"drawdown_pct"),borderColor:"#f39c75",backgroundColor:"rgba(243,156,117,.08)",fill:true,tension:.25}
    ]}, options:{plugins:{legend:{labels:{color:"#8b93a7"}}},scales:{x:{ticks:{color:"#8b93a7"},grid:{color:"#262b38"}},y:{ticks:{color:"#8b93a7",callback:value=>value+"%"},grid:{color:"#262b38"}}}}
  });
  btn.disabled = false; btn.textContent = "重新计算";
}

async function loadExperiments() {
  const btn = document.getElementById("expBtn");
  btn.disabled = true; btn.textContent = "加载中…";
  try {
    const data = await (await fetch("/api/experiments")).json();
    const news = data.news_dataset || {event_count:0,markets:{}};
    let html = `<div><b>历史新闻库</b>：${news.event_count || 0} 条真实/导入事件`;
    for (const [market, qualities] of Object.entries(news.markets || {})) {
      const detail = Object.entries(qualities).map(([quality, item]) =>
        `${quality} ${item.count}条 (${item.first_day}~${item.last_day})`).join("；");
      html += `<div class=muted>${esc(market)}：${esc(detail)}</div>`;
    }
    html += "</div>";
    if (data.research?.length) {
      const verdicts = {registered:"已注册", awaiting_data:"等待数据", awaiting_review:"等待复盘或核验",
        insufficient_data:"样本不足", unverified_engineering:"工程验收，效果未验证", reject:"未达到标准",
        not_comparable:"输入不可比", retain_for_forward_validation:"保留，需向前验证", retain_for_paper_validation:"保留，继续模拟"};
      html += "<div style='margin-top:12px'><b>A股研究闭环</b></div>";
      for (const study of data.research) {
        html += `<div class=newsitem><b>${esc(study.name)}</b> · ${esc(verdicts[study.verdict] || study.verdict)}` +
          `<div>${esc(study.hypothesis)}</div><div class=muted>已记录 ${study.attempt_count} 次运行 · ${esc(study.id)}</div>` +
          (study.next_hypothesis ? `<div>下一步：${esc(study.next_hypothesis.recommended_action)}</div>` : "") + "</div>";
      }
    }
    html += `<div style="margin-top:6px"><b>严格可比 A/B 组</b>：${data.comparisons?.length || 0} 组` +
      `<span class=muted>（数据指纹相同且变体不同）</span></div>`;
    if (data.runs?.length) {
      html += "<table style='margin-top:10px'><thead><tr><th>时间</th><th>实验</th><th>市场</th><th>Prompt / 模型</th><th>数据指纹</th><th>状态</th></tr></thead><tbody>";
      for (const run of data.runs) {
        const variant = run.variant || {};
        const roles = Object.values(variant.role_models || {}).map(x => x.model).filter(Boolean).join(" / ");
        html += `<tr><td>${esc(run.created_at)}</td><td>${esc(run.name)}</td><td>${esc(run.market)}</td>` +
          `<td>${esc(variant.prompt_version || "-")}<div class=muted>${esc(roles || variant.mode || "-")}</div></td>` +
          `<td>${esc(run.dataset_fingerprint.slice(0,12))}</td><td>${esc(run.status)}</td></tr>`;
      }
      html += "</tbody></table>";
    } else {
      html += "<div class=muted style='margin-top:8px'>尚无已登记实验；真实新闻积累后运行实验 CLI。</div>";
    }
    document.getElementById("experimentBox").innerHTML = html;
  } catch (error) {
    document.getElementById("experimentBox").textContent = "实验数据加载失败：" + error.message;
  } finally {
    btn.disabled = false; btn.textContent = "刷新实验";
  }
}

load();
loadExperiments();
setInterval(load, 60000);
</script>
</body>
</html>"""


@app.route("/")
def index():
    return PAGE


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
