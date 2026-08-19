# 双市场新闻动态选股模拟盘 Agent(A股 + 美股,学习项目)

通过一个真实可跑的项目学习:**MCP → Agent SDK 多角色决策 → 选择性记忆 → 收益风险 Evals**(阶段4将引入 LangGraph 对比)。

**免责声明:纯模拟盘学习项目,不构成投资建议。**

## 当前状态

- **实验平台第三阶段已完成**:历史新闻 SQLite、Prompt/模型版本追踪、可复现实验注册和过去训练/未来验证的置信度校准已接通。
- 已完成新闻 MCP、动态候选池、多角色只读分析和双市场离线端到端验收。
- 当前没有真实资金或券商接口；所有买卖仅写入本地 SQLite 模拟账本。
- 未使用有效 API key 执行收费的 Agent SDK 推理；该路径已完成配置、导入、权限和结构测试。

## 架构

```
┌──────────────────────────────────────────────────────────────┐
│ 投资经理(GPT/DeepSeek) → 技术+基本面(DeepSeek) → 风控(GPT/DeepSeek) │
│        ↑ 选择性历史记忆           ↓ 所有模型角色都只读             │
└──────────────────────────┬───────────────────────────────────┘
                           │ MCP
    ┌──────────────────────┼──────────────────────┐
    │A股行情/基本面         │模拟账户               │美股行情/基本面
┌───▼────┐ ┌───▼────┐ ┌───▼────┐ ┌───▼────┐
│market  │ │account │ │us_market│ │account │
│server  │ │server  │ │server   │ │server  │
│akshare │ │CNY账本  │ │yfinance │ │USD账本 │
│        │ │T+1/整手 │ │         │ │T+0/1股 │
└────────┘ └────────┘ └─────────┘ └────────┘
  决策+角色结论+记忆ID → logs/decisions/日期.市场.json
                          ├→ evals/score.py 方向评分
                          ├→ evals/report.py 收益风险/时变候选基准
                          ├→ evals/backtest.py 60–120日滚动新闻A/B
                          └→ data/experiments.db 数据/版本/指标实验注册表
```

同一份 `account_server.py` 通过环境变量 `ASTOCK_MARKET` 加载不同市场规则;
两个数据 server 工具接口完全一致,决策循环零改动复用——这是 MCP 抽象的价值。

## 市场规则对比

|  | A股 (cn) | 美股 (us) |
|---|---|---|
| 初始资金 | ¥1,000,000 | $150,000 |
| 最小单位 | 一手100股 | 1股 |
| 回转交易 | T+1 | T+0 |
| 费用 | 佣金万2.5(最低5元)+卖出印花税万5 | 零佣金 |
| 模拟滑点 | 单边5 bps | 单边3 bps |
| 单笔流动性 | 不超过当日成交量5% | 不超过当日成交量2% |
| 行情源 | akshare | yfinance |
| 新闻源 | 巨潮资讯公告 + 东方财富新闻 | yfinance/Yahoo；可选 Alpha Vantage |

## 安装(Windows,需 Python 3.10+;推荐 3.12)

```bash
cd E:\a-stock-agent
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

项目要求 `mcp>=1.19,<2.0`;MCP 2.0 已移除当前代码使用的 FastMCP 1.x 导入接口。

## 运行

```bash
# 规则策略(无需 API key,先跑通)
.venv\Scripts\python agent/run_daily.py --mock

# 完全离线且可复现(推荐用于首次验收和回归测试)
.venv\Scripts\python agent/run_daily.py --mock --offline

# OpenAI Agents SDK 混合 Provider 多角色决策(默认)
set OPENAI_API_KEY=sk-...
set DEEPSEEK_API_KEY=sk-...
.venv\Scripts\python agent/run_daily.py                # 双市场
.venv\Scripts\python agent/run_daily.py --market cn    # 只跑A股
.venv\Scripts\python agent/run_daily.py --market us    # 只跑美股

# 投资经理和风险经理可独立切换 Provider
.venv\Scripts\python agent/run_daily.py --manager-provider deepseek --risk-provider openai --market cn

# Prompt 版本显式选择；版本和完整内容哈希会写入日志
.venv\Scripts\python agent/run_daily.py --prompt-version manager-v2 --market us

# 几个交易日后,给预测打分(分市场输出命中率)
.venv\Scripts\python evals/score.py

# 按数据指纹、Prompt 和各角色模型分组登记实验；此命令不调用模型
.venv\Scripts\python evals/score.py --record-experiment

# 组合收益、无未来泄漏时变候选基准、波动、最大回撤、Sharpe、Sortino、换手率
.venv\Scripts\python evals/report.py

# 60–120交易日滚动回测 + 新闻开/关 A/B（默认90日离线可复现）
.venv\Scripts\python evals/backtest.py --market both --window 90

# 查看历史新闻覆盖；导入供应商导出的 JSON/JSONL
.venv\Scripts\python evals/news_store.py --summary
.venv\Scripts\python evals/news_store.py --import-file history.jsonl --market us

# 使用真实/导入历史新闻回测；窗口无数据时会拒绝降级成合成新闻
.venv\Scripts\python evals/backtest.py --market us --window 90 --historical-news --record-experiment

# Web 仪表盘(推荐):浏览器查看资产曲线/持仓/决策/命中率
.venv\Scripts\python dashboard/app.py
# 打开 http://127.0.0.1:5000
```

- 实时模式采用 fail-closed：行情或新闻源不可验证时停止该市场任务，绝不会静默改用模拟数据
- `--mock` 表示使用多角色规则替身且不调用模型；只有显式 `--offline` 才会生成可复现模拟行情和新闻事件
- 交易使用行情截面日而非电脑当天日期；账户会校验本轮 `run_id` 和每日批准股票名单
- 同一行情日默认只运行一次；`--force` 仅在该轮没有成交时允许覆盖分析日志，账户已有成交时仍会安全跳过
- 每笔订单都有唯一幂等键，重复请求返回原成交结果，不会再次扣款或增减持仓
- 买卖价加入保守单边滑点，并限制单笔成交量参与率；成交量无效/停牌时拒绝订单
- 每日总资产、现金、持仓、估值价和数据源写入 SQLite `valuation_snapshots`，JSON 日志不再是唯一资产历史
- 实时新闻成功抓取后幂等写入 `data/news_events.db`；历史回放的 SQL 上界严格限制在行情截面日
- `manager-v1` 保留第二阶段基线，`manager-v2` 加入证据分级与概率式 confidence；切换版本才会改变行为
- A/B 只比较 `dataset_fingerprint` 相同的实验；不同数据窗口的结果不会被伪装成模型优劣
- 账户:`data/account_cn.db` / `data/account_us.db`(删除即重置资金)
- 日志:`logs/decisions/YYYY-MM-DD.{cn,us}.json`;`logs/demo/` 为沙盒演示结果

## 为什么加入三个能力

### Agent SDK 多角色分析

把机会发现与风险批准拆开。技术分析师和基本面分析师固定使用 DeepSeek，风控经理可在 DeepSeek/GPT
之间切换，投资经理可在 GPT/DeepSeek 之间切换并最终综合。角色使用独立上下文和只读工具，降低单一
模型自己提出观点又自己批准的确认偏误。最终计划通过结构校验后由宿主按行情快照价统一执行，所有
模型角色都没有下单权限。

### 选择性历史记忆

每天只注入当前行情日之前最近5个同市场决策日、非 HOLD 行为、预测摘要和最多12笔成交，生成可审计
`memory_id`。全量日志会稀释当天信号、增加成本并诱发机械重复；选择性记忆让模型知道过去做过什么，
但明确旧理由只是证据而不是指令，并严格阻止未来数据泄漏。

### 收益风险 Evals

方向命中率不等于赚钱，更不等于风险可接受。新增累计收益、时变候选等权基准、超额收益、年化波动、
最大回撤、Sharpe、Sortino、交易次数和换手率。少于20个收益观察时不输出误导性的年化收益；
真实和模拟行情严格同源评估。每个收益区间只使用区间起点已经入选的候选股，未来新闻才入选的股票
不会被提前放进历史基准。

### 滚动回测与新闻 A/B

`evals/backtest.py` 支持60–120交易日窗口，信号只读取当日及以前数据，统一在下一交易日开盘成交，
并计入滑点、佣金、印花税和成交量上限。同一份行情分别开启/关闭新闻门控，输出收益差、回撤和成交数。
默认新闻是明确标记的可复现合成事件，只能验证工程链路和相对策略，不能当作真实历史新闻结论。

### 新闻 MCP 与动态候选池

每个市场维护12只受控候选宇宙，新闻 MCP 每日按公告/新闻事件强度选出8只，再自动并入已有持仓。
新闻只决定“分析谁”和风险门控，不直接产生订单；技术、基本面、投资经理、风险经理、宿主契约和账户
规则仍须全部通过。所有新闻保留来源与时间戳，严格过滤行情截面日之后的内容，并把正文标记为不可信
证据以降低提示词注入和未来数据泄漏风险。A股默认使用巨潮资讯公告与东方财富新闻；美股默认使用
yfinance/Yahoo，配置 `ALPHAVANTAGE_API_KEY` 后叠加 Alpha Vantage NEWS_SENTIMENT。

### 实验平台、版本 A/B 与置信度校准

实时新闻在抓取时增量落入 `data/news_events.db`，也可导入合法获得的供应商 JSON/JSONL 历史导出。
事件以内容哈希去重，同一 `event_id` 被改写时拒绝写入；回测窗口没有真实事件时会直接报错，不会
偷偷使用合成新闻。项目不会凭空附带商业历史新闻语料，因此刚升级时新闻库为 0 条是正常状态。

每次 Agent SDK 运行会记录 Prompt 版本、完整 Prompt SHA-256 和四个角色的 provider/model。
`evals/score.py --record-experiment` 将成熟预测按这些变体登记到 `data/experiments.db`；只有相同数据指纹才可做严格 A/B。
置信度报告同时输出 Brier Score 与 ECE，并严格用较早日期拟合、较晚日期验证；样本少于8条或不足两个
日期时只显示“样本不足”。这些本地评分命令不调用 GPT/DeepSeek，不产生模型 API 费用。

## 测试

```bash
.venv\Scripts\python -m unittest discover -s tests -v
```

70项测试覆盖角色级 Provider 路由、多角色权限隔离、Agent SDK 只读工具、Prompt 版本、记忆截止日、防未来泄漏、收益风险公式、时变基准、
决策与账本成交核对、宿主锁定成交价、每日幂等、离线行情可复现、预测到期判定、数据源隔离、
订单校验、仓位上限、A 股 T+1、滑点、流动性、持久化估值、次日开盘回测、新闻去重/时间上界、
实验指纹、Brier/ECE 与新闻 A/B。

## 可选环境变量

| 变量 | 用途 |
|---|---|
| `OPENAI_API_KEY` | 投资经理或风险经理使用 GPT 时所需 API key |
| `DEEPSEEK_API_KEY` | 技术、基本面及 DeepSeek 角色所需 API key |
| `ASTOCK_MANAGER_PROVIDER` | 投资经理 Provider:`openai`(默认)或 `deepseek` |
| `ASTOCK_RISK_PROVIDER` | 风险经理 Provider:`deepseek`(默认)或 `openai` |
| `ASTOCK_OPENAI_MODEL` | GPT 模型,默认 `gpt-5.6` |
| `ASTOCK_DEEPSEEK_MODEL` | DeepSeek 模型,默认 `deepseek-v4-flash` |
| `DEEPSEEK_BASE_URL` | DeepSeek OpenAI 兼容地址,默认 `https://api.deepseek.com` |
| `ASTOCK_OPENAI_TRACING=1` | 显式启用 OpenAI SDK tracing;默认关闭且不上传敏感 trace |
| `ASTOCK_MAX_TURNS` | 单市场 Agent SDK 最大轮次,默认 `24` |
| `ASTOCK_PROMPT_VERSION` | 投资经理 Prompt 版本:`manager-v1`(默认)或`manager-v2` |
| `ASTOCK_OFFLINE=1` | 强制使用可复现离线行情,等价于 `--offline` |
| `ALPHAVANTAGE_API_KEY` | 可选；增强美股新闻情绪与按时间范围检索，未配置时使用 yfinance/Yahoo |
| `ASTOCK_DATA_DIR` | 覆盖 SQLite 账本目录,适合隔离测试 |
| `ASTOCK_NEWS_DB` | 覆盖历史新闻 SQLite 路径 |
| `ASTOCK_EXPERIMENT_DB` | 覆盖实验注册表 SQLite 路径 |
| `ASTOCK_LOG_DIR` | 覆盖决策日志目录,适合隔离测试 |

## 每日自动运行

Windows 任务计划程序建两个任务:
- A股:每天 15:30 → `python E:\a-stock-agent\agent\run_daily.py --market cn`
- 美股:每天 05:30(美东收盘后)→ `python E:\a-stock-agent\agent\run_daily.py --market us`

## 学习路线图

- [x] **阶段1 MCP**:数据/账户 MCP server + MCP client 决策循环,双市场参数化
- [x] **阶段1.1 稳定化**:可复现行情、交易日对齐、账户风控、可靠评分、自动化测试
- [x] **阶段2 OpenAI Agents SDK 混合 Provider**:GPT/DeepSeek 角色路由、只读 agents-as-tools、宿主执行隔离
- [x] **阶段3 上下文工程 + Evals 基线**:选择性历史记忆、收益风险和等权基准
- [x] **阶段3.1 新闻动态候选池**:新闻 MCP、受控候选宇宙、新闻/技术共振与防未来泄漏
- [x] **阶段3.2 可信评估**:时变候选基准、每日估值快照、成交成本、60–120日新闻A/B
- [x] **阶段3.3 实验平台**:历史新闻采集/导入、Prompt/模型版本 A/B、置信度校准 ← 当前
- [ ] **阶段4 LangGraph**:图编排重写(并行分析→风控分支→人工确认),与 Agent SDK 对比

## 关键学习点(读代码时注意)

- `servers/*.py`:FastMCP 装饰器定义工具;docstring 会喂给 LLM,写法影响 agent 用得好不好
- 两个市场数据 server 接口一致 → 决策循环无需改动即可跨市场复用(接口标准化的意义)
- `account_server.py`:工具出错返回结构化 `{"ok": False, "error": ...}` 而非抛异常,agent 才能读懂并自我纠正
- `agent/sdk_runner.py` 的 `_system_prompt`:市场规则动态生成进 prompt;要求结构化 JSON 输出 + 强制预测,是 evals 的数据基础
- `evals/score.py`:预测 vs 实际的评分闭环——改 prompt 后分数变化就是迭代依据
- `agent/sdk_runner.py`:角色级 Provider 路由、agents-as-tools、只读工具白名单和本地 JSON 校验
- `agent/memory.py`:只选截止日前的紧凑历史,每次注入均可用 `memory_id` 追溯
- `evals/performance.py`:组合与无未来泄漏时变基准的风险调整收益闭环
- `evals/backtest.py`:收盘信号/次日开盘成交的滚动回测与新闻 A/B
- `evals/news_store.py`:真实历史新闻的不可变落库、时间切片、导入和数据指纹
- `agent/prompt_registry.py` / `evals/experiments.py`:Prompt 版本、角色模型与同数据集实验注册
- `evals/calibration.py`:Brier、ECE 和严格时间外验证校准
