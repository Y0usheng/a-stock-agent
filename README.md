# 新闻驱动长期投资模拟 Agent（当前重点：A股）

通过真实可跑的项目研究：**行业新闻 → 长期机会假设 → 公司证据与估值 → 多角色风控 → 模拟持仓 → 真实价格与净收益评价**。默认复用ChatGPT订阅；保留SDK/框架及旧版实验对照。

**免责声明:纯模拟盘学习项目,不构成投资建议。**

## 当前状态

- 最新默认流程为[新闻发现长期机会](docs/NEWS_OPPORTUNITIES.md)：全局快讯先形成待核验主题，再从12家受控公司中选择新增研究对象；已有持仓保留。关键词不代表看好，主题推断不能替代公司披露和估值。
- `/trading` 展示行业主题、原文片段、入选/未选理由、入选后的真实价格观察以及实际模拟净收益。页面只读；未成交观察不计账户利润。数据仅部分覆盖市场，尚未证明投资效果。
- 本机运行数据和原始日志不上传仓库；文档中的本地日志路径用于本机核验。

- **实验平台第三阶段已完成**:历史新闻 SQLite、Prompt/模型版本追踪、可复现实验注册和过去训练/未来验证的置信度校准已接通。
- 四个角色统一使用 `gpt-6.1-sol`，默认通过官方 `codex exec` 复用 ChatGPT 订阅登录，无需 API 密钥；API 对照需显式选择 `--model-access api`。
- 已接通 LangGraph 并行分析、独立风控、SQLite 检查点及计划哈希绑定的人工确认；支持同截面 SDK/图框架对照。
- 已补齐宿主逐笔风控审批、只分析模式、版本实验、历史价格导入/抓取、运行锁和失败记录。
- 已完成新闻 MCP、动态候选池、多角色只读分析和双市场离线端到端验收。
- 当前没有真实资金或券商接口；所有买卖仅写入本地 SQLite 模拟账本。
- 最小请求与完整四角色真实订阅验收均已完成（合成单股票输入、零订单）：4次请求、26211 tokens、约181秒；124项测试通过。两市场真实行情/新闻分析仍待验收。
- 2026-10-03 已用12只 A股真实日线跑通60日回测，数据为2026-04-09至2026-09-30；新闻仍是合成事件。美股在线历史源本次限流，未降级。
- 详细完成项、验收证据和未完成项见 [项目状态](docs/PROJECT_STATUS.md)。
- 2026-10-27 确认下一阶段方向：沿用角色分工，补齐逐条结论的证据追溯和研究实验闭环；具体待办及验收标准见 [下一阶段计划](docs/NEXT_PHASE.md)。
- 订阅接入、登录检查、系统代理、超时终止及运行方式见 [模型接入说明](docs/MODEL_ACCESS.md)。
- 默认订阅流程已接入逐条证据引用、内容/时间校验与新闻事件回查。来源选择和当前边界见 [证据追溯说明](docs/EVIDENCE_TRACEABILITY.md)；不将摘要引用视作已经核对原文。
- A股离线语义核验器已实现：规则预检、事实/观点/推断区分、抽样订阅复核、原句定位和缓存。独立于日常四角色运行，使用方式见 [语义核验说明](docs/SEMANTIC_REVIEW_PLAN.md)。
- A股实验闭环已实现：预先登记假设/对照/成功条件，冻结输入，训练/验证/测试，同成本重放，留存失败和复盘，提出下一轮建议。177项测试通过；真实历史价格候选按标准否决，无同期新闻的变体保留无数据。见 [实验闭环说明](docs/RESEARCH_LOOP.md)。
- 2026-10-28 首次单股票真实资料四角色shadow完成：300750，4次订阅请求、31291 tokens、约5分08秒、零订单。输出保留观望与5日预测；财务时点缺口和2处专家角色越界仍需修正，见[项目状态](docs/PROJECT_STATUS.md)。本地日报为 `logs/live/2026-10-28.cn/daily-brief.md`，尚非自动输出入口。
- 已增加[模拟交易与股价收益展示](docs/TRADING_VIEW.md)：真实180日价格、均线/成交量、实际成交标记、持仓浮盈亏和扣费净收益；本地入口为 `/trading`。今天空仓仅显示真实0%观察点，不将过去股票涨跌当作Agent收益。
- 2026-10-29 A股默认采用[长期投资协议](docs/LONG_TERM_INVESTING.md)：三年论证、252日阶段观察、季度复核；新闻发现候选并监控后续变化。区分待补核心资料/等待价格/建仓/持有/退出，持久保存底稿和有期限的补证任务；真实披露片段仍待补齐。

## 架构

```
┌──────────────────────────────────────────────────────────────┐
│ 投资经理 → 技术+基本面 → 风控（四个角色均为 GPT 6.1 Sol）          │
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

# 只分析，不提交模拟订单；无需模型 API key
.venv\Scripts\python agent/run_daily.py --mock --offline --shadow

# 先用官方客户端登录 ChatGPT 订阅账号；凭据由客户端保管
codex login
# 仅检查本机登录方式、模型配置和依赖，不调用模型
.venv\Scripts\python agent/run_daily.py --preflight

# 最小真实订阅检查：只发送固定测试文字，消耗少量订阅额度
.venv\Scripts\python scripts/check_subscription.py --smoke

# 默认订阅 + LangGraph，先只分析；行情或新闻不可用时会停止
.venv\Scripts\python agent/run_daily.py --shadow                # 默认A股长期模式
.venv\Scripts\python agent/run_daily.py --shadow --market cn    # 只跑A股
.venv\Scripts\python agent/run_daily.py --shadow --market us    # 只跑美股

# LangGraph 只分析，技术面/基本面并行，随后独立风控
.venv\Scripts\python agent/run_daily.py --framework langgraph --shadow --market cn

# LangGraph 模拟盘：有交易时先保存待确认计划，不立即提交订单
.venv\Scripts\python agent/run_daily.py --framework langgraph --market cn
# 查看 logs/decisions/pending/ 中的计划后，用输出的完整哈希确认或拒绝
.venv\Scripts\python agent/run_daily.py --framework langgraph --market cn --approve-plan PLAN_HASH
.venv\Scripts\python agent/run_daily.py --framework langgraph --market cn --reject-plan PLAN_HASH
# 保持原 Prompt、模型和 offline 设置；行情/账户/新闻/基本面/记忆变化则拒绝恢复
# 过期计划可用 --force 重新分析，再审批新哈希；--force 不能与审批参数同时使用

# Prompt 版本显式选择；版本和完整内容哈希会写入日志
.venv\Scripts\python agent/run_daily.py --prompt-version manager-v2 --market us

# 同一只读截面比较两个 Prompt，自动进入 shadow 模式（消耗订阅额度）
.venv\Scripts\python agent/run_daily.py --market cn --compare-prompts manager-v1 manager-v2

# API 对照需在本机配置 OPENAI_API_KEY；两个框架都只分析，按 API 用量计费
.venv\Scripts\python agent/run_daily.py --model-access api --market cn --compare-frameworks sdk langgraph

# 几个交易日后,给预测打分(分市场输出命中率)
.venv\Scripts\python evals/score.py

# 按数据指纹、Prompt 和各角色模型分组登记实验；此命令不调用模型
.venv\Scripts\python evals/score.py --record-experiment

# 单独评分只分析日志，按变体校准、登记实验；不会混入模拟账户收益
.venv\Scripts\python evals/score.py --shadow --record-experiment

# 组合收益、无未来泄漏时变候选基准、波动、最大回撤、Sharpe、Sortino、换手率
.venv\Scripts\python evals/report.py

# 60–120交易日滚动回测 + 新闻开/关 A/B（默认90日离线可复现）
.venv\Scripts\python evals/backtest.py --market both --window 90

# 真实日线：导出指定截止日的数据；默认新闻仍为合成事件
.venv\Scripts\python evals/backtest.py --market cn --price-source live --window 60 --end-date 2026-09-30 --dynamic-candidates --export-prices data/historical/cn.json

# 重放冻结的 JSON，或导入 CSV；不需要网络
.venv\Scripts\python evals/backtest.py --market cn --price-source file --prices-file data/historical/cn.json --window 60 --dynamic-candidates
# CSV 列：symbol,date,open,close,volume，可加 high,low；volume 单位必须是股
# --market both 时 --prices-file 可指定包含 cn.csv/us.csv 的目录

# 查看历史新闻覆盖；导入供应商导出的 JSON/JSONL
.venv\Scripts\python evals/news_store.py --summary
.venv\Scripts\python evals/news_store.py --import-file history.jsonl --market us

# 使用真实/导入历史新闻回测；窗口无数据时会拒绝降级成合成新闻
.venv\Scripts\python evals/backtest.py --market us --price-source live --window 90 --historical-news --dynamic-candidates --record-experiment

# Web 仪表盘(推荐):浏览器查看资产曲线/持仓/决策/命中率
.venv\Scripts\python dashboard/app.py
# 打开 http://127.0.0.1:5000

# A股实验闭环验收：不调用模型、不写订单，生成计划、复盘和重放回执
.venv\Scripts\python scripts/research.py demo
.venv\Scripts\python scripts/research.py demo --prices data/historical/cn-2026-10-02.json
.venv\Scripts\python scripts/research.py list
```

- 实时模式采用 fail-closed：行情或新闻源不可验证时停止该市场任务，绝不会静默改用模拟数据
- `--mock` 表示使用多角色规则替身且不调用模型；只有显式 `--offline` 才会生成可复现模拟行情和新闻事件
- 交易使用行情截面日而非电脑当天日期；账户会校验本轮 `run_id` 和每日批准股票名单
- 同一行情日默认只运行一次；`--force` 仅在该轮没有成交时允许覆盖分析日志，账户已有成交时仍会安全跳过
- 每笔订单都有唯一幂等键，重复请求返回原成交结果，不会再次扣款或增减持仓
- 风控专家必须输出逐只结构化审批；宿主直接捕获送审计划和专家结果，执行前整批核对方向与批准股数，经理无法自行填入审批绕过检查
- 决策、预测和审批均拒绝重复股票；LangGraph 两位分析师汇合后才进入计划和风控。订阅传输按角色串行发送，每个角色一次请求，共用调用预算
- `--shadow` 不调用买卖、运行授权或估值写入工具，使用独立 `logs/decisions/shadow/` 日志；首次账户读取仍可能初始化空模拟账本
- `--compare-prompts` 在一次命令内共用行情、基本面、新闻、账户、记忆及工具读取缓存，并按变体分别留存日志；相同变体同日默认跳过，`--force` 可重做只读分析
- 买卖价加入保守单边滑点，并限制单笔成交量参与率；成交量无效/停牌时拒绝订单
- 每日总资产、现金、持仓、估值价和数据源写入 SQLite `valuation_snapshots`，JSON 日志不再是唯一资产历史
- 新闻按市场时区限制在截面日收盘前；仅有日期的事件延迟到下一日使用
- 实时日线剔除未收盘的当天行情；候选池日期必须一致、价格/成交量有效，超出允许陈旧天数时停止
- A股东财日线成交量从手转换为股，两个市场使用统一流动性单位
- 同一市场使用进程锁，运行状态保存在 `data/operations.db`；一个市场失败仍继续另一个，最终退出码为失败；不自动重试订单
- 部分成交而日志缺失时保留账本并停止重复下单，需要人工核对恢复；进程中断后锁自动释放
- `manager-v1` 保留第二阶段基线，`manager-v2` 加入证据分级与概率式 confidence；切换版本才会改变行为
- A/B 只比较 `dataset_fingerprint` 相同的实验；不同数据窗口的结果不会被伪装成模型优劣
- 账户:`data/account_cn.db` / `data/account_us.db`(删除即重置资金)
- 日志:`logs/decisions/YYYY-MM-DD.{cn,us}.json`;`logs/demo/` 为沙盒演示结果

## 为什么加入三个能力

### Agent SDK 多角色分析

把机会发现与风险批准拆开。四个角色共用 GPT 6.1 Sol，但使用独立上下文、指令和只读工具。
风控审批由宿主捕获和校验，而非仅靠经理自述。最终计划通过结构校验后由宿主按行情快照价统一执行，所有
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

每次 Agent SDK 运行会记录 Prompt 版本、完整 Prompt SHA-256、稳定模板 SHA-256 和四个角色的 provider/model。
完整哈希包含每日行情日和候选池，仅用于审计；跨日实验分组使用包含专家指令/权限的稳定模板哈希。
旧日志缺少模板哈希时按原版本分组，并与新模板日志隔离。shadow 对照记录冻结输入和工具数据指纹，
评分登记同时绑定到期价格指纹；数据不同的变体不会组成严格 A/B。校准按市场和变体分别进行。
`evals/score.py --record-experiment` 将成熟预测按这些变体登记到 `data/experiments.db`；只有相同数据指纹才可做严格 A/B。
置信度报告同时输出 Brier Score 与 ECE，并严格用较早日期拟合、较晚日期验证；样本少于8条或不足两个
日期时只显示“样本不足”。这些本地评分命令不调用模型，不产生模型 API 费用。

## 测试

```bash
.venv\Scripts\python -m unittest discover -s tests -v
```

测试覆盖统一模型配置、Responses 请求与预算、多角色权限隔离、Agent SDK 只读工具、Prompt 版本、记忆截止日、防未来泄漏、收益风险公式、时变基准、
决策与账本成交核对、宿主锁定成交价、每日幂等、离线行情可复现、预测到期判定、数据源隔离、
订单校验、仓位上限、A 股 T+1、滑点、流动性、持久化估值、次日开盘回测、新闻去重/时间上界、
实验指纹、Brier/ECE 与新闻 A/B，以及审批拒绝/缩量、SDK 专家出口捕获、跨日版本分组、只读变体留存、原子日志写入、图并行/恢复、过期审批、进程锁与失败隔离。

## 可选环境变量

| 变量 | 用途 |
|---|---|
| `ASTOCK_MODEL_ACCESS` | 默认 `subscription`；显式设为 `api` 才启用密钥路径 |
| `OPENAI_API_KEY` | 仅显式 API 模式使用，订阅子进程不会接收该变量 |
| `ASTOCK_CODEX_BIN` | 可选官方 Codex 可执行文件路径，默认从 PATH 查找 |
| `ASTOCK_SUBSCRIPTION_TIMEOUT` | 单次订阅推理最长等待秒数，默认120；超时终止本次进程树 |
| `ASTOCK_SUBSCRIPTION_MAX_REQUESTS` | 默认4，跨角色共用，失败请求也计入次数 |
| `ASTOCK_SUBSCRIPTION_MAX_TOKENS` | 默认40000，完成请求后核对并阻止后续调用；不是在途硬限额 |
| `ASTOCK_SUBSCRIPTION_MAX_INPUT_CHARS` | 单角色输入字符上限，默认24000，超限会停止 |
| `ASTOCK_MANAGER_PROVIDER` / `ASTOCK_RISK_PROVIDER` | 兼容原参数名，现在仅支持 `openai` |
| `ASTOCK_OPENAI_MODEL` | 四角色统一模型，默认 `gpt-6.1-sol` |
| `ASTOCK_REASONING_EFFORT` | 订阅默认 `low`、API 默认 `medium`，支持 `low/medium/high/xhigh/max` |
| `ASTOCK_MAX_OUTPUT_TOKENS` | 仅 API：单次请求输出上限，默认8192，含推理 token |
| `ASTOCK_MAX_MODEL_REQUESTS` | 单市场/单变体跨四角色请求上限，默认40 |
| `ASTOCK_MAX_TOTAL_TOKENS` | 已完成调用累计 token 达160000后阻止后续调用；在途请求仍可能超出 |
| `ASTOCK_ANALYSIS_TIMEOUT` | 完整模型分析超时秒数，默认600；单 HTTP 请求90秒且不自动重试 |
| `ASTOCK_MAX_QUOTE_AGE_DAYS` | 行情最大陈旧自然日数，默认10；不等同交易日历校验 |
| `ASTOCK_OPENAI_TRACING=1` | 显式启用 OpenAI SDK tracing;默认关闭且不上传敏感 trace |
| `ASTOCK_MAX_TURNS` | 单市场 Agent SDK 最大轮次,默认 `24` |
| `ASTOCK_PROMPT_VERSION` | 投资经理 Prompt 版本:`manager-v1`(默认)或`manager-v2` |
| `ASTOCK_OFFLINE=1` | 强制使用可复现离线行情,等价于 `--offline` |
| `ALPHAVANTAGE_API_KEY` | 可选；增强美股新闻情绪与按时间范围检索，未配置时使用 yfinance/Yahoo |
| `ASTOCK_DATA_DIR` | 覆盖 SQLite 账本目录,适合隔离测试 |
| `ASTOCK_NEWS_DB` | 覆盖历史新闻 SQLite 路径 |
| `ASTOCK_EXPERIMENT_DB` | 覆盖实验注册表 SQLite 路径 |
| `ASTOCK_LOG_DIR` | 覆盖决策日志目录,适合隔离测试 |
| `ASTOCK_OPS_DB` | 覆盖运行状态 SQLite 路径 |

配置清单见 `.env.example`；本项目不会自动加载 `.env`。调用次数、token 和超时限制用于控制开销，不能当作精确美元硬预算。

## 每日自动运行

已提供 `scripts/run_scheduled.ps1`，默认只分析，完整输出保存在 `data/run_logs/`，失败返回非零退出码。
Windows 任务计划程序可以使用以下动作，需先在运行任务的用户环境配置密钥：

- A股15:30：`pwsh.exe -File E:\a-stock-agent\scripts\run_scheduled.ps1 -Market cn`
- 美股北京时间05:30：`pwsh.exe -File E:\a-stock-agent\scripts\run_scheduled.ps1 -Market us`

加 `-Paper` 才提交模拟订单；LangGraph 模拟盘仍需人工确认。可用 `-Mock -Offline` 验收入口。
本次没有注册常驻任务或外部通知；查看仪表盘的“运行状态”可发现失败和待确认计划。

## 学习路线图

- [x] **阶段1 MCP**:数据/账户 MCP server + MCP client 决策循环,双市场参数化
- [x] **阶段1.1 稳定化**:可复现行情、交易日对齐、账户风控、可靠评分、自动化测试
- [x] **阶段2 OpenAI Agents SDK**:四个 GPT 6.1 Sol 角色、只读 agents-as-tools、宿主执行隔离
- [x] **阶段3 上下文工程 + Evals 基线**:选择性历史记忆、收益风险和等权基准
- [x] **阶段3.1 新闻动态候选池**:新闻 MCP、受控候选宇宙、新闻/技术共振与防未来泄漏
- [x] **阶段3.2 可信评估**:时变候选基准、每日估值快照、成交成本、60–120日新闻A/B
- [x] **阶段3.3 实验平台**:历史新闻采集/导入、Prompt/模型版本记录、置信度校准
- [x] **阶段3.4 安全分析与版本对照**:宿主绑定风控审批、只分析模式、稳定模板分组和同截面多版本日志
- [x] **阶段4 LangGraph 工程实现**:并行分析→计划→风控分支→持久化人工确认，与 SDK 同截面对照
- [x] **运行保障与历史价格入口**:配置检查、进程锁、失败隔离、状态面板、真实/文件行情和重放指纹
- [ ] **真实模型验收**:配置密钥，双市场 shadow 实测 GPT 6.1 Sol
- [ ] **策略有效性验收**:真实历史新闻、长期模拟样本、模型/Prompt/编排框架的到期预测比较

## 关键学习点(读代码时注意)

- `servers/*.py`:FastMCP 装饰器定义工具;docstring 会喂给 LLM,写法影响 agent 用得好不好
- 两个市场数据 server 接口一致 → 决策循环无需改动即可跨市场复用(接口标准化的意义)
- `account_server.py`:工具出错返回结构化 `{"ok": False, "error": ...}` 而非抛异常,agent 才能读懂并自我纠正
- `agent/sdk_runner.py` 的 `_system_prompt`:市场规则动态生成进 prompt;要求结构化 JSON 输出 + 强制预测,是 evals 的数据基础
- `evals/score.py`:预测 vs 实际的评分闭环——改 prompt 后分数变化就是迭代依据
- `agent/sdk_runner.py`:统一模型路由、agents-as-tools、只读工具白名单和本地 JSON 校验
- `agent/graph_runner.py`:图节点并行、风控分支、SQLite 检查点与人工确认
- `agent/runtime.py` / `agent/operations.py`:跨角色预算、配置检查、运行锁、错误脱敏和失败隔离
- `agent/memory.py`:只选截止日前的紧凑历史,每次注入均可用 `memory_id` 追溯
- `evals/performance.py`:组合与无未来泄漏时变基准的风险调整收益闭环
- `evals/backtest.py`:收盘信号/次日开盘成交的滚动回测与新闻 A/B
- `evals/price_data.py`:真实/文件/合成价格、单位校验、数据冻结与内容指纹验证
- `evals/news_store.py`:真实历史新闻的不可变落库、时间切片、导入和数据指纹
- `agent/prompt_registry.py` / `evals/experiments.py`:Prompt 版本、角色模型与同数据集实验注册
- `evals/calibration.py`:Brier、ECE 和严格时间外验证校准
