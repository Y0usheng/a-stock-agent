# Changelog

## 2026-08-19 — Phase 1 可靠性加固

- 实时行情和新闻改为 fail-closed；只有显式 `ASTOCK_OFFLINE=1` / `--offline` 才允许模拟数据。
- 实时基本面失败时返回 `unavailable`，不再以离线参考值伪装成可交易证据。
- 账户成交日期改由宿主传入行情截面日，修复历史回放、T+1 与本机日期错位。
- 新增持久化 `run_authorizations` 每日批准名单，账户层拒绝名单外股票。
- 成交表新增 `run_id` / 唯一 `order_id`，重复订单幂等返回，不重复扣款或改变持仓。
- `--force` 检测到同轮已有成交时安全跳过并保留原日志，防止重跑改变账户状态。
- 自动迁移已有 SQLite 账户表；新增8项安全回归测试，48项测试全部通过。
- 双市场离线端到端连续运行并强制重跑后，A股/美股均保持1笔成交且无重复订单。

## 2026-08-13 — OpenAI Agents SDK + GPT/DeepSeek 角色路由

- 默认 Agent 编排从 Claude Agent SDK 迁移到官方 `openai-agents`。
- 投资经理支持 GPT/DeepSeek 切换；技术面和基本面固定 DeepSeek；风险经理支持 DeepSeek/GPT 独立切换。
- 采用 manager + agents-as-tools：投资经理必须调用技术、基本面和最终风险复核，缺失或顺序错误时宿主拒绝执行。
- 现有 MCP host 被包装为只读 function tools；所有模型角色都没有 `buy`/`sell` 权限，最终交易仍由宿主锁定快照价并通过账户规则执行。
- GPT 使用 Responses 模型路径；DeepSeek 使用 OpenAI 兼容 Chat Completions 路径，最终 JSON 统一由本地 contract 校验。
- 日志新增 SDK 版本、各角色 provider/model、工具调用序列、token usage 与 tracing 状态。
- 默认关闭 OpenAI tracing，只有显式设置 `ASTOCK_OPENAI_TRACING=1` 才启用。
- 移除运行时 Anthropic/Claude 依赖，新增 `openai-agents>=0.20,<1.0`。

### 尚未验收

- 尚未使用用户 API key 执行收费的 GPT/DeepSeek 推理；真实 Provider 的工具调用与输出一致性需要后续 shadow run 验证。

## 2026-08-13 — Phase 2/3 多角色、记忆与收益风险 Evals

- 接入官方 `claude-agent-sdk`，非 mock 模式默认使用程序化多角色编排。
- 新增技术分析师、基本面分析师和独立风控经理；全部模型角色只读，最终计划由宿主锁定行情价执行。
- 通过 Agent SDK hooks 记录子角色生命周期及下单前审计事件，并设置轮次/费用上限。
- A股和美股行情 MCP 新增一致的 `get_fundamentals` / `watchlist_fundamentals` 接口。
- 新增选择性历史记忆：严格早于行情截面日，限制日志和成交数量，生成稳定 `memory_id`。
- 日志升级为 `phase2-multi-agent-memory-v1`，记录角色结论、记忆评估和完整记忆快照。
- 增加输出完整性、计划与 SQLite 成交核对，以及同一行情日幂等保护，防止重复或虚构成交。
- 新增组合累计收益、等权基准、超额、波动、回撤、Sharpe、Sortino 与换手率 Evals。
- 仪表盘新增 `/api/performance` 和收益风险报告卡片。
- 回归测试增至 25 个，双市场离线 MCP、记忆、方向评分和收益风险报告端到端通过。

### 尚未验收

- 未使用用户 API key 执行收费的 Agent SDK 推理；已验证 SDK 版本、角色定义、MCP 配置、权限和输出 schema。
- 真实基本面接口仍依赖 AkShare/yfinance 的网络可用性，失败时会明确降级为离线参考数据。

## 2026-08-13 — Phase 1 稳定化基线

- 离线行情改用稳定哈希和固定历史锚点；同一标的、同一日期在不同进程和查询窗口保持一致。
- 决策日志使用行情截面日期，避免澳洲/亚洲时区运行美股任务时错配交易日。
- 五日预测只有经过完整五个交易日才进入评分。
- 评分器与仪表盘按日志记录的数据源评分，避免真实行情和模拟行情混用。
- 账户增加股票池、股数、价格校验；费用计入持仓成本，仓位限制按最新市值估算。
- A 股 T+1 改为只冻结当日新增股份，历史持仓仍可卖出。
- 日志新增 `as_of_date`、`generated_at`、`data_sources`、`offline`、策略及模型版本信息。
- 新增 `--offline` 和可覆盖的数据/日志目录，方便可复现及隔离验收。
- 约束 MCP 主版本 `<2.0`，防止 2.0 移除 FastMCP 1.x 接口后项目无法启动。
- 仪表盘修正持仓估值来源并转义模型生成文本。
- 新增 12 个自动化回归测试，并完成双市场 MCP 端到端 mock 验收。

### 尚未验收

- Claude 实盘决策路径需要有效的 `ANTHROPIC_API_KEY`，本轮未产生任何收费 API 调用。
- AkShare/yfinance 真实行情路径受运行环境网络条件影响，本轮以强制离线模式验收。
# 2026-08-19

- 新增双市场新闻 MCP：A股巨潮资讯/东方财富，美股 yfinance/Yahoo 与可选 Alpha Vantage。
- 固定6只股票池升级为12只受控候选宇宙、每日8只动态候选，并自动并入已有持仓。
- 新闻正文统一标记为不可信证据，按行情截面日过滤，防提示词注入与未来数据泄漏。
- 规则模式要求技术信号与新闻风险门控共同通过；重大负面事件可触发退出。
- 仪表盘新增新闻候选、事件分数、来源策略和头条摘要。
