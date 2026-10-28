# 模型接入：API 密钥与订阅登录

更新日期：2026-10-29。默认使用官方 Codex CLI 的 ChatGPT 订阅登录；最小真实 GPT 6.1 Sol 请求已成功，返回完整 JSON 和用量。订阅额度和模型权限以登录账户为准。

A股默认策略已改为manager-long-v1长期协议，未改变认证路径或切换API。原单股票真实验收仍为旧manager-v1五日协议，未重新调用模型改写结果。长期资料、准备命令与当前缺口见 `LONG_TERM_INVESTING.md`。

## 本次验收

- 最小真实请求：ChatGPT 登录、`gpt-6.1-sol`、完整 JSON 响应，用量5914 tokens。
- 完整四角色真实请求：技术分析、基本面、经理、风控均使用 `gpt-6.1-sol`；4次请求，输入25416、输出795、合计26211 tokens（推理56已包含在输出中），耗时180577毫秒。宿主风控/结果契约通过，状态为 `analysis_complete`，零订单。
- 输入是明确标记的 AAPL 单股票合成验收数据与本地虚拟账户，不代表真实市场表现；临时检查点已清理。
- 124项自动化测试通过，包含订阅/密钥隔离、系统代理、完整流、失败预算、超时终止、取消及四角色暂停/恢复。
- 双市场离线验收已通过只分析零写入、重复运行幂等和报告步骤；附加60日价格回测未在90秒测试时限内完成，未登记为通过。
- 2026-10-28 A股300750单股票真实资料四角色shadow成功：4次请求、31291 tokens、模型297.652秒、全程308.312秒，零订单。财务缺少披露时间而不可用于时点判断；技术专家有2处角色越界措辞，不能称为语义质量全通过。多股票仍待验收，美股暂停；详见 `logs/live/2026-10-28.cn/daily-brief.md`。

## 当前程序

命令行默认 `--model-access subscription --framework langgraph`。`agent/subscription.py` 调用官方 `codex exec`，只查询登录方式；登录、凭据存储和刷新由官方客户端管理，项目不读取或复制 `auth.json`。

技术分析、基本面分析、经理和风控各发一次请求，模型仍为 `gpt-6.1-sol`。宿主提供冻结后的市场资料、固定字段的虚拟账本与历史预测，模型只返回结构化结果。工具、联网搜索、插件与 shell 均在子客户端中禁用；行情和新闻仍由宿主提供。经理计划必须通过独立风控及宿主校验，有交易的模拟盘计划仍等待人工确认。

```powershell
codex login
.venv\Scripts\python agent/run_daily.py --preflight
.venv\Scripts\python scripts/check_subscription.py --smoke
.venv\Scripts\python agent/run_daily.py --market cn --shadow
```

`--preflight` 和不带 `--smoke` 的连接检查不推理；`--smoke` 发送固定短文本，会消耗订阅额度。`--offline --shadow` 可用明确标记的合成行情验证订阅分析，但其结果不能用于宣称真实市场表现。

Windows 下子进程会沿用系统代理；已设置的 `HTTP_PROXY` / `HTTPS_PROXY` 优先。请从普通本机终端运行：受限执行环境可能禁止异步循环需要的本机回环连接、临时目录或客户端状态写入。本次测试通过正常本地执行完成，没有改动系统代理或登录配置。

单次请求默认120秒超时，超时或取消时终止本次子进程树，清理使用继承目录权限的临时文件，避免管道等待。调用次数、输入字符数、完成请求的 token 用量都有预算；token 限额在请求完成后校验，不能保证一次在途请求不超限。失败、不完整输出、工具活动、无效 JSON 或风控失败都停止本轮，不自动改用 API。

需要保留的 API 对照使用 `--model-access api --framework sdk`（或 `langgraph`），并在本机设置 `OPENAI_API_KEY`。订阅子进程会过滤 API 密钥和自定义 API 地址。

## 官方可选路径

| 路径 | 认证与用量 | 本项目状态 |
|---|---|---|
| 默认官方 Codex CLI | ChatGPT 登录，消耗订阅额度 | 已实现，最小真实请求成功 |
| OpenAI API 对照 | API 项目密钥，按 API 用量计费 | 已保留，必须显式选择 |
| 直接 Sign in with ChatGPT 计划用量授权 | 应用自行完成 OAuth 授权和请求适配 | 本次未采用，以下保留为后续参考 |

Codex CLI 与 OpenAI Agents SDK 是不同接口；本项目用订阅传输适配器连接既有 LangGraph 流程，保留独立的 API 对照路径。

## 直接使用计划用量授权的适配要求

官方 Sign in with ChatGPT 文档明确支持本地/开源应用申请计划用量权限。应用需要保存稳定的主机标识，完成浏览器授权、校验授权结果、管理过期与刷新，并使用该次授权对应的 access token。不能把其他登录会话的 token 当作已完成这一授权。

当前预览要求每次 HTTP 推理使用 `store: false`、`stream: true`，发送所需上下文；`max_output_tokens`、`previous_response_id`、`metadata` 等字段在该路径不受支持。工具格式也有额外要求，因此必须专门适配，不能把 token 填进现有配置后沿用原请求。

使用官方 `https://api.openai.com/v1/responses` 端点；读取完整流并收到 `response.completed` 才算成功。开始流式输出后仍可能出现额度不足、失败或中断，不能提前提交订单。

`gpt-6.1-sol` 保留为拟定模型。账户模型目录用于选择，首次成功完成的真实推理用于验收；模型不支持时明确停止，不静默替换模型。订阅用量受额度约束，不能视为无限调用，也不等于 API 预付额度。

对外收费或远程托管服务需按官方文档的相应接入流程核实；本地授权方案不能直接推定覆盖这些场景。

## 接入验收标准

1. 用户通过官方登录授权，凭据只保存在本地安全位置，不进入仓库或实验日志。
2. 先用最小只分析请求验证 `gpt-6.1-sol`，明确记录认证路径、实际模型、完成状态和用量。
3. 验证过期、刷新、额度耗尽、流式失败、超时和结构化输出失败都能停止本轮，不产生订单。
4. 四角色调用、只读工具、独立风险审批和已有模拟账本约束通过同样的回归验收。
5. 计费路径必须显式选择；订阅失败后不自动改用收费 API。

## 官方资料

- 非交互模式、复用登录和 JSON Schema：https://learn.chatgpt.com/docs/non-interactive-mode
- 认证方式：https://learn.chatgpt.com/docs/auth
- Codex SDK（含 Python SDK 和 GPT 6.1 Sol 示例）：https://learn.chatgpt.com/docs/codex-sdk
- 本地/开源应用使用 ChatGPT 计划：https://developers.openai.com/siwc/token-sharing-open-source
- 模型与推理：https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference
- 当前预览限制：https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations
