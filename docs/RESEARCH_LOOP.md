# A股实验闭环

更新：2026-10-27。入口为 `scripts/research.py`，默认保存到 `data/experiments.db`，仪表盘“实验”页展示研究状态。当前仅支持A股，不会下单。

流程为：**登记假设和标准 → 冻结资料 → 保存模型判断（如需要）→ 训练/验证/测试 → 复盘 → 提出下一轮假设**。闭环调度、评分、复盘均为本地确定性代码；只有显式执行 `predict` 或语义评审时调用订阅模型。

## 已验收的结果

- 全量177项自动化测试通过，其中21项覆盖研究计划、冻结/追加、时间边界、首次模型输出、失败留存、固定配对、成本、复盘和重放。
- 合成工程样本：三个变体、三个阶段，9次完整运行；结论为 `unverified_engineering`，不会认定真实策略有效。
- 已有真实历史价格：固定600036、600519两个标的，训练2026-05-26至06-23、验证06-24至07-21、测试07-22至09-30。固定候选为技术策略，固定基线为受仓位上限约束的买入持有；9次尝试中6次完成、3次因没有历史新闻而保留 `no_data`。
- 真实价格测试期有90条到期配对预测、9个不重叠信号日期。技术策略命中率27.778%，基线38.889%，差值−11.111个百分点，结论为 `reject`。组合净收益分别为−1.105%和+0.994%；它们与方向命中率分别计算。
- 两份演示均保存计划、复盘JSON/Markdown和重放回执；最终验收的15次已完成运行全部原样重放成功，另3次无数据保留。汇总回执为 `logs/demo/research-acceptance-cn.json`。此次验收没有调用真实模型，没有写入模拟账户订单。Agent适配器用本地替身验证，不代表历史GPT业绩。

报告见 `logs/demo/research-engineering-cn.md`、`logs/demo/research-historical-cn.md`；完整价格、逐日标签、模型日志和成交重放包保存在本地研究库中。JSON报告列出每次尝试ID、输入指纹和旧实验注册表ID，可继续追溯。

## 快速重跑流程

在项目根目录执行：

```powershell
# 合成数据验证完整流程，不调用模型
.venv\Scripts\python scripts/research.py demo

# 使用已保存的真实历史价格；没有新闻就明确记为无数据
.venv\Scripts\python scripts/research.py demo --prices data/historical/cn-2026-10-02.json

.venv\Scripts\python scripts/research.py list
```

每次 `demo` 都新建研究，不覆盖原有实验记录；同一假设家族累计记录尝试次数。演示报告文件使用固定名称，最新一轮会更新文件；历史结果保留在库内。研究库和本地历史价格位于被Git忽略的 `data/`，需要另行备份。需要隔离演示时，在子命令前指定 `--db data/my_research.db`；仪表盘默认读取 `data/experiments.db`。

## 先登记，再运行

计划JSON示例（历史重建）：

```json
{
  "name": "技术策略相对买入持有的方向判断",
  "hypothesis": "固定股票池与成本后，技术策略能否提升5日方向命中率？",
  "market": "cn",
  "mode": "retrospective",
  "symbols": ["600036", "600519"],
  "horizon_days": 5,
  "splits": {
    "train": ["2026-05-26", "2026-06-23"],
    "validation": ["2026-06-24", "2026-07-21"],
    "test": ["2026-07-22", "2026-09-30"]
  },
  "baseline": "buy_hold",
  "candidate": "technical",
  "variants": [
    {"id": "buy_hold", "kind": "buy_hold"},
    {"id": "technical", "kind": "technical"},
    {"id": "news_gate", "kind": "news_gate"}
  ],
  "criteria": {
    "metric": "accuracy_pct",
    "min_gain": 0,
    "min_matured_predictions": 20,
    "min_nonoverlapping_dates": 4,
    "min_coverage": 1,
    "max_drawdown_pct": 20,
    "max_total_cost_pct": 2,
    "require_semantic_review": false
  }
}
```

示例只用于说明格式；历史观察后的登记不算向前预注册。正式向前研究将 `mode` 改为 `prospective`，测试期开始日必须晚于实际登记日。股票限于现有受控A股池；不支持美股、动态选股或模型自动挑股票。

`accuracy_pct` 的增益单位为命中率百分点；选择 `net_return_pct` 时为净收益百分点。成本上限是费用与滑点占初始资金的百分比。覆盖率约束防止只挑有利预测评分；不重叠日期门槛减少将相邻5日预测误当独立样本的风险，但不提供统计显著性保证。

```powershell
.venv\Scripts\python scripts/research.py plan my_plan.json
# 将上一步实际输出的ID填入变量
$researchId = "research_实际ID"
.venv\Scripts\python scripts/research.py freeze $researchId --prices data/historical/cn-2026-10-02.json --news my_news.json
.venv\Scripts\python scripts/research.py run $researchId --phase train
.venv\Scripts\python scripts/research.py run $researchId --phase validation
.venv\Scripts\python scripts/research.py run $researchId --phase test
.venv\Scripts\python scripts/research.py review $researchId
.venv\Scripts\python scripts/research.py replay $researchId attempt_实际ID
```

没有新闻时可省略 `--news`；新闻变体不会因此偷偷使用合成数据。固定基线和候选的前序阶段尚未完成时，下一阶段记为 `pending`。主要成功标准、股票池、费用、仓位上限及相关模块代码指纹在登记时固定，不能在原研究中改写。train/validation/test只负责时间隔离和顺序检查，不自动训练模型或搜索参数；改变参数须注册新变体/研究。

## 冻结什么，如何追加

价格JSON沿用 `evals/price_data.py` 的导出格式，包含 `market`、按股票分组的 `histories` 和可选 `manifest`。有文件指纹时先检查完整文件，再取研究子集。成交量单位为股。

新闻文件为事件数组或 `{"events": [...]}`。每条保存 `event_id`、`symbol`、`title`、`summary`、`source`、`url`、`published_at`、`first_seen_at`、`data_quality`、`sentiment_score`；内容指纹由宿主计算并核对。事件编号负责关联，真正进入判断的文字、时间、来源及哈希仍需在中心库保留，单独保存编号不足以复现。

- `retrospective`/`engineering` 按信息发布时间和市场收盘时间重建可见范围，明确不声称当时已经采集到。
- `prospective` 要同时满足发布时间与首次采集时间；首次采集未知的新闻不可用。
- 同一研究只允许追加未来日线与后来可见的新闻。改写旧价格、补入历史缺失交易日、删除新闻、回填旧新闻或改复权口径均拒绝；需要新研究。
- 每个阶段的重放包只包含该阶段截止日前资料。追加未来数据不会改变已完成训练阶段的指纹与结果。
- 每条方向评分绑定实际到期日与结果价格指纹。各阶段最后5个信号日不作该阶段方向评分，避免标签跨入下一阶段。

指纹证明保存内容一致，不独立证明供应商数据真实、新闻覆盖完整或来源未被篡改。

## 接入订阅Agent

支持两条路径，均只接受零订单shadow输出。

**冻结研究输入生成判断**：先获取研究协议的变体指纹，再写入计划中的 `agent` 变体。

```powershell
# 此命令只计算身份，不调用模型；合成研究须加 --synthetic
.venv\Scripts\python scripts/research.py variant --prompt-version manager-v1
```

将返回的 `variant_fingerprint` 填入 `{"id":"agent_v1","kind":"agent","prompt_version":"manager-v1","variant_fingerprint":"完整64位哈希"}`，并在运行前把它指定为候选。登记并冻结后：

```powershell
# 显式消耗订阅额度：沿用既有四角色、调用/耗时/token预算
.venv\Scripts\python scripts/research.py predict $researchId agent_v1 --day 2026-10-27
```

日期须属于计划范围、已收盘且有冻结价格和足够预热历史。该路径只读取冻结行情/最近三天可见新闻，财务明确为缺失、记忆为空、账户为固定初始虚拟账户。不能视为完整历史投研环境。历史重建需显式加 `--allow-retrospective`，仍不会成为向前证据。模型输出首次落库后拒绝重做/替换；生成失败也保留尝试记录，未知的失败token用量不补成零。

**关联现有分析日志**：用固定池日常shadow入口采集日志；从一份同协议日志获取 `variant --log 路径` 身份后，在未来测试开始前登记研究。日常固定池入口为：

```powershell
.venv\Scripts\python agent/run_daily.py --market cn --shadow --research-symbols 600036 600519
.venv\Scripts\python scripts/research.py variant --log logs/decisions/shadow/实际日志.json
.venv\Scripts\python scripts/research.py attach $researchId agent_v1 logs/decisions/shadow/实际日志.json
```

同日不同模型变体必须拥有完全一致的冻结输入，包括账户和记忆；模型身份、日期、股票、新闻内容/可见时间、价格和日志指纹均会检查。向前预测还须在下一次实际观测交易日开盘前生成并登记；不能事后补写成当时已有的决策。已有输出每个信号日缺一个就记为无数据，评分命令不会自动调用模型补历史。

日常日志与冻结研究预测使用不同的策略协议，不能混用其变体指纹。可选通过 `attach ... --semantic 核验报告.json` 关联已有离线语义核验；宿主重新核对原日志、命题范围和引用，不采信外部提供的汇总数字。把 `require_semantic_review` 预设为true时，缺关键核验会阻止保留结论；模型语义判断仍不等于人工标注真值。

## 如何读复盘

三类结果独立保存：

| 类别 | 指标与含义 |
|---|---|
| 判断质量 | 到期方向命中率、覆盖率、原始Brier/ECE；只在双方都有有效预测的相同案例上比较增益 |
| 组合表现 | 统一下一交易日开盘成交、整手、T+1、仓位/流动性限制、佣金/税费/滑点下的净收益、回撤和交易次数 |
| 研究效率/资料 | 保存日志中的模型调用、tokens、耗时，证据缺口、语义覆盖与错误；生成失败和无数据尝试亦保留 |

Agent组合是**把已保存的方向预测交给统一仓位规则分配**的模拟组合，不是原始经理订单或风控批准股数的收益。只分析日志本身不产生账户收益；冻结输出重放不调用模型。买入持有也遵守相同仓位上限，因此可能保留现金。

最终只比较预先指定的候选与基线，不从测试集自动选胜者。`insufficient_data`/`awaiting_data`/`awaiting_review` 表示尚不能判断；`not_comparable` 表示输入口径不同；`reject` 表示未达标准；工程或合成样本只能是 `unverified_engineering`。历史候选通过也只会 `retain_for_forward_validation`；向前候选通过只会 `retain_for_paper_validation`，都不自动晋级或修改风险限制。

复盘自动生成可检查的下一轮建议；它是规则模板生成的提议，不会暗中再调用模型或自动开展研究。新研究应设置 `parent_id`；父研究须已有复盘，新的测试期必须晚于父测试期。所有关联尝试累计展示。库本身不能阻止人故意换名字掩盖相同假设，仍需要研究纪律。

## 当前边界

本次跑通流程和真实价格规则基线，没有形成真实Agent连续向前效果证据。后续需要按已登记版本积累真实新闻、每日模型输出、语义复核与成熟标签。历史大模型可能已经知道后续事实，历史重建不能消除此偏差。

当前采用观测到的共同交易日，计划区间应落在冻结数据的实际交易日；不提供完整官方交易日历、复权公司行动、历史成分/退市重建、停牌/涨跌停/盘口撮合。固定现有受控池存在存活者偏差。费用沿用项目模拟配置，未声称实盘精确成本。相关代码变动后需保留旧版本运行环境才能复现旧研究，当前代码会拒绝声称原样重放。
