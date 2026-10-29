# 分支保护、CI与按日期浏览的公开日报

用户选择的运行方式是：**本机运行Agent，GitHub只接收经过筛选的日报，Pages展示只读结果**。订阅登录、公司原文、完整模型输入、检查点和模拟账本数据库继续留在本机。网页不接收交易指令，也不保存认证令牌。

## main的保护

`.github/branch-protection.json` 记录实际应用的规则：必须通过PR更新，`tests-and-publication` 检查通过并保持分支最新，讨论须解决，管理员也受约束，禁止强推和删除。个人仓库的审批人数设为0，避免只有一个维护者而无法获得第二人批准；这不免除PR和测试要求。

CI在PR、main更新或手动触发时运行：完整Git历史凭据扫描、无模型凭据的回归测试、公开站点构建检查。任务只获仓库读取权限，拉取代码不保留认证信息。CI不调用订阅、不运行每日投资决策。

## 发布链路

1. 本机运行已有日循环，写入本地真实分析或模拟账本。
2. `publish_reports.py export` 按指定日期重放实际模拟成交、裁剪行情和已保存分析，仅投影允许公开的字段。
3. 上传到独立 `public-reports` 分支，每个报告以日期与内容哈希命名。相同内容保持首次发布时间，变更生成新版本，旧版本保留。
4. 本机通过已登录的GitHub CLI触发main上的Pages工作流。临时Git索引只添加经过验证的JSON，普通快进推送不修改本地工作分支，也不强制覆盖并发发布。
5. Actions从main读取构建代码，从公开分支读取报告；只将生成的 `_site/` 打包部署。`github-pages` 环境仅允许受保护分支部署。

Pages工作流也在main更新、手动触发及工作日17:20（上海时间，UTC09:20）重新构建已发布记录。这个定时构建不生成新判断、成交或净值。本机未运行/未发布时，网站继续显示最后实际报告；不会把网站部署时间当作Agent运行时间。

## 按日期查看

主页有日期选择、前后切换和 `?date=YYYY-MM-DD` 分享链接。每页分别显示报告日期、股价日期、最近公司分析日期及只分析/模拟账本/待确认状态；股价落后会提示。当天没有新的公司分析时仍显示最近实际分析的日期。

首次历史归档由已保存输入和按日期记录的实际模拟成交重建，并标注实际发布时间；不冒充当时已经公开发布。查看旧日期不会混入之后的新闻主题、判断或交易。选股后价格观察与实际模拟净收益保持独立。

## 本机操作

```powershell
# 首次导出本机已有真实资料日期，并预览静态站点
.venv\Scripts\python scripts/publish_reports.py export --all-saved-dates
.venv\Scripts\python scripts/publish_reports.py build
python -m http.server 5101 --bind 127.0.0.1 --directory data/pages-preview

# 日常发布；先在本机完成分析/模拟交易与真实股价刷新
.venv\Scripts\python scripts/publish_reports.py export --date 2026-10-29
.venv\Scripts\python scripts/publish_reports.py publish

# 已有日循环的集成入口：正式模拟模式，随后刷新、导出并请求部署
# 默认AI自主模拟，独立风控通过后自动排定未来收盘计划。
pwsh -NoProfile -File scripts/run_scheduled.ps1 -Paper -Publish
```

GitHub CLI需先通过正常登录配置本机仓库写入权限。发布脚本验证origin与目标仓库一致；不读取、打印或复制GitHub/ChatGPT登录文件。公开分支只允许指定格式的报告文件。离线或mock日循环不能启用 `-Publish`。已有分析确认命令见长期协议与模型接入文档。

## 公开字段与限制

公开：虚拟现金/持仓/成交费用、真实价格曲线、模型判断摘要、长期持仓计划、新闻标题/来源链接、必要日期和内容身份。

保留本机：完整输入、原始新闻正文/摘录、认证文件、数据库、绝对路径、内部运行/订单标识、原始错误输出。公开报告再次进行字段、时点、数值、凭据模式和本机路径检查。来源片段在本地可追溯，网站仅提供文章链接。

模拟账本须先对账。历史重放按真实记录的成交价和费用计算；价格未覆盖持仓时不补造收益。旧版预测保留原始周期。前复权数据可能反映后来复权调整，因此历史重建不等于当时价格文件的逐字副本。

公开数据是静态快照，本机运行依赖电脑在线、登录有效、数据源可用及订阅额度；GitHub上不保管订阅身份。模式扫描与哈希并不独立证明资料真实性或投资有效。

默认日循环已采用[收盘后研究与未来执行协议](AFTER_CLOSE_EXECUTION.md)：盘后形成的判断只能提出未来交易日的收盘价模拟计划，AI计划经独立风控及宿主校验后自动排定。公开页分别显示信息截止、判断完成、行情日和计划状态；原始PDF、跨页文本及内部订单身份保留本机。后续研究失败时，发布步骤仍尝试展示此前已合法执行的计划，运行本身继续保留失败标记。历史v2报告保留当时的人工确认状态。

GitHub官方说明：[Pages自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)、[受保护分支](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)。
