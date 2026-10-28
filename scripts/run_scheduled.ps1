param(
    [ValidateSet('cn', 'us', 'both')][string]$Market = 'cn',
    [ValidateSet('sdk', 'langgraph')][string]$Framework = 'langgraph',
    [ValidateSet('subscription', 'api')][string]$ModelAccess = 'subscription',
    [switch]$Paper,
    [switch]$Offline,
    [switch]$Mock,
    [switch]$Publish
)
$ErrorActionPreference = 'Stop'
if ($Publish -and ($Market -ne 'cn' -or $Offline -or $Mock)) { throw '公开发布只支持A股真实资料；不可发布离线或mock运行。' }
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw '请先安装项目虚拟环境。' }
$runLogRoot = if ($env:ASTOCK_DATA_DIR) { Join-Path $env:ASTOCK_DATA_DIR 'run_logs' } else { Join-Path $projectRoot 'data\run_logs' }
New-Item -ItemType Directory -Path $runLogRoot -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$suffix = [guid]::NewGuid().ToString('N').Substring(0, 8)
$runLogPath = Join-Path $runLogRoot "$stamp.$Market.$suffix.log"
$effectiveFramework = if ($Mock) { 'sdk' } else { $Framework }
$runArguments = @('agent/run_daily.py', '--market', $Market, '--framework', $effectiveFramework, '--model-access', $ModelAccess)
if (-not $Paper) { $runArguments += '--shadow' }
if ($Offline) { $runArguments += '--offline' }
if ($Mock) { $runArguments += '--mock' }
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath @runArguments *> $runLogPath
    $runExitCode = $LASTEXITCODE
    if ($Publish -and $runExitCode -eq 0) {
        & $pythonPath scripts/refresh_trading_view.py --provider sina >> $runLogPath 2>&1
        if ($LASTEXITCODE -ne 0) { Write-Warning '真实股价刷新失败，将保留旧缓存并在报告中标明行情日期。' }
        $publicationDay = & $pythonPath -c "from datetime import datetime;from servers.data_quality import market_zone;print(datetime.now(market_zone('cn')).date().isoformat())"
        & $pythonPath scripts/publish_reports.py export --date $publicationDay >> $runLogPath 2>&1
        if ($LASTEXITCODE -eq 0) {
            & $pythonPath scripts/publish_reports.py publish >> $runLogPath 2>&1
        }
        $runExitCode = $LASTEXITCODE
    }
} finally { Pop-Location }
Write-Output "运行日志: $runLogPath"
exit $runExitCode
