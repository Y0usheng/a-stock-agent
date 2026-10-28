param(
    [ValidateSet('cn', 'us', 'both')][string]$Market = 'cn',
    [ValidateSet('sdk', 'langgraph')][string]$Framework = 'langgraph',
    [ValidateSet('subscription', 'api')][string]$ModelAccess = 'subscription',
    [switch]$Paper,
    [switch]$Offline,
    [switch]$Mock
)
$ErrorActionPreference = 'Stop'
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
} finally { Pop-Location }
Write-Output "运行日志: $runLogPath"
exit $runExitCode
