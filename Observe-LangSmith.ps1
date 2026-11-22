<#
.SYNOPSIS
观察一个 TCM_Fusion 样品目录或运行目录。
.DESCRIPTION
等待运行工件，跟随正在进行的任务，或上传已完成样品的回放。
从 config\langsmith.local.json 读取追踪配置。可用 -Preview 本地查看。
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string] $Directory,

    [switch] $Preview
)

$ErrorActionPreference = 'Stop'

if (-not $Directory) { $Directory = Read-Host '输入要观察的样品或运行目录' }
$Directory = $Directory.Trim()
if ($Directory -match '^(["\x27])(.+)\1$') { $Directory = $Matches[2] }
if ([string]::IsNullOrWhiteSpace($Directory)) { throw '请输入一个目录路径。' }
$observerPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Directory)
if ([IO.File]::Exists($observerPath)) { throw '请选择目录。' }

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw '项目缺少 .venv\Scripts\python.exe。' }
$observerArgs = @('-B', '-X', 'utf8', '-m', 'TCM_Fusion.cli.langsmith_trace', $observerPath,
                  '--follow', '--include-decision-summary')

if (-not $Preview) { $observerArgs += '--upload' }
Push-Location -LiteralPath $PSScriptRoot
try {
    $PSNativeCommandUseErrorActionPreference = $false
    & $python @observerArgs
    $observerExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $observerExitCode
