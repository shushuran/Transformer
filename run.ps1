# 统一的解释器入口：固定 conda env + 关掉 user-site + 把 src 加进模块路径
# 用法:  .\run.ps1 -m tlab env
#        .\run.ps1 -m tlab prepare
param([Parameter(ValueFromRemainingArguments = $true)] $Args)

$env:PYTHONNOUSERSITE = "1"
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"
$py = "D:\anaconda\envs\pytorch\python.exe"
if (-not (Test-Path $py)) { Write-Error "找不到解释器: $py"; exit 1 }
& $py @Args
exit $LASTEXITCODE
