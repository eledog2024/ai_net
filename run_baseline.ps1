$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:/Users/libh/miniconda3_soarm/envs/ai_net/python.exe"

Set-Location $ProjectRoot

if (-not (Test-Path -LiteralPath "data")) {
    New-Item -ItemType Junction -Path "data" -Target "D:/智能通信网络数据集" | Out-Null
}

& $Python "script/Q1/stage_1/split_dataset.py"

& $Python "python/Q3/stage_1/build_cache.py"

& $Python "python/Q3/stage_2/train.py" `
    --weights "result/Q3/weights/cnn4_best.pt" --eval-only

& $Python "python/Q3/stage_2/train.py" --tag "baseline_v2"
