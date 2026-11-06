$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Conda = "C:/Users/libh/miniconda3_soarm/Scripts/conda.exe"
$EnvPython = "C:/Users/libh/miniconda3_soarm/envs/ai_net/python.exe"

if (-not (Test-Path -LiteralPath $EnvPython)) {
    # 复用本机已有的 CUDA PyTorch，避免重新下载数 GB 的 GPU 运行库。
    & $Conda create -n ai_net --clone lerobot -y
}

& $EnvPython -m pip install -r (Join-Path $ProjectRoot "requirements-train.txt")

& $EnvPython -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
