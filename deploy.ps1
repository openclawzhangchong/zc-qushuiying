# zc-qushuiying 一键部署脚本（Windows）
# 作用：克隆上游 remove-ai-watermarks -> 安装 WebUI -> uv sync 装依赖 -> 防火墙放行 -> 打印启动方式
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File deploy.ps1            # 默认端口 8000
#   powershell -ExecutionPolicy Bypass -File deploy.ps1 -Port 8080 # 指定端口
#   powershell -ExecutionPolicy Bypass -File deploy.ps1 -NoFirewall # 不改防火墙

param(
    [int]$Port = 8000,
    [switch]$NoFirewall,
    [switch]$NoGpu
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$upstream = Join-Path $repoRoot "remove-ai-watermarks"

function Info($msg) { Write-Host "[zc-qushuiying] $msg" -ForegroundColor Cyan }

# 1. 依赖检查：git / uv
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "未找到 git，请先安装 Git for Windows。" -ForegroundColor Red
    exit 1
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Info "未找到 uv，正在安装..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Host "uv 安装后未生效，请重开终端再运行本脚本。" -ForegroundColor Red
        exit 1
    }
}

# 2. 克隆上游（直连失败时自动回退镜像加速）
if (-not (Test-Path (Join-Path $upstream ".git"))) {
    Info "克隆上游仓库 remove-ai-watermarks ..."
    $urls = @(
        "https://github.com/wiltodelta/remove-ai-watermarks",
        "https://ghfast.top/https://github.com/wiltodelta/remove-ai-watermarks"
    )
    $ok = $false
    foreach ($u in $urls) {
        git clone --depth 1 $u $upstream 2>$null
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
        Info "克隆失败（$u），尝试下一个源..."
    }
    if (-not $ok) { Write-Host "上游克隆失败，请检查网络。" -ForegroundColor Red; exit 1 }
}
else {
    Info "上游仓库已存在，跳过克隆。"
}

# 3. 安装 WebUI 到上游目录（app.py 按"webui 位于上游克隆目录内"定位 CLI）
Info "安装 WebUI ..."
Copy-Item -Recurse -Force (Join-Path $repoRoot "webui") (Join-Path $upstream "webui")

# 4. 安装依赖（uv.lock 锁定版本，含便携版 ffmpeg）
Info "安装依赖（video + migan，首次较慢）..."
Push-Location $upstream
try {
    uv sync --frozen --extra video --extra migan
    if ($LASTEXITCODE -ne 0) { throw "uv sync 失败" }
}
finally {
    Pop-Location
}

# 6. 可选：安装 GPU 加速（NVIDIA 显卡，MI-GAN/LaMa 推理提速）
if (-not $NoGpu) {
    $gpu = (nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($gpu) {
        Info "检测到 NVIDIA GPU: $gpu，安装 onnxruntime-gpu + CUDA 运行时..."
        $py = Join-Path $upstream ".venv\Scripts\python.exe"
        # onnxruntime-gpu 1.19.2 + cuDNN 9.10.2.21 是仍支持 Volta(sm_70) 的组合；
        # 更新的 ORT 在 V100 上报 cudaErrorNoKernelImageForDevice，
        # 更新的 cuDNN(9.12+) 在 V100 上报 CUDNN_STATUS_EXECUTION_FAILED_CUDART。
        & uv pip install --python $py `
            "onnxruntime-gpu==1.19.2" "nvidia-cudnn-cu12==9.10.2.21" `
            "nvidia-cublas-cu12" "nvidia-cuda-runtime-cu12" "nvidia-cufft-cu12"
        if ($LASTEXITCODE -eq 0) { Info "GPU 推理依赖已安装（启动时自动优先 CUDA，失败回退 CPU）" }
        else { Info "GPU 依赖安装失败，将使用 CPU（功能不受影响）" }
    }
    else {
        Info "未检测到 NVIDIA GPU，跳过 GPU 加速安装（使用 CPU）"
    }
}

# 7. 防火墙放行（需要管理员权限，失败不影响部署）
if (-not $NoFirewall) {
    netsh advfirewall firewall add rule name="Remove AI Watermarks WebUI" dir=in action=allow protocol=TCP localport=$Port | Out-Null
    if ($LASTEXITCODE -eq 0) { Info "防火墙已放行 TCP $Port（局域网可访问）" }
    else { Info "防火墙放行失败（可能无管理员权限），局域网访问时需手动放行。" }
}

Write-Host ""
Info "部署完成！启动方式："
Write-Host "  $upstream\webui\start.bat"
Write-Host ""
Write-Host "  本机访问:   http://127.0.0.1:$Port/"
Write-Host "  局域网访问: http://<本机IP>:$Port/  (启动时控制台会打印实际 IP)"