@echo off
rem zc-qushuiying - pin/verify GPU dependency versions (Windows)
rem
rem Why this exists: the V100 (Volta sm_70) only works with a specific
rem onnxruntime-gpu + cuDNN pair. Other versions load fine but fail at the
rem first CUDA kernel with cudaErrorNoKernelImageForDevice. If you changed
rem versions for testing (e.g. fault injection), run this to restore.
rem
rem Usage:  check-gpu.bat          (report only)
rem         check-gpu.bat --fix    (reinstall pinned versions)
setlocal enabledelayedexpansion

set "PY=%~dp0remove-ai-watermarks\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0.venv\Scripts\python.exe"

if not exist "%PY%" (
    echo [ERROR] Python venv not found. Run deploy.ps1 first.
    exit /b 1
)

set "WANT_ORT=1.19.2"
set "WANT_CUDNN=9.10.2.21"
set "TMPV=%~dp0.gpuver.tmp"

echo [check] Python: %PY%

rem Query versions via a temp file: nested quotes inside for/f are fragile in cmd.
"%PY%" -c "import importlib.metadata as m; o=m.version('onnxruntime-gpu') if True else ''; print(o); print(m.version('nvidia-cudnn-cu12'))" > "%TMPV%" 2>nul
set "ORT=(not installed)"
set "CUDNN=(not installed)"
set "LN=0"
for /f "usebackq delims=" %%v in ("%TMPV%") do (
    set /a LN+=1
    if !LN!==1 set "ORT=%%v"
    if !LN!==2 set "CUDNN=%%v"
)
if exist "%TMPV%" del "%TMPV%" >nul 2>&1

echo [check] onnxruntime-gpu   = !ORT!   ^(want %WANT_ORT%^)
echo [check] nvidia-cudnn-cu12 = !CUDNN!   ^(want %WANT_CUDNN%^)

set "MISMATCH=0"
if not "!ORT!"=="%WANT_ORT%" set "MISMATCH=1"
if not "!CUDNN!"=="%WANT_CUDNN%" set "MISMATCH=1"

rem Self-test: can CUDA actually execute a kernel on this GPU?
rem Suppress onnxruntime's own stderr noise; only our one-line verdict matters.
"%PY%" -c "import sys; sys.path.insert(0,r'%~dp0remove-ai-watermarks\webui'); sys.path.insert(0,r'%~dp0webui'); import gpu_accel; ok,why=gpu_accel._cuda_selftest(); print('[selftest]', ok, why); sys.exit(0 if ok else 2)" 2>nul
if errorlevel 2 set "MISMATCH=1"

if "!MISMATCH!"=="0" (
    echo [check] OK - versions pinned and CUDA self-test passed.
    exit /b 0
)

echo.
echo [check] MISMATCH detected.
if /i "%~1"=="--fix" (
    echo [fix] reinstalling pinned versions via aliyun mirror...
    pushd "%~dp0remove-ai-watermarks" 2>nul || pushd "%~dp0"
    uv pip install --python "!PY!" --index-url https://mirrors.aliyun.com/pypi/simple/ ^
        "onnxruntime-gpu==%WANT_ORT%" "nvidia-cudnn-cu12==%WANT_CUDNN%" ^
        nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12
    popd
    echo [fix] done. Restart the WebUI for changes to take effect.
) else (
    echo [check] Run:  check-gpu.bat --fix
)
exit /b 1
