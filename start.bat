@echo off
rem zc-qushuiying - Start WebUI (run deploy.ps1 first on a fresh machine)
rem
rem Environment variables:
rem   PORT=8000                listen port (keep in sync with deploy.ps1 -Port)
rem   WEBUI_ONNX_DEVICE=cpu    force CPU inference (default: GPU first, auto fallback)
rem
rem Double-click to start; close the window to stop the service.
cd /d "%~dp0"

if not exist "remove-ai-watermarks\webui\app.py" (
    echo [ERROR] Not deployed: missing remove-ai-watermarks\webui\app.py
    echo Please run first:
    echo   powershell -ExecutionPolicy Bypass -File deploy.ps1
    pause
    exit /b 1
)

if "%PORT%"=="" set PORT=8000

echo [zc-qushuiying] Starting WebUI on port %PORT% ...
echo [zc-qushuiying] Press Ctrl+C or close this window to stop.
echo.

"remove-ai-watermarks\.venv\Scripts\python.exe" "remove-ai-watermarks\webui\app.py"
pause
