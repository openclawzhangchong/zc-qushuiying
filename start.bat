@echo off
rem zc-qushuiying - 启动 WebUI（首次使用请先运行 deploy.ps1 部署）
cd /d "%~dp0"
if not exist "remove-ai-watermarks\webui\app.py" (
    echo 尚未部署。请先运行：
    echo   powershell -ExecutionPolicy Bypass -File deploy.ps1
    pause
    exit /b 1
)
set PORT=8000
"remove-ai-watermarks\.venv\Scripts\python.exe" "remove-ai-watermarks\webui\app.py"
