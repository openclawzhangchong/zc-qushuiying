@echo off
rem zc-qushuiying - Start WebUI (run deploy.ps1 first on a fresh machine)
rem
rem Works from either layout:
rem   A) repo root:      start.bat + remove-ai-watermarks\{webui,.venv}
rem   B) upstream root:  start.bat + {webui,.venv}   (where deploy.ps1 installs it)
rem
rem Environment variables:
rem   PORT=8000                listen port
rem   WEBUI_ONNX_DEVICE=cpu    force CPU inference (default: GPU first, auto fallback)
rem   ZC_NO_PAUSE=1            suppress pause-on-error (for scripted use)
rem
rem Double-click to start; close the window to stop the service.
cd /d "%~dp0"

set "APP="
set "PY="

rem --- Layout A: upstream cloned into a subdirectory (repo root) ---
if exist "remove-ai-watermarks\webui\app.py" (
    set "APP=remove-ai-watermarks\webui\app.py"
    set "PY=remove-ai-watermarks\.venv\Scripts\python.exe"
)

rem --- Layout B: start.bat sits at the upstream root (deploy.ps1 default) ---
if exist "webui\app.py" (
    set "APP=webui\app.py"
    set "PY=.venv\Scripts\python.exe"
)

if not defined APP (
    echo [ERROR] Could not find the deployed WebUI.
    echo Looked for:
    echo   remove-ai-watermarks\webui\app.py
    echo   webui\app.py
    echo Please run deploy.ps1 first:
    echo   powershell -ExecutionPolicy Bypass -File deploy.ps1
    if not defined ZC_NO_PAUSE pause
    exit /b 1
)

if not exist "%PY%" (
    echo [ERROR] Python environment not found: %PY%
    echo Please re-run deploy.ps1 to install dependencies.
    if not defined ZC_NO_PAUSE pause
    exit /b 1
)

if "%PORT%"=="" set PORT=8000

echo [zc-qushuiying] Starting WebUI on port %PORT% ...
echo [zc-qushuiying]   app: %APP%
echo [zc-qushuiying]   python: %PY%
echo [zc-qushuiying] Press Ctrl+C or close this window to stop.
echo.

"%PY%" "%APP%"

rem Pause only on failure, and only when interactive (skip for scripted use).
if errorlevel 1 if not defined ZC_NO_PAUSE pause
exit /b %errorlevel%
