@echo off
REM ============================================================
REM SC Gen 6 - Windows Launcher (Vulkan Optimized)
REM ============================================================
REM Double-click to run on native Windows with llama.cpp Vulkan.
REM ============================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ============================================================
echo    SC Gen 6 - Litigation Support RAG
echo    Windows Native (Vulkan GPU Acceleration)
echo ============================================================
echo.

REM Check for virtual environment
if exist "venv\Scripts\python.exe" (
    set "PY_CMD=venv\Scripts\python.exe"
) else (
    echo [ERROR] Virtual environment not found.
    echo.
    echo First-time setup required:
    echo   1. Open command prompt here
    echo   2. Run: python -m venv venv
    echo   3. Run: venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM ============================================================
REM Check if llama.cpp server is already running
REM ============================================================
echo [INFO] Checking llama.cpp server status...

tasklist /FI "IMAGENAME eq llama-server.exe" 2>NUL | find /I /N "llama-server.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo [OK] llama-server.exe is already running
    goto :check_health
) else (
    echo [INFO] llama-server.exe not found, starting it now...
    goto :start_server
)

:start_server
REM ============================================================
REM Start llama.cpp server with Vulkan acceleration
REM ============================================================
echo [INFO] Starting llama.cpp server in background...

REM Set Vulkan fix for AMD GPUs
set GGML_VK_FORCE_MAX_ALLOCATION_SIZE=4294967295

REM Check if start_llama_background.bat exists
if not exist "start_llama_background.bat" (
    echo [ERROR] start_llama_background.bat not found!
    pause
    exit /b 1
)

REM Start the server in background
call start_llama_background.bat

REM Give it a moment to initialize
timeout /t 3 /nobreak >NUL

:check_health
REM ============================================================
REM Wait for server to be ready (model loaded)
REM ============================================================
echo [INFO] Waiting for llama.cpp server to load model...
echo.

%PY_CMD% scripts\check_llama_server.py
if errorlevel 1 (
    echo.
    echo [ERROR] llama.cpp server failed to start or model not loaded.
    echo.
    echo Troubleshooting:
    echo   1. Check if llama-server.exe exists in llama-cpp folder
    echo   2. Check if model path is correct in config\llm_runtime.json
    echo   3. Check logs\llama_server.log for errors
    echo   4. Try manually: .\start_llama_background.bat
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] llama.cpp server is ready with model loaded!
echo.

REM ============================================================
REM Launch SC Gen 6
REM ============================================================
echo Starting application...
echo.

%PY_CMD% launch.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error.
    pause
)
