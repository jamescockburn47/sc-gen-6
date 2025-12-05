@echo off
REM ============================================================
REM Auto-start llama.cpp server on Windows login
REM ============================================================
REM To enable auto-start:
REM 1. Press Win+R and type: shell:startup
REM 2. Create a shortcut to this .bat file in that folder
REM ============================================================

cd /d "%~dp0"

REM Check if already running
tasklist /FI "IMAGENAME eq llama-server.exe" 2>NUL | find /I /N "llama-server.exe">NUL
if "%ERRORLEVEL%"=="0" (
    exit /b 0
)

REM Set Vulkan fix for AMD GPUs
set GGML_VK_FORCE_MAX_ALLOCATION_SIZE=4294967295

REM Start llama-server with proper persistence
powershell -WindowStyle Hidden -Command "Start-Process -FilePath '%~dp0llama-cpp\llama-server.exe' -ArgumentList '--model \"C:/Users/James/.lmstudio/models/lmstudio-community/Qwen2.5-72B-Instruct-GGUF/Qwen2.5-72B-Instruct-Q4_K_M.gguf\" --host 127.0.0.1 --port 8000 --ctx-size 32768 --n-gpu-layers 80 --parallel 8 --batch-size 512 --cache-type-k q8_0 --cache-type-v q8_0' -WindowStyle Minimized"

exit /b 0
