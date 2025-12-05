@echo off
echo Starting llama.cpp server...
echo.

:: CRITICAL: Fix AMD Vulkan allocation limit on Windows
set GGML_VK_FORCE_MAX_ALLOCATION_SIZE=4294967295

:: Configuration - gpt-oss-20b with MASSIVE context window
set EXECUTABLE=llama-cpp\llama-server.exe
set MODEL=C:\Users\James\.lmstudio\models\lmstudio-community\gpt-oss-20b-GGUF\gpt-oss-20b-MXFP4.gguf
set PORT=8000
set CONTEXT=131072
set GPU_LAYERS=80
set PARALLEL=8
set BATCH=512
set TIMEOUT=1800

:: Check if executable exists
if not exist "%EXECUTABLE%" (
    echo [ERROR] llama-server.exe not found at %EXECUTABLE%
    echo Please check the path.
    pause
    exit /b 1
)

:: Check if model exists
if not exist "%MODEL%" (
    echo [ERROR] Model file not found at %MODEL%
    echo Please check the path.
    pause
    exit /b 1
)

echo Launching server with:
echo   Model: %MODEL%
echo   Context: %CONTEXT% tokens (128K!)
echo   GPU Layers: %GPU_LAYERS%
echo   Parallel Streams: %PARALLEL%
echo.

"%EXECUTABLE%" -m "%MODEL%" -c %CONTEXT% -ngl %GPU_LAYERS% --host 127.0.0.1 --port %PORT% --parallel %PARALLEL% -b %BATCH% --timeout %TIMEOUT%

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server exited with error code %ERRORLEVEL%
    pause
)
