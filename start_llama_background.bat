@echo off
REM Start llama.cpp server persistently in background
REM This file auto-starts gpt-oss-20b with MASSIVE context and PARALLEL processing

REM Set Vulkan fix for AMD
set GGML_VK_FORCE_MAX_ALLOCATION_SIZE=4294967295

REM Start llama-server with proper WindowStyle to keep it running
REM KEY: --parallel 8 enables 8 concurrent request slots for true parallel processing
powershell -Command "Start-Process -FilePath '%~dp0llama-cpp\llama-server.exe' -ArgumentList '--model \"C:/Users/James/.lmstudio/models/lmstudio-community/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf\" --host 127.0.0.1 --port 8000 --ctx-size 131072 --n-gpu-layers 80 --parallel 8 --batch-size 512 --ubatch-size 512 --cache-type-k q8_0 --cache-type-v q8_0 --cont-batching' -WindowStyle Minimized"

echo llama.cpp server started in background
echo Server running at http://127.0.0.1:8000
echo.
echo Configuration:
echo   - Context: 131K tokens
echo   - Parallel slots: 8 (enables concurrent batch processing)
echo   - Continuous batching: ENABLED
echo.
echo Server window minimized. Check Task Manager to verify.
echo To stop: taskkill /IM llama-server.exe /F
