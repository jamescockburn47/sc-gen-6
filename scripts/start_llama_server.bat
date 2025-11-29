@echo off
rem This script assumes you built llama-server.exe with Vulkan support (cmake -DGGML_VULKAN=1).
rem Ensure the Vulkan runtime for your GPU is installed before launching.

setlocal

if "%LLAMA_MODEL_PATH%"=="" (
    echo Please set LLAMA_MODEL_PATH to your GGUF model file.
    echo Example: set LLAMA_MODEL_PATH=C:\models\gpt-oss-20b.Q4_K_M.gguf
    exit /b 1
)

set LLAMA_SERVER_EXE=C:\path\to\llama-server.exe
set PORT=8000
set CTX=65536
set N_GPU_LAYERS=999
set API_KEY=local-llama

if not exist "%LLAMA_SERVER_EXE%" (
    echo Could not find llama-server executable at "%LLAMA_SERVER_EXE%".
    echo Update LLAMA_SERVER_EXE in scripts\start_llama_server.bat to point to your build.
    exit /b 1
)

echo Starting llama.cpp server on port %PORT% with model %LLAMA_MODEL_PATH%
"%LLAMA_SERVER_EXE%" ^
    -m "%LLAMA_MODEL_PATH%" ^
    -c %CTX% ^
    -ngl %N_GPU_LAYERS% ^
    --host 127.0.0.1 ^
    --port %PORT% ^
    --api-key %API_KEY% ^
    --no-mmap ^
    --parallel 2 ^
    --timeout 1800 ^
    --system "Use citations exactly as provided."

endlocal














