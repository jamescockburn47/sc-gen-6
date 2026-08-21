@echo off
REM Start llama-embed-nemotron-8b embedding server
REM This runs separately from the main LLM server on port 8090

set MODEL_PATH=models\embeddings\llama-embed-nemotron-8b-Q4_K_M.gguf
set PORT=8090

REM Check if model exists
if not exist "%MODEL_PATH%" (
    echo Model not found at %MODEL_PATH%
    echo.
    echo Download from: https://huggingface.co/sabafallah/llama-embed-nemotron-8b-GGUF
    echo.
    echo Run this command to download:
    echo huggingface-cli download sabafallah/llama-embed-nemotron-8b-GGUF llama-embed-nemotron-8b-Q4_K_M.gguf --local-dir models\embeddings
    echo.
    pause
    exit /b 1
)

echo Starting embedding server on port %PORT%...
echo Model: %MODEL_PATH%
echo.

REM Start llama.cpp server in embedding mode
llama-server.exe -m "%MODEL_PATH%" --embedding --port %PORT% -ngl 99

pause
