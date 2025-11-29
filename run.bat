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

echo Starting application...
echo.

%PY_CMD% launch.py
if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error.
    pause
)
