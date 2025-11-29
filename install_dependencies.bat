@echo off
title SC Gen 6 Dependency Installer
echo Setting up Virtual Environment...

:: Force Python 3.10 for torch-directml support
set PYTHON_CMD=py -V:Astral/CPython3.10.17
%PYTHON_CMD% --version >nul 2>nul
if %errorlevel% neq 0 (
    echo Python 3.10 not found via py launcher!
    echo Please install Python 3.10 to enable GPU acceleration for PyTorch.
    echo Falling back to default python...
    set PYTHON_CMD=python
)

echo Using Python: %PYTHON_CMD%

:: Create venv if it doesn't exist
if exist .venv (
    echo Removing old venv...
    rmdir /s /q .venv
)

echo Creating .venv...
%PYTHON_CMD% -m venv .venv

:: Activate venv
call .venv\Scripts\activate

echo Installing dependencies...
:: Upgrade pip first
python -m pip install --upgrade pip

:: Install dependencies
:: We explicitly install torch-directml here
pip install torch-directml
pip install -r requirements.txt
pip install onnxruntime-directml

echo.
echo Installation complete!
echo Installation complete!
