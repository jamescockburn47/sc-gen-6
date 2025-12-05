# llama.cpp Server Auto-Start Setup

## Overview

SC Gen 6 requires a running llama.cpp server with the Qwen 2.5 72B model. The enhanced `run.bat` now automatically handles server startup and ensures the model is loaded before launching the application.

## Quick Start

Just double-click **`run.bat`** - it will:
1. ✅ Check if llama-server.exe is already running
2. ✅ Start the server automatically if not running
3. ✅ Wait for the model to fully load (can take 30-60 seconds)
4. ✅ Launch SC Gen 6 once the server is ready

## Server Status

### Check if llama-server is running:
```powershell
tasklist | findstr llama-server
```

### Check server health:
```powershell
venv\Scripts\python.exe scripts\check_llama_server.py
```

### Manual server control:
```powershell
# Start server manually
.\start_llama_background.bat

# Stop server
taskkill /IM llama-server.exe /F
```

## Windows Auto-Start (Optional)

To have llama.cpp server start automatically when you log in to Windows:

### Option 1: Using Windows Startup Folder (Recommended)
1. Press `Win + R` and type: `shell:startup`
2. Create a shortcut to `autostart_llama_server.bat` in that folder
3. Right-click the shortcut → Properties → Run: **Minimized**

### Option 2: Task Scheduler (More Control)
1. Open Task Scheduler (search in Start menu)
2. Create Basic Task:
   - Name: "llama.cpp Server"
   - Trigger: "When I log on"
   - Action: "Start a program"
   - Program: `C:\Users\James\Desktop\SC Gen 6\autostart_llama_server.bat`
   - Check: "Run with highest privileges"

## Configuration

Server settings are in: **`config\llm_runtime.json`**

Current configuration:
- **Model**: Qwen 2.5 72B Instruct (Q4_K_M)
- **Endpoint**: http://127.0.0.1:8000
- **Context**: 32,768 tokens
- **GPU Layers**: 8 (optimized for 4GB VRAM)
- **Parallel Slots**: 8
- **Batch Size**: 512

> **Note**: GPU layers set to 8 due to AMD Radeon 8060S having 4GB VRAM. Using hybrid CPU+GPU processing for stability.

## Troubleshooting

### Server won't start
1. Check if executable exists: `llama-cpp\llama-server.exe`
2. Verify model path in `config\llm_runtime.json`
3. Check GPU drivers are up to date (AMD Radeon)

### Model loading is slow
- **Normal**: First load takes 30-60 seconds for 72B model
- The model is ~42 GB and needs to load into GPU memory
- Subsequent requests are fast once loaded

### Connection refused errors
- Run: `venv\Scripts\python.exe scripts\check_llama_server.py`
- Wait for server to fully load the model
- Check Windows Firewall isn't blocking port 8000

### Check server logs
- Logs are in: `logs\llama_server.log` (if using `scripts\start_llama_logged.py`)
- Current setup uses `--log-disable` for performance

## Files

- **`run.bat`** - Main launcher with auto-start logic
- **`start_llama_background.bat`** - Starts server in background
- **`autostart_llama_server.bat`** - Windows startup script
- **`scripts\check_llama_server.py`** - Health check utility
- **`config\llm_runtime.json`** - LLM configuration

## Model Loading Time

Typical loading times on your AMD Radeon 8060S (4GB VRAM, 8 GPU layers):
- **Cold start**: 10-15 seconds (hybrid CPU/GPU)
- **Warm start**: Immediate (if server already running)

The health check script waits up to 60 seconds and shows progress updates.
