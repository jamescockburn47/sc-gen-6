# Local llama.cpp Server (Vulkan, Windows)

This guide covers building and running `llama-server` with the Vulkan backend on Windows, then pointing SC Gen 6 to it via environment variables.

## 1. Build or Download llama-server with Vulkan

1. Install prerequisites:
   - [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/downloads/)
   - [CMake](https://cmake.org/download/)
   - Latest AMD Radeon drivers (includes Vulkan runtime).
2. Clone llama.cpp and configure for Vulkan:
   ```powershell
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   mkdir build && cd build
   cmake -DLLAMA_CURL=ON -DGGML_VULKAN=1 ..
   cmake --build . --config Release --target llama-server
   ```
3. Copy the resulting `llama-server.exe` somewhere convenient, e.g. `C:\llama\bin\llama-server.exe`.

> **Note:** Vulkan builds require a GPU with Vulkan 1.2+ support and up-to-date drivers.

## 2. Launch via built-in UI (recommended)

1. Open the desktop app → `Settings` → `LLM Control` tab.
2. Pick provider `llama_cpp`, then browse to:
   - `llama-server.exe` (Vulkan build)
   - The GGUF model path
3. Adjust context/GPU-layer/parallel sliders as needed and click **Save Settings**. Use the Flash Attention toggle or the extra CLI args box if you need flags like `--flash-attn`, `--mmq`, or custom tensor-split values.
4. Click **Start llama.cpp**. The status line will show when the server is running, and all output is piped to `logs/llama_server.log` (shown live in the new “LLM Console” tab).
5. Hit **Test Connection** to verify. The UI writes everything to `config/llm_runtime.json`, so it persists.

> Prefer the UI so you can tweak parameters and restart without editing scripts. The controls also work for switching back to LM Studio (change provider + base URL, then Test Connection).

## 3. Manual launch (fallback)

If you want to keep using the batch file:

1. Set the GGUF model path (example uses GPT-OSS 20B):
   ```powershell
   setx LLAMA_MODEL_PATH "C:\models\gpt-oss-20b.Q4_K_M.gguf"
   ```
2. Edit `scripts\start_llama_server.bat` and set `LLAMA_SERVER_EXE` to your compiled `llama-server.exe`.
3. Run the script from a Developer PowerShell / CMD prompt:
   ```powershell
   scripts\start_llama_server.bat
   ```
   You should see logs indicating the server is listening on `http://127.0.0.1:8000`.

The script configures:
- 65,536 token context (`-c 65536`)
- Full GPU offload (`-ngl 999`)
- API key `local-llama` (change if desired)

## 4. Configure SC Gen 6

Create or update `.env` with:

```
LLM_PROVIDER=llama_cpp
LLM_BASE_URL=http://127.0.0.1:8000/v1
LLM_API_KEY=local-llama
LLM_MODEL_NAME=gpt-oss-20b
LLAMA_MODEL_PATH=C:\models\gpt-oss-20b.Q4_K_M.gguf
```

Restart the application (or `launch.py`) so the new environment variables are picked up.

To switch back to LM Studio, change `LLM_PROVIDER=lmstudio`, set the `LLM_BASE_URL` to `http://localhost:1234/v1`, and restart.

## 5. Smoke Test

Run the helper script to verify connectivity for either provider:

```powershell
python tools\test_llm_connection.py
```

Expected output:
- Provider + base URL info
- `Ping response: ready`

If the script fails, check:
- The provider server is running
- API key matches `LLM_API_KEY`
- No firewall is blocking `127.0.0.1:8000`

## 6. Troubleshooting

- **Vulkan validation errors**: update AMD drivers and ensure the Vulkan SDK/runtime is installed.
- **Context exceeded**: reduce `CTX` in the batch file or use a smaller model.
- **Slow throughput or KV spills**: reduce `--parallel` or trim the RAG prompt size. The in-app GPU monitor (header of the answer panel) shows VRAM usage/utilisation so you can see when you’re saturating GMTEK EVO X2’s memory.

With this setup, llama.cpp becomes the default local LLM provider, while LM Studio remains available as an alternate provider via `LLM_PROVIDER=lmstudio`.

