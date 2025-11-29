# GMKTEC EVO X2 Optimization Guide

## Hardware Overview

The GMKTEC EVO X2 features the **AMD Ryzen AI Max+ 395 (Strix Halo)** APU with:
- 16 cores / 32 threads (Zen 5 architecture)
- **96GB unified VRAM** (configurable in BIOS)
- Radeon 8060S iGPU (40 RDNA 3.5 CUs, gfx1151)
- No PCIe bandwidth bottleneck (unified memory architecture)

This makes it ideal for running 70B+ parameter LLMs locally with excellent performance.

---

## Quick Start

### 1. Use the Optimized Launcher

Double-click `launch-96gb.bat` to start SC Gen 6 with optimized settings:
- Embedding batch size: 512 (16x default)
- Rerank batch size: 256 (32x default)
- Retrieval pool: 200+200 candidates
- Max workers: 12

### 2. Or Set Environment Variable

For permanent optimization:
```batch
set SC_CONFIG=96gb
```

Add to Windows Environment Variables for persistence.

---

## Recommended GGUF Models

### Primary Recommendation: Qwen2.5-72B-Instruct

| Quantization | VRAM Required | Quality | Speed | Recommendation |
|--------------|---------------|---------|-------|----------------|
| Q4_K_M | ~42 GB | Good | Fast | Best balance |
| Q5_K_M | ~52 GB | Better | Good | Recommended |
| Q6_K | ~62 GB | Excellent | Moderate | Quality focus |
| Q8_0 | ~80 GB | Near-FP16 | Slower | Maximum quality |

**Download from HuggingFace:**
- [Qwen2.5-72B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF)
- [bartowski/Qwen2.5-72B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-72B-Instruct-GGUF)

### Alternative: Llama-3.3-70B-Instruct

| Quantization | VRAM Required | Notes |
|--------------|---------------|-------|
| Q4_K_M | ~40 GB | Excellent reasoning, fast |
| Q5_K_M | ~50 GB | Better quality |
| Q6_K | ~60 GB | Near-original quality |

**Download:**
- [bartowski/Llama-3.3-70B-Instruct-GGUF](https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF)

### For Maximum Speed: Qwen2.5-32B-Instruct

| Quantization | VRAM Required | Notes |
|--------------|---------------|-------|
| Q6_K | ~26 GB | Excellent quality, very fast |
| Q8_0 | ~35 GB | Near-original quality |
| FP16 | ~64 GB | Full precision |

---

## Optimal llama.cpp Settings for 96GB

Update `config/llm_runtime.json`:

```json
{
  "provider": "llama_cpp",
  "base_url": "http://127.0.0.1:8000/v1",
  "api_key": "local-llama",
  "model_name": "qwen2.5-72b-instruct-q5_k_m",
  "llama_server": {
    "executable": "C:/dev/llama.cpp/build/bin/Release/llama-server.exe",
    "model_path": "PATH_TO_YOUR_MODEL.gguf",
    "context": 65536,
    "gpu_layers": 999,
    "parallel": 4,
    "batch": 4096,
    "ubatch": 512,
    "timeout": 3600,
    "host": "127.0.0.1",
    "port": 8000,
    "flash_attn": true,
    "extra_args": "--cache-type-k q8_0 --cache-type-v q8_0 --cont-batching --slots 4"
  }
}
```

### Key Settings Explained

| Setting | Value | Purpose |
|---------|-------|---------|
| `context` | 65536 | Large context for legal documents |
| `gpu_layers` | 999 | Offload ALL layers to GPU |
| `parallel` | 4 | Concurrent request handling |
| `batch` | 4096 | Fast prompt processing |
| `ubatch` | 512 | Efficient token generation |
| `flash_attn` | true | 20-40% faster inference |
| `cache-type-k/v q8_0` | - | KV cache quantization saves ~50% VRAM |

---

## ROCm / llama.cpp Setup (Recommended for Best Performance)

### Why ROCm?

| Backend | Prompt Processing | Generation | Notes |
|---------|-------------------|------------|-------|
| **ROCm 6.4.4** | **~1100 tok/s** | ~40 tok/s | Best for RAG workloads |
| Vulkan | ~400 tok/s | ~50 tok/s | More stable, slower prompts |

For RAG with large prompts (10-30K tokens), ROCm saves 30+ seconds per query.

### Step 1: Install AMD HIP SDK

1. Download from: https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html
2. Run `Setup.exe` as Administrator
3. Select **all components** (especially ROCm Libraries, HIP SDK, rocWMMA)
4. Restart your computer

**Note:** gfx1151 (Strix Halo) is officially supported as of ROCm 6.4+

### Step 2: Build llama.cpp with ROCm

Run the provided script:
```batch
scripts\setup_rocm_llama.bat
```

Or manually:
```batch
cd C:\dev\llama.cpp
mkdir build-rocm && cd build-rocm

cmake .. ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_HIP=ON ^
    -DGGML_HIP_ROCWMMA_FATTN=ON ^
    -DAMDGPU_TARGETS="gfx1151"

cmake --build . --config Release -j
```

### Step 3: Update SC Gen 6 Config

Edit `config/llm_runtime.json`:
```json
{
  "llama_server": {
    "executable": "C:/dev/llama.cpp/build-rocm/bin/Release/llama-server.exe",
    "flash_attn": true,
    "batch": 4096
  }
}
```

### BIOS VRAM Allocation

1. Enter BIOS (F2 or DEL on boot)
2. Navigate to Advanced > AMD CBS > NBIO Common Options
3. Set "UMA Frame Buffer Size" to 96GB
4. Save and exit

Reference: [GMKTEC BIOS VRAM Guide](https://www.gmktec.com/pages/evo-x2-bios-vram-size-adjustment-guide)

---

## Performance Expectations

With optimized settings on GMKTEC EVO X2:

| Model | Context | Prompt Speed | Generation Speed |
|-------|---------|--------------|------------------|
| Qwen2.5-72B Q4_K_M | 32K | ~100-150 tok/s | ~15-20 tok/s |
| Qwen2.5-72B Q5_K_M | 32K | ~80-120 tok/s | ~12-16 tok/s |
| Llama-3.3-70B Q4_K_M | 32K | ~100-150 tok/s | ~15-20 tok/s |
| Qwen2.5-32B Q6_K | 32K | ~200-300 tok/s | ~25-35 tok/s |

*Actual speeds depend on prompt length, generation length, and system configuration.*

---

## Troubleshooting

### Reranker Running on CPU

If the reranker is slow, verify GPU detection:
1. Open Settings > System Diagnostics
2. Check "GPU Detection" and "GPU Compute Test"
3. Ensure PyTorch reports ROCm/CUDA availability

The reranker should now automatically use GPU with float16 precision.

### Model Not Loading

1. Check VRAM allocation in BIOS (should be 96GB)
2. Verify model file exists and path is correct
3. Try a smaller quantization (Q4_K_M) first

### Slow Inference

1. Enable flash attention: `"flash_attn": true`
2. Use KV cache quantization: `--cache-type-k q8_0 --cache-type-v q8_0`
3. Reduce context if not needed: `"context": 32768`

---

## Sources

- [GMKTEC EVO-X2 Official Page](https://www.gmktec.com/products/amd-ryzen™-ai-max-395-evo-x2-ai-mini-pc)
- [Strix Halo Wiki](https://strixhalo.wiki/Hardware/PCs/GMKtec-EVO-X2)
- [ROCm Strix Halo Guide](https://github.com/Shoresh613/rocm-strix-halo)
- [llama.cpp ROCm Performance Discussion](https://github.com/ggml-org/llama.cpp/discussions/15021)
- [AMD Strix Halo Backend Benchmarks](https://kyuz0.github.io/amd-strix-halo-toolboxes/)
- [PyTorch ROCm Wheels](https://github.com/scottt/rocm-TheRock/releases)
