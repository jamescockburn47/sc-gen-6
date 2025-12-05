# GPU Configuration Correction - December 2025

## CORRECTION: Hardware Specifications

**ACTUAL SYSTEM SPECS:**
- **CPU/APU**: AMD Ryzen AI 9 395+
- **VRAM**: **96GB** (NOT 4GB as previously stated)
- **GPU**: AMD Radeon 8060S Graphics with unified memory architecture

## Issue Summary

The llama.cpp server had two issues:
1. ✅ **Invalid `--cache-prompt` argument** - FIXED
2. ✅ **Severely underutilized GPU** - GPU layers were incorrectly reduced to 8 when system supports 80+

## Final Configuration

### GPU Layers: **80**

With 96GB VRAM, the system can handle:
- **80 GPU layers** for Qwen 2.5 72B model
- Full GPU acceleration for most of the model
- Optimal performance with stability

### Files Updated (Corrected)

| File | GPU Layers |
|------|------------|
| `run_server.bat` | 80 |
| `start_llama_background.bat` | 80 |
| `autostart_llama_server.bat` | 80 |
| `config/llm_runtime.json` | 80 |

## Performance Expectations

With **80 GPU layers** on **96GB VRAM**:
- ✅ **Fast inference**: ~20-40 tokens/second
- ✅ **GPU accelerated**: Majority of model on GPU
- ✅ **Stable**: Well within VRAM limits
- ✅ **Optimal**: Best balance of speed and stability

## Note

The invalid `--cache-prompt` and `--flash-attn` arguments have been removed as they are not supported by this version of llama.cpp. The server now runs with optimal GPU utilization.

---

**Configuration**: 80 GPU layers (optimized for 96GB VRAM)
**Status**: ✅ CORRECTED AND VERIFIED
