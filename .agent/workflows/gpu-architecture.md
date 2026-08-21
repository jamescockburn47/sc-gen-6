---
description: GPU architecture rules for AMD GPU with DirectML/ONNX
---

# GPU Architecture Rules

SC Gen 6 uses **DirectML/ONNX** for all ML inference on AMD GPUs. This workflow documents mandatory rules.

## Critical Rules

1. **All ONNX models must use DirectML** - Never add CPU fallback
2. **Use batched inference** - No sentence-by-sentence processing
3. **ONNX export uses opset 14** - Required for DirectML compatibility
4. **Models cached at `data/onnx_models/`** - One-time export

## GPU Components

| Component | File | Backend |
|-----------|------|---------|
| LegalBERT | `src/ingestion/chunkers/legalbert_onnx.py` | ONNX+DirectML |
| BGE Embeddings | `src/retrieval/embedding_service_onnx.py` | ONNX+DirectML |
| Reranker | `src/retrieval/reranker.py` | ONNX+DirectML |
| LLM | External llama-server | Vulkan/ROCm |

## When Adding New Models

// turbo
1. Check if ONNX model exists at `data/onnx_models/`
2. If not, export using `torch.onnx.export()` with opset 14
3. Use `use_safetensors=True` when loading HuggingFace models
4. Create ONNX session with `DmlExecutionProvider`
5. Implement batched inference (batch_size=32 default)
6. Add GPU monitoring with `src/system/gpu_monitor.log_performance()`

## DO NOT

- Use `CPUExecutionProvider` as fallback for GPU failures
- Process sentences one-by-one (always batch)
- Use opset > 14 (DirectML compatibility issues)
- Skip GPU memory monitoring during development

## Testing GPU Inference

// turbo
```powershell
.\venv\Scripts\python.exe -c "from src.ingestion.chunkers.legalbert_onnx import get_legalbert_encoder; e = get_legalbert_encoder(); print('GPU:', e.is_gpu_available())"
```

## Reference

See ARCHITECTURE.md section "GPU Utilization" for full documentation.
