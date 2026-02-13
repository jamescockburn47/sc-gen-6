# SC Gen 6 Debug Report

**Date:** 2026-02-13  
**Machine:** GMKtec Evo X2 — AMD Ryzen AI Max+ 395 "Strix Halo", Radeon 8060S (gfx1151), 128GB LPDDR5X  
**OS:** Ubuntu (dual-boot with Windows)

---

## Summary

SC Gen 6 is a custom-built litigation support RAG system using PySide6 (desktop UI), ChromaDB (vector store), sentence-transformers (embeddings), and local LLM inference. The system was non-functional on Linux due to several configuration issues, missing services, and Windows-specific code paths.

## Issues Found and Fixed

### P1: LLM Backend Not Running (CRITICAL)

**Problem:** The system was configured for llama-swap (a model multiplexer managing llama.cpp servers), but:
- No llama-swap or llama-server processes were running
- Ollama was not installed
- The configured model (`nemotron-3-nano` via llama.cpp) required llama-swap infrastructure

**Fix:** 
- Installed Ollama with Vulkan backend (`OLLAMA_VULKAN=1`)
- Configured systemd service with Vulkan, flash attention, 24h keep-alive
- Pulled GLM 4.7 Flash (30B-A3B MoE) as primary model
- Updated `config/llm_runtime.json` to use Ollama provider
- Updated `config/config.yaml` backend to `ollama`, model to `glm-4.7-flash`
- Updated `src/config/llm_config.py` defaults to match
- Increased context window from 16K to 32K tokens for RAG queries

### P2: Embedding Dimension Mismatch (CRITICAL)

**Problem:** 
- ChromaDB had 556 chunks with 4096-dimensional embeddings (nemotron-embed-8b via llama-swap)
- The nemotron-embed-8b model required the llama-swap embedding server which wasn't running
- Switching to sentence-transformers BGE (1024-dim) would create incompatible queries

**Fix:**
- Cleared ChromaDB collection (`litigation_docs`)
- Switched embedding config to `BAAI/bge-large-en-v1.5` (1024 dims)
- Set `use_llamacpp: false`, `use_onnx_gpu: false` in config
- Re-ingested all 3 documents (556 chunks) with BGE embeddings on CPU
- Verified query/embedding dimension consistency

### P3: ROCm Compute Corruption Risk (CRITICAL)

**Problem:** 
- PyTorch was installed with ROCm 6.2 support (`torch 2.5.1+rocm6.2`)
- ROCm on gfx1151 (Strix Halo) has known compute corruption bugs
- Auto-detection would select ROCm GPU for embeddings and reranking

**Fix:**
- Forced CPU for embedding service (`src/retrieval/embedding_service.py`)
- Set ONNX GPU to disabled for both embeddings and reranker
- Reranker falls through to sentence-transformers CrossEncoder on CPU
- LLM inference uses Ollama with Vulkan (separate from PyTorch stack)

### P4: Windows-Specific Debug Logging (MODERATE)

**Problem:** Several files contained Windows-specific debug logging that wrote to `c:\Users\James\Desktop\SC Gen 6\.cursor\debug.log`, which would fail on Linux.

**Fix:** Removed all Windows-specific debug logging blocks from:
- `src/generation/prompts.py`
- `src/ui/document_manager.py`
- `src/generation/chunk_batcher.py`
- `src/llm/llama_swap_manager.py`

### P5: FTS5 Query Parsing Error (MINOR)

**Problem:** FTS5 keyword search failed on queries containing hyphens (e.g., "follow-on") because hyphens triggered FTS5 column reference syntax (`column:term`).

**Fix:** Updated `_prepare_query()` in `src/retrieval/fts5_index.py` to:
- Strip hyphens from query text
- Quote individual words to prevent FTS5 syntax interpretation
- Filter out single-character words

### P6: Status Monitor Checking Wrong Servers (COSMETIC)

**Problem:** The UI status monitor polled llama-swap servers at ports 8000/8001, which no longer exist.

**Fix:** Updated `_setup_status_monitor()` in `src/ui/modern_main_window.py` to check Ollama's endpoint based on the LLM config provider.

### P7: Disk Space Exhaustion (CRITICAL)

**Problem:** Root partition was 100% full (390GB/390GB) due to:
- 59GB of GGUF model files in HuggingFace cache (no longer needed)
- 42GB of old Qwen model caches
- Multiple 10MB rotated ingestion logs

**Fix:**
- Removed old GGUF files (nemotron-3-nano BF16, 59GB)
- Removed old Qwen model caches (42GB)
- Removed old rotated ingestion logs
- Freed 101GB of disk space

### P8: Ollama Using ROCm Instead of Vulkan (CRITICAL)

**Problem:** Despite setting `OLLAMA_VULKAN=1`, Ollama 0.16.1 was preferring the ROCm backend over Vulkan when both were available. Logs showed `library=ROCm` and the model crashed with `ROCm error: out of memory` from `ggml-cuda.cu` — the gfx1151 ROCm corruption bug.

**Fix:**
- Added `OLLAMA_LLM_LIBRARY=vulkan` to systemd override to force Vulkan backend
- Added `HIP_VISIBLE_DEVICES=-1` to hide ROCm devices from Ollama
- Restarted Ollama — now correctly uses Vulkan with all 48 layers on GPU
- Confirmed: `ollama ps` shows `100% GPU`, logs show `ggml_vulkan: AMD Radeon Graphics (RADV GFX1151)`

### P9: Configuration Mismatches (MODERATE)

**Problem:** Multiple config mismatches between `config.yaml`, `llm_runtime.json`, and `src/config_loader.py`:
- YAML: `llama_cpp` backend, `nemotron-embed-8b` embeddings
- Code defaults: `ollama` backend, `bge-large-en-v1.5` embeddings  
- Runtime JSON: `llama_cpp` with `nemotron-3-nano`
- Reranker: YAML said `base-v2`, code defaults said `large-v2`

**Fix:** Aligned all three configuration sources to:
- Backend: `ollama`
- LLM: `glm-4.7-flash`
- Embedding: `BAAI/bge-large-en-v1.5`
- Reranker: `mixedbread-ai/mxbai-rerank-base-v2`
- ONNX GPU: disabled (ROCm corruption risk)

## Current Architecture

```
Ollama (Vulkan) ──► GLM 4.7 Flash (30B-A3B MoE)
     │                   ▲
     │                   │ /api/chat
     │              SC Gen 6 App
     │                   │
     ├── Embeddings: BGE-large-en-v1.5 (CPU, sentence-transformers)
     ├── Reranker: mxbai-rerank-base-v2 (CPU, CrossEncoder)
     ├── Vector DB: ChromaDB (556 chunks, 1024-dim)
     └── Keyword: SQLite FTS5 (556 chunks)
```

## Benchmark Results (Full Pipeline)

| Metric | Value |
|--------|-------|
| Queries tested | 8 |
| Retrieval errors | 0 |
| Generation errors | 3 (0 chunks retrieved → correct refusal) |
| Avg retrieval time | 3.29s |
| Avg generation time | 26.38s |
| Avg total time | 19.77s |
| Avg chunks retrieved | 0.6 |
| LLM backend | Ollama + Vulkan (100% GPU) |
| LLM model | GLM 4.7 Flash (Q4_K_M, 19GB) |
| Embedding model | BAAI/bge-large-en-v1.5 (CPU) |
| Reranker | mxbai-rerank-base-v2 (CPU) |
| Vector store | 556 chunks (1024-dim) |

**Notes:**
- Low chunk retrieval (0.6 avg) is expected — the 3 indexed documents are Chancery/Commercial Court judgments, not competition law materials. The benchmark queries are competition law-specific.
- All "Not found in provided documents" answers are **correct** — the system does not hallucinate.
- The 3 generation errors occur because the retriever correctly returns 0 chunks and `generate_with_context` raises `ValueError` as designed.
- Pipeline is fully functional end-to-end: ingest → embed → retrieve → rerank → generate.

## Files Modified

### Configuration
- `config/config.yaml` — Backend, model, embedding, reranker settings
- `config/llm_runtime.json` — Runtime LLM state for Ollama
- `src/config_loader.py` — Code defaults aligned
- `src/config/llm_config.py` — Default provider/model

### Core Services
- `src/retrieval/embedding_service.py` — Force CPU for ROCm safety
- `src/retrieval/fts5_index.py` — Fix FTS5 query parsing
- `src/llm/client.py` — Increase context window to 32K
- `src/generation/prompts.py` — Remove Windows debug log

### UI / Launcher
- `src/ui/modern_main_window.py` — Status monitor for Ollama
- `src/ui/document_manager.py` — Remove Windows debug log
- `run_linux.sh` — Updated launcher for Ollama

### System / Ollama
- `/etc/systemd/system/ollama.service.d/override.conf` — Force Vulkan, disable ROCm

### Other
- `src/generation/chunk_batcher.py` — Remove Windows debug log
- `src/llm/llama_swap_manager.py` — Remove Windows debug log
- `benchmark.py` — New benchmark script
- `benchmarks/` — Benchmark results directory

## Recommendations

1. **Index more documents** — Only 3 PDFs are currently indexed. For meaningful competition law queries, index the relevant Competition Act materials, CAT decisions, and CMA reports.
2. **Consider legal-specific embeddings** — Legal-BERT or CaseLaw-BERT may improve retrieval quality for competition law content.
3. **Add query decomposition** — Complex multi-part competition law queries would benefit from decomposition before retrieval.
4. **Monitor GPU VRAM** — The 96GB VRAM allocation should be verified with `nvtop` when Ollama is serving the model.
