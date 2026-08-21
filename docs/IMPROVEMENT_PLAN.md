# SCGen6 Improvement Plan
*Generated: 2026-02-18 | Hardware: AMD Strix Halo (gfx1151 / Radeon 8060S) | 96GB unified VRAM*

---

## Executive Summary

This plan covers three areas:
1. **LLM Speed** — techniques from Alex Ziskind and current llama.cpp/Ollama research to increase tokens/sec for GLM 4.7 Flash and Nemotron 3 Nano without sacrificing accuracy.
2. **Accuracy** — RAG pipeline improvements to improve answer quality and citation fidelity.
3. **Codebase Quality** — bugs, dead code, and architectural improvements found during the review.

Current baseline: **GLM 4.7 Flash via Ollama/Vulkan** (~20–40 t/s generation). **Nemotron 3 Nano BF16 via llama.cpp/Vulkan** (~20 t/s generation, ~240 t/s prompt).

---

## Part 1 — LLM Speed Improvements

### 1.1 Speculative Decoding (Highest Impact — 20–50% speed gain)

**What it is:** A small "draft" model rapidly generates candidate tokens; the large target model verifies them in a single parallel pass. Because LLMs are memory-bandwidth-bound (not compute-bound), batching the verification step dramatically increases effective throughput.

**Research basis:** Alex Ziskind demonstrated 20–40% gains with LM Studio's speculative decoding (March 2025). llama.cpp users on Reddit report 1.5×–2× gains for 30B+ models, with some reaching 3–6× on memory-bound hardware like APUs.

#### For Nemotron 3 Nano (llama.cpp / Vulkan)

llama.cpp natively supports speculative decoding via `--model-draft` and `-ngld`.

**Recommended draft model:** `Nemotron-Mini-4B-Instruct-Q4_K_M.gguf` (~2.5GB) — same architecture family, high token acceptance rate.

**Updated `llama-swap/config.yaml`:**
```yaml
"nemotron-3-nano":
  cmd: |
    ${llama_server}
    --model /home/james/.cache/huggingface/gguf/Nemotron-3-Nano-30B-A3B-BF16-00001-of-00002.gguf
    --model-draft /home/james/.cache/huggingface/gguf/Nemotron-Mini-4B-Q4_K_M.gguf
    --ctx-size 32768
    --n-gpu-layers 99
    --draft-max 8
    --draft-min 1
    --draft-p-min 0.6
    --host 127.0.0.1
    --port ${PORT}
    --batch-size 2048
    --ubatch-size 512
    --flash-attn
    --cache-type-k q4_0
    --cache-type-v q4_0
    --timeout 300
    --metrics
```

Key flags:
- `--draft-max 8` — draft model generates up to 8 speculative tokens per step (sweet spot for 30B models)
- `--draft-p-min 0.6` — only speculate when draft is ≥60% confident (avoids wasted work)
- `--flash-attn` — **critical** for Vulkan; merged into llama.cpp May 2025, doubles effective context at Q8 KV
- `--cache-type-k q4_0 --cache-type-v q4_0` — KV cache quantization (already in `llm_runtime_bf16.json` but missing from `config.yaml`)

**Expected gain:** 20–40% on generation speed (from ~20 t/s to ~25–28 t/s).

#### For GLM 4.7 Flash (Ollama)

Ollama does not natively expose speculative decoding via its API. Options:
- **Switch GLM 4.7 Flash to llama.cpp directly** (via llama-swap) and use `--model-draft` with `GLM-4-9B-Q4_K_M.gguf` as draft. This is the cleanest path.
- **Or** use Ollama's `num_predict` and `num_ctx` tuning (see §1.3).

### 1.2 KV Cache Quantization (Free Speed + Memory)

**Current state:** `llm_runtime_bf16.json` has `--cache-type-k q4_0 --cache-type-v q4_0` but `llama-swap/config.yaml` does NOT include these flags. This is an inconsistency — the Nemotron config in llama-swap is missing KV cache quantization.

**Fix:** Add to `llama-swap/config.yaml` Nemotron entry:
```
--cache-type-k q4_0
--cache-type-v q4_0
```

**Impact:** ~30–40% reduction in KV cache VRAM. For a 32K context with a 30B model, this frees ~8–12GB, allowing either:
- Larger context (64K–128K) at same VRAM
- More parallel slots (increase `--parallel` from 4 to 6–8)

GLM 4.7 Flash uses MLA (Multi-Latent Attention) which already compresses KV cache by ~73% — this is one reason it's faster than Nemotron at the same parameter count. No additional KV quantization needed for GLM via Ollama.

### 1.3 Ollama Tuning for GLM 4.7 Flash

The current `run_linux.sh` and `llm_runtime.json` use default Ollama settings. Add these to the Ollama Modelfile or systemd environment:

```
OLLAMA_NUM_PARALLEL=4          # Allow 4 concurrent requests (matches Ollama's default)
OLLAMA_MAX_LOADED_MODELS=1     # Keep only GLM loaded (prevents model eviction)
OLLAMA_FLASH_ATTENTION=1       # Enable Flash Attention (significant speed boost)
OLLAMA_KV_CACHE_TYPE=q8_0      # Quantize KV cache (saves ~50% KV VRAM vs f16)
```

**Add to `/etc/systemd/system/ollama.service.d/override.conf`:**
```ini
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
```

**Expected gain:** 10–20% speed improvement from Flash Attention alone.

### 1.4 Context Window Right-Sizing

**Current issue:** `start_llama_server.sh` uses `--ctx-size 32768` but the `llm_service.py` defaults to 32768 and `llm_runtime.json` comment says "GLM 4.7 Flash supports 198K context, using 32K for RAG."

For RAG workloads with 20 chunks × ~768 chars = ~15K tokens of context, a 32K context is appropriate. However:

- **Nemotron via llama.cpp:** 32K is correct given 63GB weight + KV overhead.
- **GLM via Ollama:** Could safely use 64K context since MLA compresses KV by 73%. This allows more chunks to be sent to the LLM without batching, potentially improving answer quality.

**Recommendation:** Increase GLM context to 65536 in `llm_runtime.json` and update `config.yaml` `context_to_llm` from 20 to 25 chunks.

### 1.5 Flash Attention for llama.cpp (Vulkan)

**Status:** `--flash-attn` is present in `llm_runtime_bf16.json` but NOT in `start_llama_server.sh` or `llama-swap/config.yaml`.

**Fix:** Add `--flash-attn` to both `start_llama_server.sh` and `llama-swap/config.yaml`. This is a free speed improvement (10–15% on prompt processing) and enables Q8 KV cache doubling of effective context.

### 1.6 Continuous Batching (`--cont-batching`)

**Status:** Present in `llm_runtime_bf16.json` extra_args but missing from `llama-swap/config.yaml`.

**Fix:** Add `--cont-batching` to the llama-swap Nemotron config. This allows the server to interleave multiple requests efficiently, critical when `--parallel 4` is set.

### 1.7 Thread Count Optimization

**Current:** No `--threads` flag in `llama-swap/config.yaml` (uses llama.cpp default).

**Recommendation:** Strix Halo has a 16-core CPU. For Vulkan-offloaded models (all layers on GPU), CPU threads are only used for sampling and non-GPU ops. Set `--threads 8` to avoid over-subscribing the CPU while leaving headroom for the OS and embedding service.

---

## Part 2 — Accuracy Improvements

### 2.1 Nemotron 3 Nano — Missing Thinking Mode Support

**Issue:** `llm_service.py` `_build_thinking_kwargs()` handles Qwen3 and DeepSeek-R1 but NOT Nemotron 3 Nano, which supports a `reasoning_content` thinking mode via llama.cpp.

**Fix:** Add Nemotron to `thinking_models` in `config.yaml`:
```yaml
thinking_models:
  - qwen3
  - deepseek-r1
  - nemotron-3-nano
  - nemotron3
  - o1
  - claude-3.5
  - gemini-2.0
```

And add handling in `_build_thinking_kwargs()`:
```python
elif "nemotron" in model_lower:
    # Nemotron 3 Nano supports reasoning via budget_tokens
    kwargs["extra_body"] = {
        "thinking": {"type": "enabled", "budget_tokens": self.settings.models.llm.thinking_budget}
    }
```

**Impact:** Nemotron's thinking mode significantly improves complex legal reasoning accuracy.

### 2.2 GLM 4.7 Flash — Thinking Mode Not Configured

**Issue:** GLM 4.7 Flash (the current default model) is NOT in `thinking_models`. GLM 4.7 Flash supports a `/think` suffix or `enable_thinking` parameter via Ollama.

**Fix:** Add `glm` to `thinking_models` and handle in `_build_thinking_kwargs()`:
```python
elif "glm" in model_lower:
    kwargs["extra_body"] = {
        "enable_thinking": True,
    }
```

**Note:** GLM thinking mode adds latency (~2–5s) but dramatically improves multi-step legal reasoning. Consider making this opt-in per query mode (e.g., only in "Deep Analysis" mode).

### 2.3 Prompt Engineering — Remove Debug Print Statements

**Issue:** `llm_service.py` has multiple `print()` debug statements in production code paths:
- Line 512: `[POST_PROCESS DEBUG]`
- Line 517: `[POST_PROCESS DEBUG]`
- Line 521: `[POST_PROCESS DEBUG]`
- Lines 531, 542, 553, 560, 565, 569: `[CHANNEL DEBUG]`

These are noise in production logs and slow down the hot path slightly. Replace with `logging.debug()` calls.

### 2.4 Chunk Batcher — Debug Prints in Production

**Issue:** `chunk_batcher.py` has extensive `[BATCH DEBUG]` and `[SYNTHESIS DEBUG]` print statements throughout the hot generation path (lines 169, 183, 190, 195, 197, etc.).

**Fix:** Convert all to `logging.debug()`. This is particularly important because the batcher runs in a `ThreadPoolExecutor` — print() calls from multiple threads interleave and corrupt log output.

### 2.5 Retrieval — MMR Re-embeds Query (Wasted Work)

**Issue:** `hybrid_retriever.py` `_apply_mmr()` (line 760) calls `self.embedding_service.embed_query(query)` again, even though the query was already embedded at line 190 for the semantic search. This is a duplicate embedding call — wasted latency (~50–200ms on CPU).

**Fix:** Cache the query embedding and pass it to `_apply_mmr()`:
```python
# In retrieve():
query_embedding = self.embedding_service.embed_query(query)
# ... later ...
results = self._apply_mmr(query=query, query_embedding=query_embedding, results=results, ...)

# In _apply_mmr():
def _apply_mmr(self, query, results, top_k, lambda_mult=0.5, query_embedding=None):
    query_emb = query_embedding or self.embedding_service.embed_query(query)
```

### 2.6 Retrieval — `_get_chunk_data` Opens New DB Connection Each Call

**Issue:** `_get_chunk_data()` (line 634) calls `self.keyword_index._get_conn()` which may open a new SQLite connection on each retrieval. SQLite connections are cheap but not free. For high-frequency queries this adds up.

**Fix:** Cache the connection in `HybridRetriever.__init__` or use the existing connection pool in `FTS5IndexCompat`.

### 2.7 Status Monitor — Excessive Debug Logging

**Issue:** `status_monitor.py` lines 121 and 136 print `[StatusMonitor]` debug messages on EVERY poll (every 2 seconds). This floods the log with ~30 lines/minute of noise.

**Fix:** Remove or convert to `logging.debug()`.

### 2.8 Confidence Threshold — Too Low for Legal Use

**Current:** `confidence_threshold: 0.12` in `config.yaml`.

**Issue:** A threshold of 0.12 is very permissive. For a legal RAG system where hallucinations are professionally dangerous, this risks including weakly-relevant chunks. The `MIN_GUARANTEED = 3` logic in `hybrid_retriever.py` already ensures at least 3 chunks are returned even if scores are low.

**Recommendation:** Raise to `0.20` and rely on `MIN_GUARANTEED` for the safety net. This will improve answer precision at the cost of occasionally returning fewer chunks for obscure queries.

### 2.9 Chunking — Semantic Chunker Not Used

**Issue:** `config.yaml` sets `strategy: semantic` but the semantic chunker requires LegalBERT embeddings during ingestion. Given the current CPU-only embedding setup (sentence-transformers BGE), semantic chunking adds significant ingestion latency.

**Recommendation:** Either:
- Switch to `strategy: robust` (the `RobustChunker` / `AdaptiveChunker`) for faster, more predictable ingestion.
- Or keep `semantic` but ensure the embedding server is running before ingestion starts (the status monitor already handles this interlock).

### 2.10 Context Window Utilisation — Prompt Token Estimation

**Issue:** `estimate_token_count()` in `llm_service.py` uses `len(text) // 4` (4 chars/token). For legal documents with many short words, legal citations, and numbers, the actual ratio is closer to 3–3.5 chars/token. This means the prompt size check at line 461 is **underestimating** token counts by 15–25%, potentially sending prompts that exceed the actual context window.

**Fix:** Use a more conservative estimate of 3.5 chars/token, or better, use `tiktoken` (for OpenAI-compatible models) or the llama.cpp `/tokenize` endpoint for exact counts:
```python
def estimate_token_count(text: str) -> int:
    # More conservative: 3.5 chars/token for legal/mixed content
    return max(1, int(len(text) / 3.5))
```

---

## Part 3 — Codebase Quality

### 3.1 Windows Paths in Linux Config Files

**Issue:** `config/llm_runtime_bf16.json` still contains Windows paths:
```json
"executable": "C:/Users/James/Desktop/SC Gen 6/llama-cpp/llama-server.exe"
"model_path": "C:/Users/James/Desktop/SC Gen 6/models/..."
```

This file is dead on Linux. Either:
- Delete it and create `config/llm_runtime_nemotron_linux.json` with correct Linux paths.
- Or add a comment that it's Windows-only and create a Linux equivalent.

### 3.2 Multiple Download Scripts — Cleanup

The root directory contains 6 Nemotron download scripts:
- `download_nemotron_bf16.py`
- `download_nemotron_bf16_fast.py`
- `download_nemotron_bf16_throttled.py`
- `download_nemotron_bf16_v2.py`
- `download_nemotron_q8.py`
- `download_model.py`

These are one-time utility scripts. Move to `tools/` or `scripts/` and keep only the best version.

### 3.3 `.bat` Files on Linux

The root contains Windows batch files (`run.bat`, `run_server.bat`, `run_embed_server.bat`, `autostart_llama_server.bat`, `start_llama_background.bat`, `install_dependencies.bat`, `run_timeline_extraction.bat`). These are dead code on Linux. Move to `archive/windows/` or add a `.gitignore` exclusion.

### 3.4 `llm_service.py` — `generate()` Has Unused `stream` Parameter

**Issue:** `generate()` (line 164) accepts a `stream: bool = False` parameter but ignores it entirely (line 167 docstring says "stream: Whether to stream (ignored in non-streaming mode)"). This is confusing API design. Remove the parameter or raise `NotImplementedError` if `stream=True` is passed.

### 3.5 `chunk_batcher.py` — `_run_single_batch` Uses `stream=False`

**Issue:** Line 380 calls `generate_with_context(..., stream=False, ...)`. For parallel batch generation, streaming would actually be beneficial — it would allow the Ollama/llama.cpp server to start serving the next batch sooner. However, the current architecture collects full responses before synthesis, so this is a design trade-off, not a bug.

**Recommendation:** For GLM 4.7 Flash with Ollama, consider enabling streaming per batch and collecting tokens in a thread-safe buffer. This would reduce perceived latency.

### 3.6 `status_monitor.py` — Singleton Pattern is Fragile

**Issue:** `get_status_monitor()` uses a module-level `_monitor` global. If the URLs change (e.g., switching from llama-swap to direct llama-server), the singleton retains the old URLs. Add a `reset_monitor()` function or make the singleton URL-keyed.

### 3.7 `config_loader.py` — `LLMConfig.backend` Default Mismatch

**Issue:** `LLMConfig.backend` defaults to `"ollama"` in `config_loader.py` (line 53), but `config/llm_runtime.json` is the actual runtime source of truth. The config.yaml `backend: ollama` is correct, but the Pydantic default could mislead developers who don't read the runtime JSON.

**Recommendation:** Add a docstring comment to `LLMConfig` clarifying that `backend` is overridden by `config/llm_runtime.json` at runtime.

### 3.8 Missing `--ubatch-size` in `start_llama_server.sh`

**Issue:** `start_llama_server.sh` uses `--batch-size 2048` but does NOT set `--ubatch-size`. The physical batch size (`ubatch-size`) defaults to 512 in llama.cpp, which is fine, but `llm_runtime_bf16.json` explicitly sets `--ubatch-size 512`. Make this explicit in `start_llama_server.sh` for clarity and to prevent future llama.cpp default changes from silently affecting performance.

### 3.9 `hybrid_retriever.py` — `reranked` Variable Undefined in Stats Block

**Issue:** When `skip_reranking=True`, the stats block at line 491 references `reranked` (line 494: `"reranked_count": len(reranked)`). In the skip-reranking path, `reranked` is set to a fake list at line 351. This works but is confusing. Rename to `_reranked_for_stats` to make the intent clear.

### 3.10 `run_linux.sh` — Checks Wrong Systemd Config

**Issue:** Line 40 checks `sudo systemctl cat ollama | grep "OLLAMA_VULKAN=1"`. This requires `sudo` just to check a config, which will prompt for a password in non-interactive mode. Use `systemctl show ollama --property=Environment` instead (no sudo needed).

---

## Part 4 — Priority Implementation Order

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| 🔴 HIGH | 1.5 Add `--flash-attn` to llama-swap config | 5 min | +10–15% speed |
| 🔴 HIGH | 1.2 Add KV cache quantization to llama-swap config | 5 min | +30% VRAM savings |
| 🔴 HIGH | 1.6 Add `--cont-batching` to llama-swap config | 5 min | Better parallelism |
| 🔴 HIGH | 2.3 Remove debug prints from `llm_service.py` | 30 min | Cleaner logs |
| 🔴 HIGH | 2.4 Remove debug prints from `chunk_batcher.py` | 30 min | Cleaner logs |
| 🟡 MED | 1.1 Speculative decoding (download draft model + config) | 2 hrs | +20–40% speed |
| 🟡 MED | 1.3 Ollama Flash Attention + KV cache env vars | 15 min | +10–20% speed |
| 🟡 MED | 2.1 Add Nemotron thinking mode support | 1 hr | Better accuracy |
| 🟡 MED | 2.5 Fix duplicate query embedding in MMR | 30 min | -50–200ms latency |
| 🟡 MED | 2.10 Fix token count estimation (3.5 chars/token) | 15 min | Prevent context overflow |
| 🟢 LOW | 2.7 Remove status monitor debug prints | 15 min | Cleaner logs |
| 🟢 LOW | 3.1 Fix Windows paths in llm_runtime_bf16.json | 10 min | Cleanliness |
| 🟢 LOW | 3.2 Consolidate download scripts | 30 min | Cleanliness |
| 🟢 LOW | 3.8 Add `--ubatch-size` to start_llama_server.sh | 5 min | Explicitness |
| 🟢 LOW | 3.10 Fix sudo in run_linux.sh | 5 min | UX |

---

## Part 5 — Draft Model Download (for Speculative Decoding)

For Nemotron speculative decoding, download a small Nemotron-family draft model:

```bash
# Option A: Nemotron-Mini-4B (best acceptance rate for Nemotron 3 Nano)
huggingface-cli download \
  bartowski/nvidia_Nemotron-Mini-4B-Instruct-GGUF \
  --include "Nemotron-Mini-4B-Instruct-Q4_K_M.gguf" \
  --local-dir ~/.cache/huggingface/gguf/

# Option B: If Nemotron-Mini not available, use Llama-3.2-3B (good general draft)
huggingface-cli download \
  bartowski/Llama-3.2-3B-Instruct-GGUF \
  --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  --local-dir ~/.cache/huggingface/gguf/
```

For GLM speculative decoding (if switching to llama.cpp):
```bash
# GLM-4-9B as draft for GLM-4.7-Flash (same family)
huggingface-cli download \
  bartowski/THUDM_GLM-4-9B-Chat-GGUF \
  --include "GLM-4-9B-Chat-Q4_K_M.gguf" \
  --local-dir ~/.cache/huggingface/gguf/
```

---

## Part 6 — Immediate Quick Wins (Do Now)

These can be applied in under 30 minutes total with zero risk:

```bash
# 1. Update llama-swap config (add flash-attn, KV cache, cont-batching)
# Edit: /home/james/SCGen6/llama-swap/config.yaml

# 2. Update Ollama systemd override
sudo mkdir -p /etc/systemd/system/ollama.service.d/
sudo tee /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
EOF
sudo systemctl daemon-reload
sudo systemctl restart ollama

# 3. Verify Ollama picked up the settings
ollama ps  # Should show model with updated KV cache type
```
