# SC Gen 6 — Litigation Support RAG (Local Desktop)

## Mission

Build a **fully local RAG pipeline** for litigation factual/procedural work (civil fraud & competition law). Outputs must be **doctrinally exact**, **zero hallucinations**, and **fully cited** to source page/paragraph.

## Scope & Constraints

- **Local for RAG** (parsing → chunking → embeddings → retrieval → reranking → generation are on-device).
- **APIs allowed** for other non-RAG features (e.g., calendaring, redaction QA, analytics), never for case materials.
- **Hardware:** GMTEK EVO X2 (128GB RAM, 96GB VRAM allocation); AMD Radeon 8060S (RDNA 3.5). Optimize for **32B class** models.
- **Platform:** **Windows 10/11 only** (Native Vulkan GPU acceleration via llama.cpp).
- **Corpus:** Up to 10,000 docs; UK statutes, pleadings, court filings, witness statements, disclosure (emails, contracts, spreadsheets, scans). Incremental updates on demand.

---

## Architecture (High Level)

**Ingestion → Parsing/OCR → Adaptive Chunking → Embedding → Vector DB (Chroma)**

<br>**Query → (BM25 + Dense) → RRF → Cross-Encoder Rerank → LLM (local) with enforced citations**

**Key properties:**

- **Hybrid retrieval** (BM25 + vector) with **Reciprocal Rank Fusion (RRF)**.
- **Cross-encoder rerank** is mandatory before generation (unless `skip_reranking: true`).
- **Adaptive chunking** by document type with high overlap to preserve context.
- **Citations enforced and verified** post-generation (fail-safe refusal if not met).

---

## Configuration System (CRITICAL FOR CONSISTENCY)

### Three-Tier Configuration Architecture

The system uses a **three-tier configuration system** that must be kept in sync:

1. **`config/config.yaml`** - Application defaults and user preferences
2. **`config/llm_runtime.json`** - Runtime LLM provider state (llama.cpp server settings)
3. **Environment variables** - Override for provider selection (optional)

### Configuration Priority (Highest to Lowest)

1. **Runtime state** (`config/llm_runtime.json`) - Controls active LLM provider
2. **Environment variables** (`LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL_NAME`)
3. **YAML config** (`config/config.yaml`) - Application defaults

### LLM Provider System

**IMPORTANT:** The system is **Windows-only** with **llama.cpp** as the primary LLM backend.

#### System 1: `config/config.yaml` → `src/config_loader.py`
- **Backend field:** `models.llm.backend` (values: `"ollama"` | `"llama_cpp"`)
- **Purpose:** Application-level preference
- **Used by:** UI model dropdown, first-run checks, general app logic
- **Default:** `"llama_cpp"` (in `src/config_loader.py`)

#### System 2: `config/llm_runtime.json` → `src/config/llm_config.py`
- **Provider field:** `provider` (values: `"llama_cpp"` | `"ollama"`)
- **Purpose:** Runtime LLM server connection (what's actually running)
- **Used by:** LLM client connections, server management
- **Default:** `"llama_cpp"` (in `src/config/llm_config.py`)

**CRITICAL RULE:** When updating configuration:
- **Primary:** Use `llama.cpp` with Vulkan backend for optimal AMD GPU performance
- **Fallback:** Ollama is supported but not recommended (slower due to abstraction overhead)
- LM Studio support has been **removed** due to API limitations with high VRAM allocation

### Current Configuration State (As of Latest Review)

**`config/config.yaml`:**
- `backend: lmstudio`
- `default: mistralai/mistral-small-3.2`
- Retrieval: `semantic_top_n: 22`, `keyword_top_n: 22`, `rerank_top_k: 14`, `context_to_llm: 14`, `confidence_threshold: 0.25`
- Chunking: Larger sizes (witness_statement: 2048, overlaps: 1024)

**`config/llm_runtime.json`:**
- `provider: llama_cpp`
- `model_name: gpt-oss-20b-MXFP4`
- Server: `http://127.0.0.1:8000/v1`

**MISMATCH DETECTED:** YAML says `lmstudio` but runtime uses `llama_cpp`. This is intentional if user wants llama.cpp server but UI shows LM Studio models.

### Configuration Update Rules for AI Agents

**When modifying LLM configuration:**

1. **Check both files:**
   - `config/config.yaml` → `models.llm.backend` and `models.llm.default`
   - `config/llm_runtime.json` → `provider` and `model_name`

2. **Update consistently:**
   - If changing to llama.cpp: Update `llm_runtime.json` provider AND model path
   - If changing to LM Studio: Update both YAML backend AND runtime provider
   - If changing to Ollama: Update YAML backend, ensure Ollama running

3. **Verify defaults match:**
   - `config.yaml` `default` should match `llm_runtime.json` `model_name` (or be compatible)
   - `config.yaml` `ui.default_model` should match the active model

4. **Test after changes:**
   - Run `python -m src.utils.first_run` to verify configuration
   - Check `python -c "from src.config_loader import get_settings; print(get_settings().models.llm.backend)"`
   - Check `python -c "from src.config.llm_config import load_llm_config; print(load_llm_config().provider)"`

**When modifying retrieval/chunking:**

1. **Update `config/config.yaml`** only (no runtime file needed)
2. **Check defaults in `src/config_loader.py`** - these are fallbacks if YAML missing
3. **Update tests** in `tests/test_config_loader.py` if changing defaults
4. **Document rationale** in commit message if changing from spec values

**When modifying model lists:**

1. **Update `config/config.yaml`** → `models.llm.available`
2. **Update `config/model_presets.json`** if using llama.cpp presets
3. **Update `src/config_loader.py`** defaults if changing standard list
4. **Ensure model names match** what the provider expects (Ollama format vs LM Studio format)

---

## Models (Local, Switchable)

### Generation (LLM)

**Current Active Setup:** llama.cpp server with Vulkan backend (Windows native)

**Supported Providers:**
- **llama.cpp** (`llama_cpp`) - **PRIMARY** - Direct Vulkan GPU access, best performance on AMD
- **Ollama** (`ollama`) - Fallback option, simpler but ~1.5x slower

**Default Models:**
- **llama.cpp:** User-configured GGUF model (set in `config/llm_runtime.json`)
- **Ollama:** `qwen2.5:32b-instruct`

**Model Switching:**
- **Via UI:** Model dropdown in query panel (uses presets for llama.cpp)
- **Via Config:** Update `llm_runtime.json` `model_name` and `llama_server.model_path`
- **Via Runtime:** Use model preset system (`config/model_presets.json`)

### Embeddings

- **Default:** `BAAI/bge-large-en-v1.5` (English litigation corpora; fast/lean).
- **Alternatives:**
  - `BAAI/bge-m3` (multilingual, multi-function: dense/sparse/multi-vector)
  - `Qwen/Qwen3-Embedding-8B` (SOTA multilingual & long-context; heavy VRAM)

### Reranker (Cross-Encoder)

- **Default:** `mixedbread-ai/mxbai-rerank-base-v2` (0.5B, faster) - **NOTE:** Current config uses base, spec recommends large
- **Alternatives:**
  - `mixedbread-ai/mxbai-rerank-large-v2` (1.5B, SOTA 2025) - **Recommended for production**
  - (Avoid NVIDIA `nv-rerankqa-mistral-4b-v3` due to 2025-12-19 EOL.)

**CONFIGURATION NOTE:** Current `config.yaml` uses `rerank-base-v2` but spec recommends `rerank-large-v2`. Consider updating for better accuracy.

---

## Chunking (Document-Type Adaptive)

| Doc Type             | Chunk Size (tokens) | Overlap (tokens) | Strategy / Separators                                 |
|----------------------|---------------------|------------------|-------------------------------------------------------|
| Witness statements   | 2048 (config) / 1024 (spec) | 1024 (config) / 500 (spec) | Recursive char; prefer paragraph/sentence boundaries  |
| Pleadings/filings    | 1024 (config) / 512 (spec) | 512 (config) / 200 (spec) | Paragraph-based; preserve numbered paras              |
| Statutes             | 1024 (config) / 512 (spec) | 512 (config) / 200 (spec) | Section/subsection boundaries                         |
| Contracts/disclosure | 1536 (config) / 768 (spec) | 768 (config) / 300 (spec) | Semantic + headings; respect clauses/tables           |
| Emails               | 768 (config) / 512 (spec) | 384 (config) / 150 (spec) | Header-aware (From/To/Subject/Date)                   |
| Scanned PDFs         | 1024 (config) / 512 (spec) | 512 (config) / 200 (spec) | OCR text; flag low-confidence regions                 |

**CONFIGURATION NOTE:** Current `config.yaml` uses **larger chunk sizes** than spec. This may be intentional for better context preservation but increases memory usage. Document rationale if keeping larger sizes.

**Notes**
- Use RCTS with doc-specific separators (`\n\n`, `\n`, `. `, ` `).
- Default to **40–50% overlap** for litigation to avoid boundary loss.
- Store **page #, paragraph #, section**, and **char offsets** per chunk.

---

## Retrieval & Ranking

**Pipeline:**
`Query → Dense (Chroma) [Top N]` + `BM25 [Top N]` → **RRF** → `Rerank Top K (cross‑encoder)` → **Top M** to LLM

**Current Configuration:**
- **N** (semantic/keyword): 22 (config) / 50 (spec)
- **K** (rerank candidates): 14 (config) / 20 (spec)
- **M** (context to LLM): 14 (config) / 5 (spec)
- **Confidence threshold:** 0.25 (config) / 0.70 (spec)
- **RRF k:** 60 (both)

**CONFIGURATION NOTE:** Current config uses **more aggressive retrieval** (higher N, K, M) and **lower confidence threshold** than spec. This trades precision for recall. Document if intentional.

**User‑configurable controls (UI):**
- **N** initial per-retriever results (default from config, range 10–200)
- **K** rerank candidates (default from config, range 5–120)
- **M** chunks to LLM context (default from config, range 1–150)
- **Confidence threshold** on reranker score (default from config, 0.0–1.0)
- **Skip reranking** toggle (faster, lower quality)

**Fusion:** RRF (rank‑based; robust across domains).  
**Filtering:** by doc type, date range, file name/party metadata.  
**Caching:** memoize query vectors and recent reranks.

---

## Citation & Hallucination Controls

**Hard rules given to LLM:**

1. Use **only** provided sources.
2. **Cite every factual claim**: `[Source: filename | Page X, Para Y | "Section Title"]`.
3. If not in sources: respond **"Not found in provided documents."**
4. Quote exact text where possible; note conflicts and cite both.

**Verifier (post‑generation):**

- Parse citations; map to provided chunk IDs.
- Every sentence ending with a period must have a matching citation span.
- If **no chunks ≥ threshold**, return refusal message before generation.
- Show per‑citation **confidence** (reranker score) in UI.

---

## Desktop UI (PySide6/PyQt6)

**Panels**

- **Documents:** Drag‑drop ingest; type override; progress/errors; delete/update.
- **Query:** Text box; model dropdown; filters; sliders for N/K/M/threshold; run/stop.
- **Results:** Streamed answer; citations (click → open doc at page); confidence toggles; retrieved chunk stack trace.
- **Settings:** Model registry; paths; defaults; chunk overrides; index management; logs.
- **Export:** DOCX/PDF reports with citations and quoted excerpts.

**Responsiveness:** Background workers for parsing/embedding/rerank/generation; UI never blocks.

---

## Storage & Indexes

- **Vector DB:** **Chroma v1.0+** persistent collection `litigation_docs` (cosine).
- **Keyword index:** BM25 sidecar (disk‑persisted).
- **File system:** `data/documents`, `data/chroma_db`, `data/bm25_index`, `logs/`.
- **Ops:** add/update/remove document; rebuild indexes; show stats.

---

## Evaluation

- **Benchmarking:** Integrate **LegalBench‑RAG** (span‑level, citation‑aware) for sanity checks.
- **Metrics:**
  - Retrieval recall@K (K=20)
  - Citation accuracy (pinpoint page/para)
  - Refusal correctness (no false positives)
  - Latency (P95 end‑to‑end)
- **Datasets:** Your prior cases (redacted) + LegalBench‑RAG artifacts.

---

## Packaging

- **LLM:** llama.cpp server on Windows (auto-started if configured); Ollama optional; LM Studio optional.
- **App:** PyInstaller bundle for desktop; first‑run model check & local model registry warm‑up.
- **Logs:** rotate in `logs/sc-gen-6.log`.
- **Privacy:** no network calls from the RAG pipeline.

---

## Launcher Policy (CRITICAL)

### Single Unified Launcher

**RULE: There is ONE launcher file: `run.bat`**

Do NOT create additional launcher scripts.

### Usage

**Double-click `run.bat`** - Launches the Windows application directly with Vulkan GPU acceleration.

```
============================================================
   SC Gen 6 - Litigation Support RAG
   Windows Native (Vulkan GPU Acceleration)
============================================================

Starting application...
```

### Launcher Files (Authoritative List)

| File | Purpose | Status |
|------|---------|--------|
| `run.bat` | **UNIFIED LAUNCHER** - Use this | ACTIVE |
| `launch.py` | Python entry point (called by run.bat) | ACTIVE |
| `scripts/start_llama_server.bat` | Start llama.cpp server (optional) | ACTIVE |

### Deprecated Launchers (Do Not Use)

These files exist for backwards compatibility but should NOT be used:
- `launch.bat` - Use `run.bat` instead
- `launch-96gb.bat` - Use `run.bat` instead (config profiles not implemented)
- `setup_rocm.sh` - **REMOVED** (WSL no longer supported)
- `install_dependencies.bat` - Use `pip install -r requirements.txt`
- `rebuild_bm25.bat` - Use Python directly

### For AI Agents

**NEVER create new launcher scripts.** If new launch functionality is needed:
1. Add arguments to `run.bat`
2. Add logic to `launch.py`
3. Document in this section

**Examples of what NOT to do:**
- ❌ Creating `launch_gpu.bat`
- ❌ Creating `run_wsl.bat`
- ❌ Creating `start_app.sh`

**Instead:**
- ✅ Add `--gpu` flag to `run.bat`
- ✅ Add platform detection to `launch.py`

---

## GPU Acceleration & Backend Support

### Windows-Only Deployment (Vulkan)

**Platform:** Windows 10/11 only. WSL2 support has been **removed** due to:
- ROCm requires `/dev/kfd` which WSL2 doesn't expose
- DirectML adds translation overhead (~10-20% slower)
- Memory management issues with 128GB RAM allocation

### Supported GPU Backends (Priority Order)

| Backend | Device | Platform | Notes |
|---------|--------|----------|-------|
| **Vulkan** | llama.cpp | Windows | **PRIMARY** - Direct AMD GPU access |
| **CUDA** | `cuda` | Windows | NVIDIA GPUs (if available) |
| **CPU** | `cpu` | All | Fallback with AVX-512 optimization |

### Current Hardware Configuration

**System:** GMKTEC EVO X2 with AMD Ryzen AI HX 370
- **RAM:** 128GB (96GB allocated to VRAM)
- **GPU:** AMD Radeon 8060S (RDNA 3.5 integrated, 16 CUs)
- **Platform:** Windows 11 Native
- **Backend:** llama.cpp with Vulkan

### llama.cpp Vulkan Optimization Flags

The following flags are configured in `config/llm_runtime.json` for optimal AMD performance:

```json
{
  "llama_server": {
    "gpu_layers": 999,
    "context": 32768,
    "batch": 4096,
    "flash_attn": true,
    "extra_args": "--ubatch-size 512 --threads 12 -cb -no-mmap"
  }
}
```

**Flag explanations:**
- `gpu_layers: 999` - Offload all layers to GPU (Vulkan)
- `context: 32768` - Large context for RAG document stuffing
- `batch: 4096` - High batch size for fast prompt processing
- `flash_attn: true` - Flash attention for long-context speed
- `--ubatch-size 512` - Micro-batch optimized for AMD
- `--threads 12` - CPU threads for Ryzen AI HX 370
- `-cb` - Continuous batching for streaming
- `-no-mmap` - Force model into RAM (128GB available)

### Library Compatibility Notes

| Library | Vulkan (llama.cpp) | CPU | Notes |
|---------|-------------------|-----|-------|
| llama.cpp | ✅ Primary | ✅ | Best performance on AMD |
| sentence-transformers | N/A | ✅ | Embeddings run on CPU |
| transformers | N/A | ✅ | Reranker runs on CPU |

---

## Configuration (YAML)

**File:** `config/config.yaml`

**Structure:**
```yaml
python:
  version: "3.11"   # 3.11 or 3.12 supported

models:
  embedding:
    default: BAAI/bge-large-en-v1.5
    alternatives:
      - BAAI/bge-m3
      - Qwen/Qwen3-Embedding-8B
  reranker:
    default: mixedbread-ai/mxbai-rerank-large-v2  # RECOMMENDED: large for production
    alternatives:
      - mixedbread-ai/mxbai-rerank-base-v2
  llm:
    backend: ollama  # or "lmstudio" - application preference
    default: qwen3:32b-instruct  # Must match active provider's model format
    available:
      - qwen3:14b-instruct
      - qwen3:32b-instruct
      - qwen3:72b-instruct
      - llama4:70b-instruct
      - deepseek-v3:32b
  ollama:
    host: http://localhost:11434
    keep_alive_ms: 600000
  lmstudio:
    host: http://localhost:1234
    api_key: lm-studio

retrieval:
  semantic_top_n: 50  # RECOMMENDED: 50 for balanced precision/recall
  keyword_top_n: 50
  skip_reranking: false  # Set true only for very large context models
  rerank_top_k: 20  # RECOMMENDED: 20 for good reranking coverage
  context_to_llm: 5  # RECOMMENDED: 5 for focused answers, increase for complex queries
  confidence_threshold: 0.70  # RECOMMENDED: 0.70 for high precision, lower for recall
  rrf_k: 60

chunking:
  sizes:
    witness_statement: 1024  # RECOMMENDED: Match spec for memory efficiency
    court_filing: 512
    pleading: 512
    statute: 512
    contract: 768
    disclosure: 512
    email: 512
  overlaps:
    witness_statement: 500  # RECOMMENDED: 40-50% overlap
    court_filing: 200
    pleading: 200
    statute: 200
    contract: 300
    disclosure: 200
    email: 150
  separators: ["\n\n", "\n", ". ", " "]

paths:
  documents: data/documents
  vector_db: data/chroma_db
  bm25_index: data/bm25_index
  logs: logs

performance:
  embed_batch_size: 32
  rerank_batch_size: 16
  max_workers: 4
  cache_embeddings: true

ui:
  show_confidence_scores: true
  theme: system
  default_model: qwen3:32b-instruct  # Must match models.llm.default
```

**Runtime Configuration:** `config/llm_runtime.json` (for llama.cpp provider)

```json
{
  "provider": "llama_cpp",
  "base_url": "http://127.0.0.1:8000/v1",
  "api_key": "local-llama",
  "model_name": "gpt-oss-20b-MXFP4",
  "llama_server": {
    "executable": "C:/path/to/llama-server.exe",
    "model_path": "C:/path/to/model.gguf",
    "context": 32768,
    "gpu_layers": 999,
    "parallel": 2,
    "batch": 1024,
    "timeout": 1800,
    "host": "127.0.0.1",
    "port": 8000,
    "flash_attn": false,
    "extra_args": ""
  }
}
```

---

## Success Criteria

### Functional Requirements

- ✅ Parse 95%+ of uploaded documents successfully
- ✅ Retrieve relevant chunks in <500ms
- ✅ Generate answers with citations in <10 seconds
- ✅ Zero uncited factual claims (enforced by citation parser)
- ✅ User can adjust retrieval parameters and see effect
- ✅ User can switch models for different tasks
- ✅ Export reports preserve citations and formatting

### Non-Functional Requirements

- ✅ UI remains responsive during processing (threading)
- ✅ Data persists across sessions (indexes, config)
- ✅ Errors are logged and shown to user
- ✅ System can handle 10,000 document corpus

### User Experience

- "Can I trust the citations?" → YES (verified against sources)
- "Can I find facts quickly?" → YES (<10 sec queries)
- "Can I adjust context depth?" → YES (M slider in UI)
- "Can I try different models?" → YES (dropdown selector)

---

## Implementation Notes

### For AI Agents / Claude Code CLI

**CRITICAL RULES FOR CONFIGURATION UPDATES:**

1. **Always check both configuration systems:**
   - `config/config.yaml` (application defaults)
   - `config/llm_runtime.json` (runtime LLM provider)
   - `src/config_loader.py` (code defaults)

2. **When changing LLM settings:**
   - Update `config.yaml` `models.llm.backend` AND `models.llm.default`
   - Update `llm_runtime.json` `provider` AND `model_name` if using llama.cpp
   - Ensure `ui.default_model` matches `models.llm.default`
   - Verify provider compatibility (Ollama format vs LM Studio format vs llama.cpp)

3. **When changing retrieval/chunking:**
   - Update `config.yaml` only
   - Check if defaults in `src/config_loader.py` need updating
   - Update tests in `tests/test_config_loader.py` if changing defaults
   - Document rationale if deviating from spec values

4. **Test after configuration changes:**
   ```bash
   python -m src.utils.first_run  # Verify configuration
   python -c "from src.config_loader import get_settings; s=get_settings(); print(f'Backend: {s.models.llm.backend}, Default: {s.models.llm.default}')"
   python -c "from src.config.llm_config import load_llm_config; c=load_llm_config(); print(f'Provider: {c.provider}, Model: {c.model_name}')"
   ```

5. **Document configuration decisions:**
   - If deviating from spec (e.g., larger chunks, lower confidence), explain why
   - If using different defaults, note hardware/use-case rationale
   - Update this `agents.md` file if configuration patterns change

6. **Code Style:**
   - Python 3.11+ features
   - Type hints (PEP 484)
   - Docstrings (Google style)
   - Black formatting
   - Import organization: stdlib → third-party → local

7. **Testing Philosophy:**
   - Write tests as you build
   - Test edge cases (empty docs, corrupted files, etc.)
   - Mock expensive operations (LLM calls) in unit tests
   - Integration tests use real models
   - Update `tests/test_config_loader.py` when changing defaults

8. **Performance Optimization:**
   - Batch operations where possible
   - Cache expensive computations
   - Use threading/async for I/O
   - Profile before optimizing

### Configuration Consistency Checklist

Before committing configuration changes:

- [ ] `config/config.yaml` `models.llm.backend` matches intended provider
- [ ] `config/config.yaml` `models.llm.default` matches active model
- [ ] `config/config.yaml` `ui.default_model` matches `models.llm.default`
- [ ] `config/llm_runtime.json` `provider` matches runtime setup (if using llama.cpp)
- [ ] `config/llm_runtime.json` `model_name` matches model file (if using llama.cpp)
- [ ] `src/config_loader.py` defaults are reasonable fallbacks
- [ ] `tests/test_config_loader.py` expectations match new defaults
- [ ] Configuration rationale documented (if deviating from spec)
- [ ] `agents.md` updated if configuration patterns change

---

## Known Configuration Discrepancies

**As of latest review, these discrepancies exist:**

1. **Chunk sizes:** Config uses larger sizes (2048/1024) vs spec (1024/512)
   - **Rationale needed:** Document why larger chunks if keeping them
   - **Action:** Either update config to match spec OR document hardware/use-case rationale

2. **Retrieval parameters:** Config uses more aggressive settings (N=22, K=14, M=14, threshold=0.25) vs spec (N=50, K=20, M=5, threshold=0.70)
   - **Rationale needed:** Document precision vs recall tradeoff
   - **Action:** Either align with spec OR document why different values

3. **Reranker:** Config uses `base-v2` vs spec recommends `large-v2`
   - **Rationale needed:** Document performance vs accuracy tradeoff
   - **Action:** Consider updating to `large-v2` for production

4. **LLM backend mismatch:** YAML says `lmstudio` but runtime uses `llama_cpp`
   - **Status:** Intentional if user wants llama.cpp server
   - **Action:** Document this is intentional OR align both

**RECOMMENDATION:** Review these discrepancies and either:
- Update config to match spec (if spec values are preferred)
- Update spec to match config (if config values are intentional)
- Document rationale for each discrepancy
