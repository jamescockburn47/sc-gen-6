# Technical Specifications & Architecture

## 1. Retrieval Augmented Generation (RAG) Pipeline

### Architecture Overview
The system uses a **local-first Hybrid RAG** architecture designed for high-precision legal document retrieval. It runs entirely on-device (desktop/laptop) without external API dependencies for the RAG process.

**Pipeline Steps:**
1.  **Ingestion:** Parsing -> Adaptive Chunking -> Embedding -> Vector Storage
2.  **Retrieval:** Query -> Hybrid Search (Dense + Sparse) -> Reciprocal Rank Fusion (RRF) -> Cross-Encoder Reranking -> Context Filtering
3.  **Generation:** Context Assembly -> LLM Generation -> Citation Verification

---

## 2. Ingestion Process

### Document Parsing
*   **PDFs:** Uses `PyMuPDF` (fitz) for text extraction. Detects headers and page numbers.
*   **Word/Excel/Email:** Specialized parsers for `.docx`, `.xlsx`, `.eml`, `.msg`.
*   **OCR:** Tesseract-based fallback for scanned documents (triggered if text density is low).

### Document Type Detection
The system uses a **weighted scoring algorithm** to automatically classify documents:

| Type | Filename Patterns | Content Indicators |
|------|-------------------|-------------------|
| **Witness Statement** | "witness", "ws_", "statement" | "witness statement", "I make this statement", "statement of truth" |
| **Pleading** | "claim", "defence", "particulars", "poc" | "in the high court", "particulars of claim", "the claimant/defendant" |
| **Court Filing** | "order", "judgment", "motion" | "it is ordered", "before the honourable", "sealed by the court" |
| **Statute** | "act_", "regulation", "legislation" | "an act to", "be it enacted", "section 1", subsection patterns |
| **Contract** | "contract", "agreement", "deed", "nda" | "this agreement", "parties agree", "whereas", "in witness whereof" |
| **Email** | ".eml", ".msg", "email" | "from:", "to:", "subject:", "sent:" headers |
| **Disclosure** | "disclosure", "exhibit", "bundle" | Default for legal document bundles |

**Manual Override:** Users can manually change the document type in the Document Manager panel after ingestion.

### Adaptive Chunking (RCTS)
We use a **Recursive Character Text Splitting (RCTS)** strategy tailored for legal documents.
*   **Strategy:** Recursively splits text based on natural boundaries: `\n\n` (paragraphs), `\n` (lines), `. ` (sentences).
*   **Recursion Safety:** Implements a **maximum recursion depth limit (100)**. If text is too deeply nested or repetitive, it falls back to character slicing to prevent crashes (addressing `RecursionError`).
*   **Document-Type Rules:**
    *   *Witness Statements:* Large chunks (1024 tokens) with high overlap (500 tokens) to preserve narrative context.
    *   *Pleadings/Statutes:* Smaller chunks (512 tokens) for precise citation.
    *   *Contracts:* Medium chunks (768 tokens).

### Embedding
*   **Model:** `BAAI/bge-large-en-v1.5` (Dense vector embedding, 1024 dimensions).
*   **GPU Acceleration:** Uses **ONNX Runtime + DirectML** for GPU-accelerated embeddings on AMD/Intel/NVIDIA GPUs.
    *   **Speedup:** ~3.6x faster than CPU for large batches (500+ chunks).
    *   **Fallback:** Automatically falls back to CPU (`sentence-transformers`) if GPU unavailable.
    *   **ONNX Export:** Model is exported to ONNX format on first use and cached in `data/onnx_models/`.
*   **Offline Robustness:** The embedding service prioritizes loading models from the local HuggingFace cache (`~/.cache/huggingface/hub`) to operate without internet access.
*   **Storage:** **ChromaDB (v0.4+)** persistent on-disk vector store.

---

## 3. Retrieval & Ranking

### Hybrid Search
Combines two distinct search technologies:
1.  **Semantic Search (Dense):** Uses cosine similarity on vector embeddings to find *conceptual* matches (e.g., "breach of duty" matches text describing failures).
2.  **Keyword Search (Sparse):** Uses **BM25** algorithm to find *exact* matches (dates, specific clause numbers, names).

### Fusion (RRF)
*   **Algorithm:** Reciprocal Rank Fusion (RRF).
*   **Purpose:** Merges the top results from Semantic and BM25 searches into a single unified list, ensuring neither method dominates unfairly.

### Cross-Encoder Reranking
*   **Model:** `mixedbread-ai/mxbai-rerank-base-v2` (or large-v2).
*   **GPU Acceleration:** Uses **ONNX Runtime + DirectML** for GPU-accelerated reranking.
    *   **Speedup:** ~2.9x faster than CPU for 50 chunks.
    *   **Export:** Uses Hugging Face Optimum for robust ONNX export of Qwen2-based models.
*   **Process:** The top **50** candidates (increased from 10) from the fusion step are passed to a Cross-Encoder. This model "reads" the query and document pair together to output a precise relevance score (0-1).
*   **Bias Mitigation:** High `rerank_top_k` (50) ensures semantic results aren't pushed out by keyword floods before the deep ranking stage.

### Filtering & Context
*   **Confidence Threshold:** Default **0.15**. Chunks below this score are typically discarded.
*   **Minimum Guarantee:** To prevent empty answers, the system **guarantees at least 3 chunks** are passed to the LLM, even if they fall below the threshold.
*   **Context Window:** Default **15 chunks** (up from 5). This feeds significantly more evidence to the LLM.

---

## 4. Generation (LLM)

*   **Providers:** Supports `Ollama` (default), `LM Studio`, and `llama.cpp` server.
*   **Local Models:** Optimized for 14B-32B parameter models (e.g., `deepseek-r1:14b`, `qwen2.5:32b`).
*   **Citation Enforcement:** The system prompts the LLM to strictly cite sources using `[Source: File.pdf]` format.
*   **Post-Processing:** A citation verifier runs after generation to check if cited sources actually exist in the provided context.

---

## 5. System Resilience

*   **Reset Capability:** A "Reset All" feature completely wipes the Vector DB and BM25 indexes to recover from corrupted states.
*   **Connection Recovery:** The Vector Store automatically refreshes its database connection if it detects a "Collection does not exist" error (common after resets).
*   **Hardware Acceleration:** Auto-detects GPU availability (CUDA for NVIDIA, DirectML/Vulkan for AMD on Windows).

---

## 6. GPU Acceleration Details

### Supported Backends (Priority Order)
| Backend | Device | Platform | Use Case |
|---------|--------|----------|----------|
| **CUDA** | NVIDIA GPUs | Windows/Linux | Best performance for NVIDIA |
| **DirectML** | AMD/Intel/NVIDIA | Windows/WSL2 | AMD GPUs on Windows |
| **ROCm** | AMD GPUs | Linux only | Native AMD on Linux |
| **CPU** | All | All | Fallback |

### Component GPU Usage
| Component | GPU Backend | Notes |
|-----------|-------------|-------|
| **Embeddings** | ONNX + DirectML | 3.6x speedup over CPU |
| **Reranking** | ONNX + DirectML | 2.9x speedup over CPU |
| **LLM** | llama.cpp / Ollama | Separate GPU allocation |

### Performance Benchmarks (AMD Radeon 8060S)

**Embeddings (500 texts):**
| Method | Time | Per-Text |
|--------|------|----------|
| GPU (ONNX+DirectML) | 3,037ms | 6.1ms |
| CPU (sentence-transformers) | 11,000ms | 22.0ms |
| **Speedup** | **3.6x** | - |

**Reranking (50 chunks):**
| Method | Time | Per-Chunk |
|--------|------|-----------|
| GPU (ONNX+DirectML) | 880ms | 17.6ms |
| CPU (CrossEncoder) | 2,535ms | 50.7ms |
| **Speedup** | **2.9x** | - |

### Configuration
In `config/config.yaml`:
- `models.embedding.use_onnx_gpu: true` - Enable GPU for embeddings (default: true)
- `models.reranker.use_onnx_gpu: true` - Enable GPU for reranking (default: true)

Both are enabled by default. Set to `false` to force CPU-only mode.

---

*Last Updated: 2025-11-26*

