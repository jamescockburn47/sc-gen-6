# Nemotron 3 Nano 30B RAG Configuration Guide

## Model Overview

Based on the [NVIDIA Technical Report](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/README.md):

| Property | Value |
|----------|-------|
| **Total Parameters** | 30B |
| **Active Parameters** | 3.5B per token (MoE) |
| **Architecture** | Hybrid Mamba-2 + Transformer + MoE |
| **Context Window** | 128K tokens |
| **Supported Languages** | EN, ES, FR, DE, IT, JA |
| **Reasoning Mode** | Built-in (configurable) |

### Why Nemotron 3 is Excellent for Legal RAG

1. **128K Context Window** - Can process 80-100 pages of legal documents in a single call
2. **Hybrid Architecture** - Mamba-2 layers for efficient long-context processing
3. **Reasoning Capabilities** - Built-in chain-of-thought that's toggleable
4. **MoE Efficiency** - Only 3.5B params active per token = fast inference

---

## Configuration for RAG

### 1. Chunk Count Configuration

Your system controls chunk count through these parameters:

| Parameter | Description | Recommended for Nemotron |
|-----------|-------------|--------------------------|
| `context_to_llm` | Max chunks sent to LLM | **40-80** (can handle more) |
| `confidence_threshold` | Min reranker score | **0.25-0.35** (include borderline chunks) |
| `rerank_top_k` | Candidates for reranking | **40-60** |

### Current Issue: Why You're Getting < 5 Chunks

The search modes in the UI were overriding your config with restrictive defaults:

| Mode | Old `context_to_llm` | Old `confidence_threshold` | Problem |
|------|----------------------|---------------------------|---------|
| Fact Lookup | 5 | 0.65 | Too few chunks |
| Deep Analysis | 20 | 0.40 | Still limiting |

**Fix applied:** Updated to:
- Fact Lookup: 10 chunks, 0.50 threshold
- Deep Analysis: 40 chunks, 0.30 threshold

### 2. Recommended Config for Nemotron 3 (128K Context)

Update your `config/config.yaml`:

```yaml
retrieval:
  semantic_top_n: 100      # Cast wide net
  keyword_top_n: 100       # Cast wide net
  rerank_top_k: 60         # Rerank more candidates
  context_to_llm: 60       # Send more chunks (Nemotron can handle it)
  confidence_threshold: 0.25  # Include borderline chunks
  rrf_k: 60
  use_summaries: true      # Include document summaries for context

generation:
  enable_batching: true    # Auto-enabled for very large contexts
  min_chunks_for_batching: 80  # Only batch when really needed
  chunk_batch_size: 20     # Larger batches with 128K context
  enable_synthesis: true   # Combine batch outputs
  synthesis_max_tokens: 32768  # Long synthesis for comprehensive answers
```

### 3. How Context Budgeting Works

With Nemotron's **128K context window**:

| Component | Tokens | Notes |
|-----------|--------|-------|
| System Prompt | ~500 | Fixed |
| Document Summaries | ~2,000 | 4-5 summaries @ 400 tokens each |
| 60 Chunks | ~45,000 | Average 750 tokens/chunk |
| Safety Buffer | ~10,000 | For overhead |
| **Available for Chunks** | **~112,000** | Can fit 140+ average chunks |

**Key Insight:** You're massively underutilizing Nemotron's context window with 5-10 chunks.

---

## Advanced Techniques for Legal RAG

### 1. Reasoning Mode (Thinking)

Nemotron 3 has built-in reasoning that can be toggled. Your system already supports this:

```yaml
# config/config.yaml
models:
  llm:
    enable_thinking: true     # Enable extended reasoning
    thinking_budget: 8192     # Tokens for reasoning before answer
    thinking_models:
      - nemotron              # Nemotron is in the list
```

**For Complex Legal Queries:**
- Reasoning mode helps with multi-step analysis
- The model reasons internally before generating the final answer
- Reasoning is extracted and shown in the UI (collapsible)

**When to Disable:**
- Simple fact lookups (faster without reasoning overhead)
- Set `enable_thinking: false` in settings

### 2. Document Summaries for Context

Your system generates document summaries that provide "big picture" context:

```yaml
summary:
  enabled: true
  auto_generate: false    # Generate on-demand or during ingestion
  use_summaries: true     # Include in retrieval context
```

**How It Works:**
1. Each document gets a 300-word summary during ingestion
2. When retrieving chunks, the document summary is included once per document
3. LLM sees both the summary (context) and specific chunks (evidence)

**Recommended:** Enable for broader queries where understanding document relationships matters.

### 3. Query Expansion

For broad queries, enable query expansion:

```yaml
retrieval:
  use_query_expansion: true
```

This generates multiple related queries to retrieve more diverse chunks.

### 4. Hybrid Search Tuning

Your system uses:
- **Semantic Search (Dense):** Meaning-based matching
- **Keyword Search (FTS5):** Exact term matching
- **RRF Fusion:** Combines both

**For Legal Documents:**
```yaml
retrieval:
  hybrid_alpha: 0.7       # 70% semantic, 30% keyword
  rrf_k: 60               # Standard RRF parameter
```

**When to Favor Keywords:**
- Searching for specific statutes (e.g., "Section 42(1)")
- Named parties or case citations
- Exact legal terms

### 5. Batching Strategy (for Very Broad Queries)

When you need to process 100+ chunks, batching splits the work:

```
Query → Retrieve 100 chunks → Split into 5 batches of 20
                            ↓
                    Process each batch in parallel
                            ↓
                    Synthesize final answer
```

**Current Config:**
```yaml
generation:
  enable_batching: true
  min_chunks_for_batching: 10  # Batch if > 10 chunks
  chunk_batch_size: 8          # 8 chunks per batch
  max_batches: 8               # Max 8 batches = 64 chunks
  parallel_workers: 4          # Process 4 batches simultaneously
  enable_synthesis: true       # Combine batch outputs
```

**Recommended for Nemotron (128K context):**
```yaml
generation:
  min_chunks_for_batching: 60  # Only batch for very large contexts
  chunk_batch_size: 20         # Larger batches
  max_batches: 5               # 5 × 20 = 100 chunks max
```

---

## Search Mode Recommendations

| Query Type | Mode | `context_to_llm` | `confidence_threshold` |
|------------|------|------------------|------------------------|
| "What was the judgment date?" | Fact Lookup | 10 | 0.50 |
| "Summarize the key allegations" | Standard | 30 | 0.35 |
| "Analyze the liability position" | Deep Analysis | 40 | 0.30 |
| "Trace the fraud scheme timeline" | Deep Analysis | 60 | 0.25 |

---

## Quantization Quality Comparison

| Quantization | Size | Quality | Speed | VRAM |
|--------------|------|---------|-------|------|
| **BF16** (unquantized) | 63 GB | 100% | Slowest | 63+ GB |
| **Q8_0** | 33.6 GB | 95-97% | Fast | ~35 GB |
| **Q4_K_M** | 18.6 GB | 85-90% | Fastest | ~20 GB |

**For Legal RAG:**
- **Q8_0 recommended** - Best balance of quality and performance
- **BF16** - Only if you need absolute maximum quality and have VRAM
- **Q4_K_M** - Good for testing, but may lose nuance in legal language

---

## llama.cpp Server Configuration

Optimized settings for Nemotron 3 with your hardware:

```json
{
  "llama_server": {
    "executable": "C:/Users/James/Desktop/SC Gen 6/llama-cpp/llama-server.exe",
    "model_path": "C:/Users/James/Desktop/SC Gen 6/models/nemotron3-nano-30b-q8/Nemotron-3-Nano-30B-A3B-Q8_0.gguf",
    "context": 131072,
    "gpu_layers": 999,
    "parallel": 1,
    "batch": 4096,
    "timeout": 600,
    "flash_attn": true,
    "extra_args": "--ubatch-size 512 --threads 12 --cont-batching --cache-type-k q4_0 --cache-type-v q4_0"
  }
}
```

**Key Flags:**
- `--flash-attn` - Essential for 128K context efficiency
- `--cache-type-k q4_0 --cache-type-v q4_0` - Quantize KV cache to fit 128K context
- `--cont-batching` - Enable continuous batching for streaming
- `--ubatch-size 512` - Optimized for AMD Vulkan

---

## Troubleshooting

### "Only getting 3-5 chunks"

1. **Check search mode** - Fact Lookup defaults to fewer chunks
2. **Lower confidence threshold** - Set to 0.25-0.30
3. **Check reranker scores** - Run a test query and check logs:
   ```
   [RERANK DEBUG] Scores for X chunks:
     Min: 0.05, Max: 0.75, Avg: 0.35
     Threshold: 0.50
   ```
4. **Override in UI** - Manually adjust sliders before query

### "Response too short / summarized"

1. **Increase `synthesis_max_tokens`** in config (currently 24576, can go to 32768)
2. **Check prompt** - System prompt says "be comprehensive"
3. **Disable batching** for single-pass comprehensive answers

### "Reasoning traces in output"

Nemotron uses `<|channel|>` markers internally. The post-processor should strip these:
- Check `src/generation/llm_service.py` `_post_process_output()`
- Reasoning is extracted to `reasoning_content` in stats

---

## Quick Start Checklist

- [ ] Q8_0 or BF16 model downloaded
- [ ] `llm_runtime.json` points to correct model path
- [ ] `context: 131072` set in llama server config
- [ ] `flash_attn: true` enabled
- [ ] `context_to_llm: 40` or higher for comprehensive queries
- [ ] `confidence_threshold: 0.30` or lower for recall
- [ ] `enable_thinking: true` in config.yaml
- [ ] Test with broad query: "Summarize all key events in chronological order"
