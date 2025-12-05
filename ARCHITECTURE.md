# SC Gen 6 Architecture

> **⚠️ CRITICAL FOR AI AGENTS**: This document defines the fundamental architecture of SC Gen 6. **DO NOT make changes to the core structure** without explicit user approval. Follow established patterns when adding features.

## Core Principles

1. **Fully Local**: All processing happens on-device (no cloud dependencies)
2. **Parallel by Default**: Maximize CPU/GPU utilization through parallelization
3. **Modular Design**: Clear separation of concerns (ingestion, retrieval, generation, UI)
4. **Type Safety**: Use Python type hints throughout
5. **Configuration-Driven**: Behavior controlled via YAML configs, not hardcoded values

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Desktop UI (PySide6)                    │
│  ┌──────────┬──────────┬──────────┬──────────┬───────────┐  │
│  │  Query   │Documents │  Graph   │ Quality  │ Perform.  │  │
│  └──────────┴──────────┴──────────┴──────────┴───────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Core Services Layer                       │
│  ┌──────────────┬──────────────┬──────────────────────────┐ │
│  │  Ingestion   │  Retrieval   │  Generation              │ │
│  │  Pipeline    │  (Hybrid)    │  (LLM + Batch)           │ │
│  └──────────────┴──────────────┴──────────────────────────┘ │
│                 ┌──────────────────────────┐                │
│                 │  Quality Assessment      │                │
│                 │  (Cloud Evaluator)       │                │
│                 └──────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer                             │
│  ┌──────────┬──────────┬──────────┬──────────┬───────────┐  │
│  │ Vector   │  BM25    │  Graph   │ Assess.  │ Perform.  │  │
│  │  Store   │  Index   │  Store   │  DB      │  Logs     │  │
│  └──────────┴──────────┴──────────┴──────────┴───────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Ingestion Pipeline

**Location**: `src/ingestion/`

**Purpose**: Parse documents and prepare them for retrieval

**Architecture**:
```python
IngestionPipeline
├── Parallel Parsing (N workers)
│   ├── PDFParser (PyMuPDF + OCR fallback)
│   ├── DOCXParser (python-docx)
│   ├── EmailParser (.eml, .msg)
│   └── SpreadsheetParser (.xlsx, .xls)
├── AdaptiveChunker (document-type aware)
├── Batch Embedding (GPU accelerated)
└── Parallel Indexing (Vector + BM25)
```

**Key Files**:
- `ingestion_pipeline.py` - Orchestrates parallel ingestion
- `parsers/` - Document format parsers
- `chunkers/adaptive_chunker.py` - RCTS-based chunking

**DO NOT CHANGE**:
- Parallel processing architecture
- Parser interface contracts
- Chunking algorithm (RCTS)

### 2. Retrieval System

**Location**: `src/retrieval/`

**Purpose**: Hybrid retrieval with reranking

**Architecture**:
```
Query
├── Dense Retrieval (Chroma) → Top N
├── Keyword Retrieval (BM25) → Top N
├── Reciprocal Rank Fusion (RRF, k=60)
├── Cross-Encoder Reranking (GPU) → Top K
└── Final Top M chunks to LLM
```

**Key Files**:
- `hybrid_retriever.py` - Orchestrates hybrid retrieval
- `vector_store.py` - Chroma vector database
- `bm25_index.py` - BM25 keyword index
- `reranker.py` - Cross-encoder reranking (ONNX GPU)

**DO NOT CHANGE**:
- Hybrid retrieval pipeline order
- RRF fusion algorithm
- Mandatory reranking step

### 3. Generation System

**Location**: `src/generation/`

**Purpose**: LLM-based text generation with citations

**Architecture**:
```
LLMService
├── Single Generation (streaming)
├── Batch Generation (parallel)
│   ├── Chunk batching
│   ├── Parallel LLM calls
│   └── Synthesis step
└── Citation Verification
```

**Key Files**:
- `llm_service.py` - Main LLM service
- `chunk_batcher.py` - Parallel batch generation
- `citation.py` - Citation verification
- `background_task_manager.py` - Background task orchestrator

**DO NOT CHANGE**:
- Citation verification requirement
- Batch generation algorithm
- LLM client interface

### 4. Quality Assessment System

**Location**: `src/assessment/`

**Purpose**: Automated evaluation of RAG outputs using cloud models

**Architecture**:
```
AssessmentPipeline
├── AssessmentCollector (metadata capture)
├── CloudEvaluator (OpenAI/Anthropic/Google)
├── AssessmentWorker (async background thread)
├── SuggestionParser (feedback analysis)
└── SuggestionApplicator (auto-optimization)
```

**Key Files**:
- `cloud_evaluator.py` - Cloud API client
- `assessment_worker.py` - Background worker
- `assessment_db.py` - SQLite storage for results
- `suggestion_applicator.py` - Config optimization

### 5. Performance Analytics System

**Location**: `src/analytics/`

**Purpose**: Real-time monitoring and insights

**Architecture**:
```
PerformanceSystem
├── PerformanceLogger (metrics collection)
├── PerformanceDashboard (visualization)
└── InsightGenerator (LLM-based analysis)
```

**Key Files**:
- `performance_logger.py` - Metrics logging
- `performance_dashboard.py` - UI widget

### 6. Background Tasks

**Location**: `src/generation/` and `src/graph/`

**Purpose**: Generate case insights from document summaries

**Architecture**:
```
BackgroundTaskManager
├── Model Selection (largest available)
├── Task Execution (with progress tracking)
└── Incremental Updates (timestamp-based)

Generators (inherit from SummaryBasedGenerator)
├── CaseGraphGenerator (entities + relationships)
├── TimelineGenerator (events with sources)
├── CaseOverviewGenerator (synthesis)
└── DocumentRenamer (intelligent naming)
```

**Key Files**:
- `background_task_manager.py` - Task orchestration
- `summary_based_generator.py` - Base generator class
- `graph_generator.py` - Entity/relationship extraction
- `timeline_generator.py` - Timeline event extraction
- `case_overview_generator.py` - Case overview synthesis
- `document_renamer.py` - Document naming

**DO NOT CHANGE**:
- Summary-based generation pattern
- Incremental update mechanism
- Model selection strategy

### 7. Graph System

**Location**: `src/graph/`

**Purpose**: Entity and relationship management

**Architecture**:
```
CaseGraph
├── Entity Storage (deduplication)
├── Relationship Storage
├── Timeline Events (with source tracking)
└── Persistence (JSON)
```

**Key Files**:
- `case_graph.py` - Graph storage and management
- `entities.py` - Entity, Relationship, TimelineEvent dataclasses
- `graph_generator.py` - LLM-based extraction
- `timeline_generator.py` - Timeline extraction

**DO NOT CHANGE**:
- Entity/Relationship data structures
- Graph persistence format
- Source document tracking

### 8. UI System

**Location**: `src/ui/`

**Purpose**: PySide6 desktop interface

**Architecture**:
```
ModernMainWindow
├── Sidebar Navigation
├── Stacked Views
│   ├── Query View (chat interface)
│   ├── Documents View (management)
│   ├── Graph View (visualization)
│   ├── Timeline View (chronological)
│   ├── Summaries View (generation)
│   ├── Case Overview View (synthesis)
│   ├── Performance View (analytics)
│   └── Quality View (suggestions)
└── Details Panel (results, progress)
```

**Key Files**:
- `modern_main_window.py` - Main window
- `document_manager.py` - Document management
- `case_overview_widget.py` - Case overview UI
- `suggestions_viewer.py` - Quality suggestions
- `performance_dashboard.py` - Performance metrics

**DO NOT CHANGE**:
- Main window structure
- Navigation pattern
- View stacking architecture

## Data Flow

### Ingestion Flow

```
User uploads file
    ↓
IngestionPipeline.ingest_files()
    ↓
Parallel parsing (N workers)
    ↓
AdaptiveChunker.chunk_document()
    ↓
Batch embedding (GPU)
    ↓
VectorStore.add_chunks() + BM25Index.add_chunks()
    ↓
DocumentCatalog.update_record()
```

### Query Flow

```
User enters query
    ↓
HybridRetriever.retrieve()
    ├── VectorStore.search() → Top N
    ├── BM25Index.search() → Top N
    ├── RRF fusion
    └── Reranker.rerank() → Top K
    ↓
LLMService.generate_with_context()
    ├── Build prompt with chunks
    ├── Stream LLM response
    └── Verify citations
    ↓
Display result with citations
    ↓
(Async) AssessmentWorker.run()
    ├── Collect metadata
    ├── Call CloudEvaluator
    └── Save suggestions
```

### Background Task Flow

```
User clicks "Generate All"
    ↓
BackgroundTaskManager.run_all_tasks()
    ↓
For each task:
    ├── Select optimal model (largest available)
    ├── Get summaries (all or incremental)
    ├── Run generator (with progress tracking)
    └── Save results
    ↓
Update UI with results
```

## Configuration

### Main Config (`config/config.yaml`)

```yaml
models:
  llm:
    provider: "ollama"
    available: [list of models]
    default: "qwen2.5:32b-instruct"

retrieval:
  semantic_top_n: 30
  keyword_top_n: 30
  rerank_top_k: 40
  context_to_llm: 10

quality:
  provider: "openai"
  model: "gpt-5.1-instant"

background_tasks:
  model_selection: "largest_available"
  model_priority:
    - "gpt-oss-120b-mxfp4"
    - "qwen2.5:32b-instruct"
```

### Model Presets (`config/model_presets.json`)

```json
[
  {
    "label": "GPT-OSS-120B (MXFP4) - Fast",
    "model_name": "gpt-oss-120b-mxfp4",
    "path": "C:/Users/James/.lmstudio/models/.../gpt-oss-120b-MXFP4-00001-of-00002.gguf",
    "vram_gb": 59.0,
    "description": "120B parameter model with MXFP4 quantization"
  }
]
```

## Performance Optimization

### Parallelization Strategy

1. **Ingestion**: `max(1, cpu_count - 1)` workers
2. **Embedding**: Batch size 32-64 for GPU efficiency
3. **Generation**: Automatic batching for large contexts
4. **Background Tasks**: Sequential with progress tracking

### GPU Utilization

- **Embeddings**: ONNX Runtime with DirectML (Windows) or CUDA (Linux)
- **Reranking**: ONNX Runtime with GPU acceleration
- **LLM**: Ollama or llama.cpp with GPU layers

### Memory Management

- **Vector Store**: Chroma with persistent storage
- **BM25 Index**: In-memory with pickle persistence
- **Graph**: JSON persistence with lazy loading
- **Summaries**: SQLite with FTS5 indexing

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Mock external dependencies (LLM, GPU)
- Focus on business logic

### Integration Tests
- Test component interactions
- Use real models (small ones)
- Verify end-to-end flows

### Performance Tests
- Measure ingestion throughput
- Measure retrieval latency
- Measure generation speed

## Adding New Features

### DO Follow These Patterns

1. **Use existing base classes**: Inherit from `SummaryBasedGenerator` for new generators
2. **Follow configuration pattern**: Add settings to `config.yaml` and `config_loader.py`
3. **Implement progress tracking**: Use callbacks for long-running operations
4. **Add UI integration**: Create widgets following existing patterns
5. **Write tests**: Unit tests for logic, integration tests for flows

### DO NOT Do These Things

1. **Change core retrieval pipeline**: The hybrid retrieval + reranking is fundamental
2. **Remove citation verification**: Citations are mandatory
3. **Break parallel processing**: Parallelization is critical for performance
4. **Modify data structures**: Entity, Relationship, TimelineEvent are stable
5. **Change UI navigation**: The sidebar + stacked views pattern is established

## Common Pitfalls

### For AI Agents

1. **Don't assume synchronous operations**: Most operations are async or parallel
2. **Don't hardcode paths**: Use `config.yaml` and `Settings` class
3. **Don't skip type hints**: Type safety is required
4. **Don't ignore errors**: Proper error handling and logging is critical
5. **Don't change architecture**: Follow established patterns

### For Developers

1. **Always use virtual environment**: Avoid global package pollution
2. **Test with real documents**: Edge cases appear with real data
3. **Monitor GPU memory**: Large models can OOM
4. **Check logs**: Detailed logging in `logs/` directory
5. **Follow code style**: Black + Ruff + mypy

## Version History

- **v6.1** (2025-12-05): AI Quality Assessment, Performance Analytics, Native PySide6 UI
- **v6.0** (2025-11-29): Background tasks, case overview, parallel processing
- **v5.0** (2025-11): Graph system, timeline, summaries
- **v4.0** (2025-10): Hybrid retrieval, reranking
- **v3.0** (2025-09): Desktop UI, document management
- **v2.0** (2025-08): Adaptive chunking, citation verification
- **v1.0** (2025-07): Initial RAG pipeline

---

**Last Updated**: 2025-12-05
**Maintained By**: SC Gen 6 Development Team
