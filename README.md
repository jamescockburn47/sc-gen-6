# SC Gen 6 — Litigation Support RAG (Local Desktop)

A **fully local RAG pipeline** for litigation factual/procedural work (civil fraud & competition law). Outputs are **doctrinally exact**, **zero hallucinations**, and **fully cited** to source page/paragraph.

> **⚠️ IMPORTANT FOR AI AGENTS**: This codebase has a well-established architecture. Do NOT make fundamental structural changes without explicit user approval. See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

## Features

### Core RAG Pipeline
- **Fully Local**: All operations (parsing → chunking → embeddings → retrieval → reranking → generation) run on-device
- **Hybrid Retrieval**: BM25 + Dense vector search with Reciprocal Rank Fusion (RRF)
- **Cross-Encoder Reranking**: Mandatory reranking before generation for accuracy
- **Adaptive Chunking**: Document-type aware chunking with high overlap (40-50%)
- **Citation Enforcement**: Every factual claim must be cited; verification post-generation
- **Parallel Processing**: Multi-threaded ingestion and batch generation for maximum GPU utilization

### Advanced Features
- **AI Quality Assessment**: Automated evaluation of RAG outputs using cloud models (GPT-5.1, Claude 3.5, Gemini 1.5)
- **Performance Analytics**: Real-time tracking of generation speed, token usage, and system health
- **Case Graph**: Automatic entity and relationship extraction from documents
- **Timeline Generation**: Chronological event extraction with source tracking
- **Case Overview**: AI-generated high-level case summaries
- **Document Renaming**: Intelligent document naming from content
- **Background Tasks**: User-triggered generation with incremental updates
- **Multiple Model Support**: Ollama, llama.cpp, and LM Studio backends

### Desktop UI
- **PySide6 Interface**: Modern, responsive desktop application
- **Document Management**: Drag-and-drop ingestion with metadata editing
- **Query Interface**: Real-time streaming responses with citation verification
- **Quality Dashboard**: View assessment scores and apply optimization suggestions
- **Performance Dashboard**: Monitor system metrics and LLM performance
- **Graph Visualization**: Interactive entity and relationship graphs
- **Timeline View**: Chronological case events with source documents
- **Case Overview**: High-level case summary with key parties and dates

## Requirements

- **Python**: 3.11, 3.12, or 3.13 (3.14 NOT supported due to PySide6)
- **LLM Backend**: Ollama (recommended) or llama.cpp server
- **Tesseract OCR**: For scanned PDFs
- **Hardware**:
  - 8GB+ VRAM for 32B models
  - 64GB+ VRAM for 120B models (GPT-OSS 120B supported)
  - 64GB RAM recommended for large corpora
  - DirectML GPU acceleration (Windows) or CUDA (Linux)

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd "SC Gen 6"
```

### 2. Install Python 3.12 (Recommended)
- Download from: https://www.python.org/downloads/
- **Note**: Python 3.14 is NOT supported (PySide6 limitation)
- See `INSTALL_PYTHON.md` for detailed instructions

### 3. Create Virtual Environment
```bash
# Using Python 3.12 (recommended)
py -3.12 -m venv venv

# Activate on Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
# Windows:
py -3.12 -m pip install -r requirements.txt

# Or use install script:
install_dependencies.bat
```

### 5. Set Up Models
See `MODEL_SETUP.md` for detailed instructions.

**Quick Setup**:
```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5:32b-instruct

# Embedding & reranker models download automatically on first use
# Or run: python get_models.py
```

**Large Model Support** (GPT-OSS 120B):
- Configured in `config/model_presets.json`
- Requires 64GB+ VRAM
- See `ARCHITECTURE.md` for quantization options

### 6. Configure Application
```bash
cp .env.example .env
# Edit .env with your settings (optional)
```

### 7. Create Data Directories
```bash
mkdir -p data/documents data/chroma_db data/bm25_index logs
```

## Quick Start

### Desktop Application

**Quick Launch**:
```bash
python launch.py
# Or on Windows: double-click launch.bat
```

**Manual Launch**:
```bash
python -m src.ui.main
```

### Using the Interface

#### 1. Document Management
- Click **"Add Files..."** or **"Add Folder..."** to ingest documents
- Enable **"Auto-index after ingestion"** for automatic processing
- Parallel ingestion uses all CPU cores for speed
- View, delete, or update documents in the list

#### 2. Query & Answer
- Enter your question in the query box
- Adjust retrieval parameters (N, K, M, confidence threshold)
- Select model and document type filters
- Click **"Run Query"** for streaming results with citations
- Toggle **"Assess Quality"** to enable cloud-based evaluation

#### 3. Quality & Performance
- Navigate to **"Quality"** tab to review assessment scores and suggestions
- Check **"Performance"** tab for real-time system metrics and insights

#### 4. Case Graph & Timeline
- Navigate to **"Case Graph"** tab for entity visualization
- View **"Timeline"** for chronological events
- Check **"Case Overview"** for AI-generated summary

#### 5. Background Tasks (Advanced)
- Go to **Settings** → **Background Tasks**
- Click **"🚀 Generate All (Full)"** to run all generators
- Or use **"⚡ Update All (Incremental)"** for new documents only
- Individual task buttons for specific features

## Architecture

### Parallel Processing

**Ingestion Pipeline**:
```
Files → Parallel Parsing (N workers)
     → Parallel Chunking
     → Batch Embedding (GPU)
     → Parallel Indexing
```

**Generation Pipeline**:
```
Query → Retrieval (Hybrid)
     → Reranking (GPU)
     → Batch Generation (if enabled)
     → Citation Verification
     → Quality Assessment (Async)
```

**Background Tasks**:
```
Summaries → Case Graph (entities + relationships)
         → Timeline (events with sources)
         → Case Overview (synthesis)
         → Document Renaming
```

### Retrieval Pipeline

```
Query → Dense (Chroma) [Top N] + BM25 [Top N]
     → RRF (k=60)
     → Cross-Encoder Rerank [Top K]
     → Top M chunks to LLM
```

### Citation Format

```
[Source: filename | Page X, Para Y | "Section Title"]
```

Every factual claim must include a citation. The system verifies citations post-generation.

## Project Structure

```
SC Gen 6/
├── src/
│   ├── ingestion/            # Parsing, chunking, indexing pipeline
│   ├── retrieval/            # Hybrid retrieval, vector store, reranking
│   ├── generation/           # LLM service, batch generation, citation verification
│   ├── assessment/           # Quality assessment, cloud evaluator, suggestions
│   ├── analytics/            # Performance logging, dashboard backend
│   ├── graph/                # Case graph, timeline generation
│   ├── ui/                   # PySide6 desktop application
│   └── config/               # Configuration management
├── config/                   # Config files (yaml, json)
├── data/                     # Data storage (documents, db, indexes)
├── docs/                     # Documentation
├── models/                   # Model registry
└── tests/                    # Test suite
```

## Configuration

Edit `config/config.yaml` to customize:
- **Models**: LLM, embeddings, reranker selection
- **Retrieval**: N, K, M, confidence threshold
- **Quality**: Cloud provider (OpenAI, Anthropic, Google), API keys
- **Background Tasks**: Model selection strategy, enabled tasks
- **Paths**: Data directories, model paths

### Background Task Configuration

```yaml
background_tasks:
  model_selection: "largest_available"  # or "specific" or "default"
  specific_model: null
  model_priority:
    - "gpt-oss-120b-mxfp4"
    - "qwen2.5:32b-instruct"
  enabled_for:
    - "case_graph_generation"
    - "timeline_generation"
    - "case_overview_generation"
    - "document_renaming"
```

## Testing

Run the test suite:
```bash
pytest -q
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

## Development

### Code Style

- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking

Format code:
```bash
black src tests
ruff check --fix src tests
```

### Adding New Features

1. Create feature branch
2. Write tests first (TDD approach)
3. Implement feature
4. Ensure all tests pass
5. Update documentation
6. **Do NOT change fundamental architecture without approval**

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Check `OLLAMA_HOST` in `.env` matches your setup

### Model Not Found
- Pull the model: `ollama pull qwen2.5:32b-instruct`
- Check `config/config.yaml` for correct model names

### OCR Not Working
- Install Tesseract: https://github.com/tesseract-ocr/tesseract
- Ensure `pytesseract` can find the binary

### GPU Not Detected
- **Windows**: DirectML should work automatically
- **Linux**: Check CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- Models fall back to CPU automatically

### Background Task Errors
- Check `logs/` directory for detailed error messages
- Ensure document summaries exist before running tasks
- Verify model availability in Settings

## Performance Optimization

### Ingestion
- Uses all CPU cores for parallel parsing
- Batch embedding on GPU for speed
- Configurable worker count in `ingestion_pipeline.py`

### Generation
- Automatic batch generation for large contexts
- GPU acceleration for embeddings and reranking
- Parallel chunk processing where possible

### Background Tasks
- Incremental updates process only new documents
- Model selection prioritizes largest available model
- Progress tracking and cancellation support

## License

MIT License

## Support

For issues and questions, please open an issue on the repository.

---

**Version**: 6.1
**Last Updated**: 2025-12-05
