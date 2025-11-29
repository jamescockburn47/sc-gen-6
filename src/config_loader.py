"""Configuration loader with Pydantic validation and defaults.

This module loads config/config.yaml, validates all keys, and provides
a typed Settings object with sane defaults if keys are missing.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PythonConfig(BaseSettings):
    """Python version configuration."""

    version: str = "3.12"


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""

    default: str = "BAAI/bge-large-en-v1.5"
    use_onnx_gpu: bool = True  # Use ONNX Runtime + DirectML for GPU acceleration
    alternatives: list[str] = Field(default_factory=lambda: ["bge-m3", "Qwen/Qwen3-Embedding-8B"])


class RerankerConfig(BaseSettings):
    """Reranker model configuration."""

    default: str = "mixedbread-ai/mxbai-rerank-large-v2"
    use_onnx_gpu: bool = True  # Use ONNX Runtime + DirectML for GPU acceleration
    alternatives: list[str] = Field(
        default_factory=lambda: ["mixedbread-ai/mxbai-rerank-base-v2"]
    )


class OllamaConfig(BaseSettings):
    """Ollama service configuration."""

    host: str = "http://localhost:11434"
    keep_alive_ms: int = 600000


class LLMConfig(BaseSettings):
    """LLM model configuration."""

    backend: Literal["ollama", "llama_cpp"] = "ollama"  # Backend selector
    default: str = "qwen3:32b-instruct"
    temperature: float = 0.7  # Default temperature
    available: list[str] = Field(
        default_factory=lambda: [
            "qwen3:14b-instruct",
            "qwen3:32b-instruct",
            "qwen3:72b-instruct",
            "llama4:70b-instruct",
            "deepseek-v3:32b",
        ]
    )
    
    # Thinking mode configuration (enables extended reasoning for compatible models)
    enable_thinking: bool = True  # Enable thinking/reasoning mode when model supports it
    thinking_budget: int = 8192   # Max tokens for thinking (before visible response)
    
    # Models that support thinking/reasoning mode (partial name match)
    thinking_models: list[str] = Field(
        default_factory=lambda: [
            "qwen3",          # Qwen3 models with /think mode
            "deepseek-r1",    # DeepSeek-R1 with built-in reasoning
            "o1",             # OpenAI o1 models (if used via API)
            "claude-3.5",     # Claude with extended thinking
            "gemini-2.0",     # Gemini with thinking
        ]
    )


class ModelsConfig(BaseSettings):
    """Models configuration section."""

    model_config = SettingsConfigDict(extra="ignore")

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)



class AdvancedRetrievalConfig(BaseSettings):
    """Advanced retrieval configuration."""
    
    enabled: bool = False               # Master toggle for advanced pipeline
    
    # MMR (Maximal Marginal Relevance)
    enable_mmr: bool = True
    mmr_lambda: float = 0.5             # 0.5 = balance between relevance and diversity
    
    # LLM Grading
    enable_llm_grading: bool = True
    grading_model: str = "qwen2.5:7b"   # Fast model for grading
    grading_prompt: str = (
        "You are a strict judge. Is this text relevant to the query? "
        "Output JSON: {\"relevant\": true/false, \"reason\": \"...\"}"
    )
    
    # Context Expansion
    enable_context_expansion: bool = False
    expansion_window: int = 1           # Number of chunks before/after to fetch


class RetrievalConfig(BaseSettings):
    """Retrieval parameters configuration."""

    semantic_top_n: int = 50
    keyword_top_n: int = 50
    skip_reranking: bool = False
    rerank_top_k: int = 25              # Reduced from 50 (20-25 is sufficient)
    rerank_max_chars: int = 512         # Truncate chunks for reranking (speed optimization)
    context_to_llm: int = 15
    confidence_threshold: float = 0.15  # Lower default
    rrf_k: int = 60
    
    # Graph-enhanced search (user-optional, OFF by default)
    use_query_expansion: bool = False  # Expand query with entity aliases
    use_graph_context: bool = False    # Add entity/timeline context to LLM
    date_filter_mode: Literal["document", "mentioned", "both"] = "document"
    
    # Summary-enhanced search
    use_summaries: bool = True         # Include doc summaries in LLM context
    search_summaries: bool = False     # Also search summary text (experimental)

    @field_validator("confidence_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Ensure confidence threshold is in [0.0, 1.0]."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        return v


class GraphConfig(BaseSettings):
    """Case graph configuration."""

    enabled: bool = True                 # Master toggle - allows extraction
    auto_extract: bool = True            # Extract entities during ingestion
    extraction_confidence: float = 0.7   # Min confidence for auto-added entities
    max_context_entities: int = 10       # Max entities in graph context for LLM
    max_context_events: int = 5          # Max timeline events in graph context

    @field_validator("extraction_confidence")
    @classmethod
    def validate_extraction_confidence(cls, v: float) -> float:
        """Ensure extraction confidence is in [0.0, 1.0]."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("extraction_confidence must be between 0.0 and 1.0")
        return v


class ChunkingSizesConfig(BaseSettings):
    """Chunk sizes per document type."""

    witness_statement: int = 1024
    court_filing: int = 512
    pleading: int = 512
    statute: int = 512
    contract: int = 768
    disclosure: int = 512
    email: int = 512


class ChunkingOverlapsConfig(BaseSettings):
    """Chunk overlaps per document type."""

    witness_statement: int = 500
    court_filing: int = 200
    pleading: int = 200
    statute: int = 200
    contract: int = 300
    disclosure: int = 200
    email: int = 150


class ChunkingConfig(BaseSettings):
    """Chunking configuration."""

    sizes: ChunkingSizesConfig = Field(default_factory=ChunkingSizesConfig)
    overlaps: ChunkingOverlapsConfig = Field(default_factory=ChunkingOverlapsConfig)
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " "]
    )


class SummaryConfig(BaseSettings):
    """Document summary configuration."""
    
    enabled: bool = True                # Enable summary generation
    auto_generate: bool = False         # Auto-generate during ingestion (slow)
    summary_types: list[str] = Field(
        default_factory=lambda: ["overview", "key_points"]
    )
    max_summary_length: int = 300       # Target summary length in words
    include_in_search: bool = True      # Include summaries in search results
    
    # Dedicated summarization model (faster than main generation model)
    # Use a smaller, faster model for bulk summarization
    summarization_model: str = "qwen2.5:14b"  # Better reasoning for party extraction
    summarization_model_alternatives: list[str] = Field(
        default_factory=lambda: [
            "qwen2.5:14b",      # Excellent balance
            "qwen2.5:7b",       # Fast and accurate
            "qwen2.5:3b",       # Fastest, good for simple docs
            "llama3.2:3b",      # Very fast
            "mistral:7b",       # Good quality
            "phi3:mini",        # Microsoft's efficient model
        ]
    )
    parallel_summaries: int = 4         # Number of docs to summarize in parallel (increased for 96GB VRAM)
    batch_size: int = 10                # Documents per batch for progress tracking


class PathsConfig(BaseSettings):
    """Paths configuration."""

    documents: str = "data/documents"
    vector_db: str = "data/chroma_db"
    keyword_index: str = "data/keyword_index"  # Used for both BM25 and FTS5
    logs: str = "logs"


class PerformanceConfig(BaseSettings):
    """Performance tuning configuration."""

    embed_batch_size: int = 128     # Increased for 96GB VRAM GPU
    rerank_batch_size: int = 64     # Larger batches for GPU reranking
    max_workers: int = 4            # Reduced to prevent ONNX model loading race conditions
    cache_embeddings: bool = True


class UIConfig(BaseSettings):
    """UI configuration."""

    show_confidence_scores: bool = True
    theme: Literal["system", "light", "dark"] = "system"
    default_model: str = "qwen3:32b-instruct"


class GenerationConfig(BaseSettings):
    """Generation configuration."""

    enable_batching: bool = True
    min_chunks_for_batching: int = 6
    chunk_batch_size: int = 4       # Optimized for larger context
    max_batches: int = 8            # More batches for complex queries
    parallel_workers: int = 8       # Match Ollama parallelism (was 2)
    batch_max_tokens: int = 500
    enable_synthesis: bool = True
    synthesis_max_tokens: int = 4096    # Increased for comprehensive answers
    show_batch_headers: bool = False
    header_template: str = "### Batch {index}/{total} - chunks {start}-{end}"
    join_separator: str = "\n\n"


class BackgroundTasksConfig(BaseSettings):
    """Background/overnight task configuration."""
    
    model_selection: Literal["largest_available", "specific", "default"] = "largest_available"
    specific_model: str | None = None
    model_priority: list[str] = Field(
        default_factory=lambda: [
            "gpt-oss-120b-mxfp4",
            "qwen2.5:72b-instruct",
            "gemma2:27b",
            "qwen2.5:32b-instruct",
        ]
    )
    enabled_for: list[str] = Field(
        default_factory=lambda: [
            "case_graph_generation",
            "timeline_generation",
            "case_overview_generation",
            "document_renaming",
        ]
    )


class Settings(BaseSettings):
    """Main settings class with all configuration sections."""

    model_config = SettingsConfigDict(extra="ignore")

    python: PythonConfig = Field(default_factory=PythonConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    advanced_retrieval: AdvancedRetrievalConfig = Field(default_factory=AdvancedRetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    background_tasks: BackgroundTasksConfig = Field(default_factory=BackgroundTasksConfig)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Settings":
        """Load settings from a YAML file.

        Args:
            config_path: Path to config.yaml file

        Returns:
            Settings instance with loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
            ValidationError: If configuration doesn't match schema
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

        # Recursively merge with defaults
        return cls(**cls._merge_with_defaults(config_dict))

    @classmethod
    def _merge_with_defaults(cls, config_dict: dict) -> dict:
        """Merge config dict with default settings.

        This ensures missing keys get default values from Pydantic models.

        Args:
            config_dict: Dictionary loaded from YAML file

        Returns:
            Merged dictionary with defaults filled in
        """
        # Create a default instance to get all default values
        defaults = cls().model_dump()

        def deep_merge(base: dict, override: dict) -> dict:
            """Recursively merge override into base."""
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        return deep_merge(defaults, config_dict)


def get_settings(config_path: str | Path | None = None) -> Settings:
    """Get settings instance, loading from config file or using defaults.

    Args:
        config_path: Optional path to config.yaml. If None, tries config/config.yaml
                     relative to project root, then falls back to defaults.

    Returns:
        Settings instance
    """
    if config_path is None:
        # Try to find config/config.yaml relative to this file
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "config.yaml"

    if isinstance(config_path, Path) and config_path.exists():
        return Settings.from_yaml(config_path)

    # Return defaults if config file doesn't exist
    return Settings()


def get_config_profile() -> str:
    """Get the current configuration profile from environment.

    Returns:
        Profile name (default, 96gb, 128gb) or 'default' if not set.
    """
    import os
    return os.environ.get("SC_CONFIG", "default")

