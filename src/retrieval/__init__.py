"""Retrieval components: embeddings, vector store, BM25/FTS5, hybrid retrieval, reranking, summaries."""

from typing import Optional, TYPE_CHECKING

from src.retrieval.fts5_index import FTS5Index, FTS5IndexCompat
from src.retrieval.embedding_service import EmbeddingService
from src.retrieval.embedding_service_onnx import ONNXEmbeddingService
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.rrf import ScoredChunk, fuse
from src.retrieval.summary_store import SummaryStore, DocumentSummary, generate_document_summary
from src.retrieval.vector_store import VectorStore

if TYPE_CHECKING:
    from src.config_loader import Settings


def get_embedding_service(
    model_name: Optional[str] = None,
    settings: Optional["Settings"] = None,
    force_cpu: bool = False,
) -> EmbeddingService | ONNXEmbeddingService:
    """Factory function to get the best available embedding service.
    
    Automatically selects:
    1. llama.cpp via llama-swap if use_llamacpp is True
    2. ONNX+DirectML for GPU acceleration when available
    3. sentence-transformers on CPU as fallback
    
    Args:
        model_name: HuggingFace model name. If None, uses default from config.
        settings: Settings instance. If None, loads from config.
        force_cpu: Force CPU-only mode (uses sentence-transformers).
        
    Returns:
        Either LlamaCppEmbeddingService, ONNXEmbeddingService (GPU), or EmbeddingService (CPU)
    """
    from src.config_loader import get_settings
    
    settings = settings or get_settings()
    
    # Check if llama.cpp mode is enabled (via llama-swap)
    if settings.models.embedding.use_llamacpp and not force_cpu:
        try:
            from src.retrieval.embedding_service_llamacpp import LlamaCppEmbeddingService
            
            # Parse host/port from llamacpp_url
            url = settings.models.embedding.llamacpp_url
            if url.startswith("http://"):
                url_parts = url[7:].split(":")
                host = f"http://{url_parts[0]}"
                port = int(url_parts[1]) if len(url_parts) > 1 else 8000
            else:
                host = "http://localhost"
                port = 8000
            
            embed_model = getattr(settings.models.embedding, 'embedding_model', settings.models.embedding.default)
            print(f"[Embeddings] Using llama.cpp via llama-swap ({embed_model})")
            return LlamaCppEmbeddingService(host=host, port=port, settings=settings)
        except Exception as e:
            print(f"[Embeddings] Failed to use llama.cpp: {e}, falling back to ONNX")
    
    # Check if ONNX GPU mode is enabled and not forced to CPU
    use_onnx = settings.models.embedding.use_onnx_gpu and not force_cpu
    
    if use_onnx:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            
            # Only use ONNX if DirectML or CUDA is available
            if "DmlExecutionProvider" in providers or "CUDAExecutionProvider" in providers:
                print("[Embeddings] Using ONNX Runtime with GPU acceleration")
                return ONNXEmbeddingService(model_name=model_name, settings=settings)
            else:
                print("[Embeddings] ONNX GPU providers not available, using CPU")
        except ImportError:
            print("[Embeddings] ONNX Runtime not installed, using CPU")
    
    # Fallback to sentence-transformers (CPU)
    print("[Embeddings] Using sentence-transformers (CPU)")
    return EmbeddingService(model_name=model_name, settings=settings)




__all__ = [
    "EmbeddingService",
    "ONNXEmbeddingService",
    "get_embedding_service",
    "VectorStore",
    "FTS5Index",
    "FTS5IndexCompat",
    "HybridRetriever",
    "ScoredChunk",
    "fuse",
    "SummaryStore",
    "DocumentSummary",
    "generate_document_summary",
]
