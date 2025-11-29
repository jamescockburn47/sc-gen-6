"""Reranker model wrappers."""

from typing import Optional, TYPE_CHECKING

from src.models.reranker.reranker_service import RerankerService
from src.models.reranker.reranker_service_onnx import ONNXRerankerService

if TYPE_CHECKING:
    from src.config_loader import Settings


def get_reranker_service(
    model_name: Optional[str] = None,
    settings: Optional["Settings"] = None,
    force_cpu: bool = False,
) -> RerankerService | ONNXRerankerService:
    """Factory function to get the best available reranker service.
    
    Automatically selects ONNX+DirectML for GPU acceleration when available,
    falling back to sentence-transformers CrossEncoder on CPU otherwise.
    
    Args:
        model_name: HuggingFace model name. If None, uses default from config.
        settings: Settings instance. If None, loads from config.
        force_cpu: Force CPU-only mode (uses sentence-transformers CrossEncoder).
        
    Returns:
        Either ONNXRerankerService (GPU) or RerankerService (CPU)
    """
    from src.config_loader import get_settings
    
    settings = settings or get_settings()
    
    # Check if ONNX GPU mode should be used
    use_onnx = getattr(settings.models.reranker, 'use_onnx_gpu', True) and not force_cpu
    
    if use_onnx:
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            
            # Only use ONNX if DirectML or CUDA is available
            if "DmlExecutionProvider" in providers or "CUDAExecutionProvider" in providers:
                print("[Reranker] Using ONNX Runtime with GPU acceleration")
                return ONNXRerankerService(model_name=model_name, settings=settings)
            else:
                print("[Reranker] ONNX GPU providers not available, using CPU")
        except ImportError:
            print("[Reranker] ONNX Runtime not installed, using CPU")
    
    # Fallback to sentence-transformers CrossEncoder (CPU)
    print("[Reranker] Using sentence-transformers CrossEncoder (CPU)")
    return RerankerService(model_name=model_name, settings=settings)


__all__ = ["RerankerService", "ONNXRerankerService", "get_reranker_service"]




