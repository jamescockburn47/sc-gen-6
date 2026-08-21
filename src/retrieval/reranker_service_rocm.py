"""ROCm PyTorch-based reranker service.

Uses the same ROCm backend as Docling, LegalBERT, and embeddings for unified GPU usage.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple

from loguru import logger


class ROCmRerankerService:
    """GPU-accelerated reranker using ROCm PyTorch.
    
    Uses CrossEncoder from sentence-transformers on ROCm GPU.
    """
    
    DEFAULT_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        settings=None,
    ):
        """Initialize the reranker service.
        
        Args:
            model_name: HuggingFace model name. If None, uses default.
            settings: Settings instance (optional)
        """
        self.settings = settings
        
        # Resolve model name
        if model_name is None:
            if settings:
                try:
                    model_name = settings.models.reranker.model_name
                except AttributeError:
                    pass
            if model_name is None:
                model_name = self.DEFAULT_MODEL
        
        self.model_name = model_name
        
        # Lazy loading
        self.model = None
        self.device = None
        self.device_label = None
    
    def _load_model(self) -> None:
        """Load the CrossEncoder model onto GPU."""
        logger.info(f"[Reranker] Loading {self.model_name} with ROCm PyTorch...")
        
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.device_label = f"ROCm GPU ({torch.cuda.get_device_name(0)})"
        else:
            self.device = "cpu"
            self.device_label = "CPU"
        
        logger.info(f"[Reranker] Using device: {self.device_label}")
        
        try:
            from sentence_transformers import CrossEncoder
            
            # Load CrossEncoder with GPU device
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
            
            # Move model to GPU explicitly
            # NOTE: Not using FP16 for reranker - causes score precision issues
            if self.device != "cpu":
                self.model.model = self.model.model.to(self.device)
                # Keep FP32 for better score accuracy
            
            logger.success(f"[Reranker] Model loaded on {self.device_label}")
            
        except Exception as e:
            logger.error(f"[Reranker] Failed to load on {self.device_label}: {e}")
            # Fallback to CPU
            self.device = "cpu"
            self.device_label = "CPU (fallback)"
            
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(
                self.model_name,
                device="cpu",
                trust_remote_code=True,
            )
            logger.warning(f"[Reranker] Loaded on CPU (fallback)")
    
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if self.model is None:
            self._load_model()
    
    def preload(self) -> None:
        """Preload the model."""
        self._ensure_loaded()
    
    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, str]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Rerank documents by relevance to query.
        
        Args:
            query: Query text
            documents: List of (chunk_id, text) tuples to rerank
            top_k: Number of top results to return (default: all)
            
        Returns:
            List of (chunk_id, score) tuples, sorted by score descending
        """
        if not documents:
            return []
        
        self._ensure_loaded()
        
        # Extract chunk_ids and texts from tuples
        chunk_ids = [doc[0] for doc in documents]
        texts = [doc[1] for doc in documents]
        
        # Create query-document pairs
        pairs = [[query, text] for text in texts]
        
        # Get scores from CrossEncoder
        with torch.no_grad():
            scores = self.model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        
        # Create (chunk_id, score) pairs and sort by score descending
        id_score_pairs = list(zip(chunk_ids, scores))
        id_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k if specified
        if top_k is not None:
            id_score_pairs = id_score_pairs[:top_k]
        
        return id_score_pairs
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is being used."""
        return self.device is not None and "cuda" in self.device


# Factory function
def get_rocm_reranker_service(
    model_name: Optional[str] = None,
    settings=None,
) -> ROCmRerankerService:
    """Get a ROCm-based reranker service.
    
    Args:
        model_name: Model name (defaults to mxbai-rerank)
        settings: Settings instance
        
    Returns:
        ROCmRerankerService instance
    """
    return ROCmRerankerService(model_name=model_name, settings=settings)
