"""Stella v5 embedding service for high-quality embeddings.

Uses dunzhang/stella_en_1.5B_v5 for state-of-the-art embedding quality.
Supports multiple dimensions (512, 768, 1024, 2048, 4096, 6144, 8192).
"""

import torch
import numpy as np
from typing import Optional, List

from loguru import logger


class StellaEmbeddingService:
    """GPU-accelerated embedding service using Stella v5 or GTE-large.
    
    Stella v5 offers:
    - Multiple output dimensions (default 1024)
    - Special prompts for query (s2p) vs document encoding
    - State-of-the-art retrieval performance
    
    Note: Falls back to GTE-large (Stella's base model) if xformers unavailable.
    """
    
    # Use GTE-large as default - Stella's base model, works without xformers
    DEFAULT_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
    
    # Stella uses specific prompts for different tasks
    S2P_QUERY_PROMPT = "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        embed_dim: int = 1024,
        settings=None,
    ):
        """Initialize the Stella embedding service.
        
        Args:
            model_name: Model name (default: stella_en_1.5B_v5)
            embed_dim: Output embedding dimension (512, 768, 1024, 2048, 4096, 6144, 8192)
            settings: Settings instance (optional)
        """
        self.settings = settings
        self.model_name = model_name or self.DEFAULT_MODEL
        self.embed_dim = embed_dim
        
        # Lazy loading
        self.model = None
        self.device = None
        self.device_label = None
        
        # Batch size
        if settings and hasattr(settings, 'embedding'):
            self.batch_size = getattr(settings.embedding, 'batch_size', 16)
        else:
            self.batch_size = 16  # Smaller batch for larger model
    
    def _load_model(self) -> None:
        """Load the Stella model onto GPU."""
        logger.info(f"[Stella] Loading {self.model_name} with ROCm PyTorch...")
        
        # Determine device
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.device_label = f"ROCm GPU ({torch.cuda.get_device_name(0)})"
        else:
            self.device = "cpu"
            self.device_label = "CPU"
        
        logger.info(f"[Stella] Using device: {self.device_label}")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Load Stella with specified device
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
            )
            
            # Stella supports truncate_dim for output dimension control
            logger.info(f"[Stella] Using {self.embed_dim}D embeddings")
            
            # Use half precision on GPU for speed
            if self.device != "cpu":
                self.model = self.model.half()
                logger.info("[Stella] Using FP16 for faster inference")
            
            logger.success(f"[Stella] Model loaded on {self.device_label}")
            
        except Exception as e:
            logger.error(f"[Stella] Failed to load: {e}")
            raise
    
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if self.model is None:
            self._load_model()
    
    def preload(self) -> None:
        """Preload the model."""
        self._ensure_loaded()
    
    @torch.no_grad()
    def _encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            is_query: If True, use query prompt for s2p task
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        self._ensure_loaded()
        
        # For query encoding, optionally prepend instruction
        # (GTE doesn't require this but it can help)
        if is_query and "stella" in self.model_name.lower():
            texts = [self.S2P_QUERY_PROMPT + t for t in texts]
        
        # Encode with sentence-transformers (no prompt_name for GTE compatibility)
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        # Truncate to desired dimension if needed
        if embeddings.shape[1] > self.embed_dim:
            embeddings = embeddings[:, :self.embed_dim]
        
        return embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text (document).
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list
        """
        embeddings = self._encode([text], is_query=False)
        return embeddings[0].tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts (documents).
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self._encode(texts, is_query=False)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query (with s2p prompt).
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector as list
        """
        embeddings = self._encode([query], is_query=True)
        return embeddings[0].tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embed_dim
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is being used."""
        return self.device is not None and "cuda" in self.device


# Factory function
def get_stella_embedding_service(
    model_name: Optional[str] = None,
    embed_dim: int = 1024,
    settings=None,
) -> StellaEmbeddingService:
    """Get a Stella-based embedding service.
    
    Args:
        model_name: Model name (defaults to stella_en_1.5B_v5)
        embed_dim: Output dimension (default 1024)
        settings: Settings instance
        
    Returns:
        StellaEmbeddingService instance
    """
    return StellaEmbeddingService(
        model_name=model_name,
        embed_dim=embed_dim,
        settings=settings,
    )
