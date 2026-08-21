"""ROCm PyTorch-based embedding service for GPU-accelerated embeddings.

Uses the same ROCm backend as Docling and LegalBERT for unified GPU usage.
Replaces ONNX/DirectML with native PyTorch for simpler GPU management.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
from transformers import AutoModel, AutoTokenizer


class ROCmEmbeddingService:
    """GPU-accelerated embedding service using ROCm PyTorch.
    
    Uses the same backend as Docling for unified GPU usage.
    """
    
    # BGE query prefix for better retrieval performance
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages:"
    
    DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        settings=None,
    ):
        """Initialize the embedding service.
        
        Args:
            model_name: HuggingFace model name. If None, uses settings or default.
            settings: Settings instance (optional)
        """
        self.settings = settings
        
        # Resolve model name with fallbacks
        if model_name is None:
            # Try to get from settings
            if settings:
                try:
                    model_name = settings.models.embedding.model_name
                except AttributeError:
                    pass
            # Use default if still None
            if model_name is None:
                model_name = self.DEFAULT_MODEL
        
        self.model_name = model_name
        
        # Lazy loading
        self.model = None
        self.tokenizer = None
        self.device = None
        self.device_label = None
        
        # Get batch size from settings
        if settings and hasattr(settings, 'embedding'):
            self.batch_size = getattr(settings.embedding, 'batch_size', 32)
        else:
            self.batch_size = 32

    
    def _load_model(self) -> None:
        """Load the model and tokenizer onto GPU."""
        print(f"[Embeddings] Loading {self.model_name} with ROCm PyTorch...")
        
        # Determine device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.device_label = f"ROCm GPU ({torch.cuda.get_device_name(0)})"
        else:
            self.device = torch.device("cpu")
            self.device_label = "CPU"
        
        print(f"[Embeddings] Using device: {self.device_label}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,  # Required for some models like GTE
        )
        
        # Move to GPU and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Use half precision for speed on GPU
        if self.device.type == "cuda":
            self.model = self.model.half()
            print("[Embeddings] Using FP16 for faster inference")
        
        print(f"[Embeddings] Model loaded on {self.device_label}")
    
    def _ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if self.model is None or self.tokenizer is None:
            self._load_model()
    
    def preload(self) -> None:
        """Preload the model."""
        self._ensure_loaded()
    
    @torch.no_grad()
    def _encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            is_query: If True, prepend BGE query prefix
            
        Returns:
            numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        self._ensure_loaded()
        
        # Add BGE query prefix if needed
        if is_query and "bge" in self.model_name.lower():
            texts = [f"{self.BGE_QUERY_PREFIX} {t}" for t in texts]
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            
            # Move to GPU
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            
            # Handle token_type_ids if present
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in encoded:
                inputs["token_type_ids"] = encoded["token_type_ids"].to(self.device)
            
            # Run inference
            outputs = self.model(**inputs)
            
            # Get [CLS] token embedding (first token)
            # For BGE models, this is the sentence embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Normalize
            cls_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)
            
            # Convert to numpy
            all_embeddings.append(cls_embeddings.float().cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list
        """
        embeddings = self._encode([text])
        return embeddings[0].tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self._encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query (adds BGE prefix).
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector as list
        """
        embeddings = self._encode([query], is_query=True)
        return embeddings[0].tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        self._ensure_loaded()
        
        # Try to get from model config
        if hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        
        # Fallback: encode a test string
        test_embedding = self.embed_text("test")
        return len(test_embedding)
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is being used."""
        return self.device is not None and self.device.type == "cuda"


# Factory function
def get_rocm_embedding_service(
    model_name: Optional[str] = None,
    settings=None,
) -> ROCmEmbeddingService:
    """Get a ROCm-based embedding service.
    
    Args:
        model_name: Model name (defaults to BGE-large)
        settings: Settings instance
        
    Returns:
        ROCmEmbeddingService instance
    """
    if model_name is None:
        if settings and hasattr(settings, 'embedding'):
            model_name = getattr(settings.embedding, 'model_name', 'BAAI/bge-large-en-v1.5')
        else:
            model_name = 'BAAI/bge-large-en-v1.5'
    
    return ROCmEmbeddingService(model_name=model_name, settings=settings)
