"""Embedding service using llama.cpp server's /embedding endpoint.

Uses nvidia/llama-embed-nemotron-8b via llama.cpp for high-quality embeddings.
This model achieves state-of-the-art on MMTEB benchmark.
"""

import time
import requests
from typing import Optional
from pathlib import Path

from src.config_loader import Settings, get_settings


class LlamaCppEmbeddingService:
    """Embedding service using llama-swap's /embedding endpoint.
    
    Routes embedding requests through llama-swap proxy which manages
    the backend llama-server with nemotron-embed-8b model.
    """
    
    # Default embedding server config
    DEFAULT_HOST = "http://localhost"
    DEFAULT_PORT = 8001  # Dedicated embedding server port
    
    # Default embedding dimension for nemotron-embed-8b
    DEFAULT_EMBEDDING_DIM = 4096
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        settings: Optional[Settings] = None,
        auto_start: bool = True,
    ):
        """Initialize llama.cpp embedding service.
        
        Args:
            host: Server host URL
            port: Server port
            settings: Settings instance
            auto_start: If True, auto-start server if not running
        """
        self.settings = settings or get_settings()
        self.host = host or self.DEFAULT_HOST
        self.port = port or self.DEFAULT_PORT
        self.base_url = f"{self.host}:{self.port}"
        self._is_ready = False
        self._auto_start = auto_start
        
        # Read embedding model name from config
        self.embedding_model = getattr(
            self.settings.models.embedding, 'embedding_model', 'nemotron-embed-8b'
        )
        self.embedding_dim = self.DEFAULT_EMBEDDING_DIM
        
    def _check_server(self) -> bool:
        """Check if embedding server is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def _ensure_server_running(self) -> bool:
        """Ensure llama-swap proxy is running, auto-start if needed."""
        if self._check_server():
            return True
            
        if not self._auto_start:
            return False
        
        # Auto-start llama-swap proxy (manages embedding server)
        try:
            from src.llm.llama_swap_manager import llama_swap_manager
            return llama_swap_manager.ensure_running(port=self.port)
        except Exception as e:
            print(f"[LlamaCpp Embed] Failed to auto-start llama-swap: {e}")
            return False
    
    def preload(self) -> None:
        """Ensure server is running and ready."""
        if self._ensure_server_running():
            self._is_ready = True
            print(f"[LlamaCpp Embed] Ready on {self.base_url}")
        else:
            print(f"[LlamaCpp Embed] WARNING: Server not available at {self.base_url}")
    
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings from llama-swap proxy.
        
        Uses OpenAI-compatible /v1/embeddings endpoint so llama-swap
        can route to the correct backend based on model name.
        """
        if not texts:
            return []
        
        try:
            # Use OpenAI-compatible format for llama-swap routing
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer local-llama"
                },
                json={
                    "model": self.embedding_model,  # From config
                    "input": texts  # OpenAI format uses "input" not "content"
                },
                timeout=300,  # 5 min for model loading
            )
            response.raise_for_status()
            
            result = response.json()
            
            # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
            if "data" in result:
                # Sort by index to ensure order matches input
                data = sorted(result["data"], key=lambda x: x.get("index", 0))
                return [item["embedding"] for item in data]
            # llama.cpp direct format fallback
            elif isinstance(result, list):
                return [item.get("embedding", item) for item in result]
            elif "embedding" in result:
                return [result["embedding"]]
            else:
                raise ValueError(f"Unexpected response format: {result}")
                
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to llama-swap at {self.base_url}. "
                f"Ensure llama-swap is running with embedding model configured."
            )
    
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        embeddings = self._embed([text])
        return embeddings[0]
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            return []
        
        # Process in smaller batches to avoid timeout
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            embeddings = self._embed(batch)
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def embed_query(self, query: str) -> list[float]:
        """Embed a query (same as embed_text for this model)."""
        return self.embed_text(query)
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim
    
    def is_gpu_available(self) -> bool:
        """Check if server is running (GPU status depends on server config)."""
        return self._check_server()
    
    def get_status(self) -> tuple[str, str]:
        """Get detailed status of the embedding server.
        
        Returns:
            Tuple of (status, message) where status is one of:
            - "ready": Server is running and responsive
            - "loading": Server is starting or loading model
            - "error": Server is not available
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            if response.status_code == 200:
                return ("ready", "Embedding server ready")
            else:
                return ("loading", f"Server returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            return ("error", f"Cannot connect to {self.base_url}")
        except requests.exceptions.Timeout:
            return ("loading", "Server responding slowly, may be loading model")
        except Exception as e:
            return ("error", f"Error checking server: {str(e)}")


# Factory function to get the appropriate embedding service
def get_embedding_service(use_llamacpp: bool = False, **kwargs):
    """Get embedding service based on configuration.
    
    Args:
        use_llamacpp: If True, use llama.cpp embedding server
        **kwargs: Additional arguments for the service
        
    Returns:
        Embedding service instance
    """
    if use_llamacpp:
        return LlamaCppEmbeddingService(**kwargs)
    else:
        from src.retrieval.embedding_service_onnx import ONNXEmbeddingService
        return ONNXEmbeddingService(**kwargs)
