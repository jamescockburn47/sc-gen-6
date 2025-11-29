"""Embedding service using BGE models with sentence-transformers.

Supports GPU acceleration, batch processing, and query prefix for BGE models.
"""

import warnings
from typing import Optional

from sentence_transformers import SentenceTransformer

from src.config_loader import Settings, get_settings
from src.utils.device_utils import is_accelerated_device, resolve_torch_device

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class EmbeddingService:
    """Embedding service using BGE models.

    Wraps sentence-transformers SentenceTransformer with GPU support,
    batch processing, and BGE-specific query prefix handling.
    """

    # BGE query prefix for better retrieval performance
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages:"

    def __init__(
        self,
        model_name: Optional[str] = None,
        settings: Optional[Settings] = None,
        device: Optional[str] = None,
    ):
        """Initialize embedding service.

        Args:
            model_name: HuggingFace model name. If None, uses default from config.
            settings: Settings instance. If None, loads from config.
            device: Device to use ('dml', 'cuda', 'cpu', or None for auto-detect)
        """
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.models.embedding.default
        self.batch_size = self.settings.performance.embed_batch_size

        # Force CPU for embeddings - DirectML causes "version_counter for inference tensor" errors
        # sentence-transformers has issues with DirectML. CPU is fast enough for embeddings.
        if device is None:
            # Auto-detect device
            self.device, self.device_label = resolve_torch_device(None)
        else:
            self.device, self.device_label = resolve_torch_device(device)

        # Load model
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model.

        Loads the SentenceTransformer model on initialization.
        """
        if self.model is None:
            try:
                # Try loading from local cache first (offline mode)
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    local_files_only=True
                )
            except Exception:
                # Fallback to standard load (online check/download)
                try:
                    self.model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                    )
                except Exception as e:
                    if self.device != "cpu":
                        print(
                            f"Warning: Failed to load embedding model on {self.device_label}, falling back to CPU"
                        )
                        print(f"Error: {str(e)}")
                        self.device = "cpu"
                        self.device_label = "cpu"
                        try:
                            # Try CPU with local files first
                            self.model = SentenceTransformer(
                                self.model_name,
                                device="cpu",
                                local_files_only=True
                            )
                            return
                        except Exception:
                            try:
                                self.model = SentenceTransformer(
                                    self.model_name,
                                    device="cpu",
                                )
                                return
                            except Exception as cpu_error:
                                raise RuntimeError(
                                    f"Failed to load embedding model {self.model_name} on CPU: {str(cpu_error)}"
                                ) from cpu_error
                    raise RuntimeError(
                        f"Failed to load embedding model {self.model_name}: {str(e)}"
                    ) from e

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats (1024 dimensions for bge-large-en-v1.5)
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter empty texts
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            raise ValueError("No valid texts to embed")

        embeddings = self.model.encode(
            valid_texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a query string with BGE query prefix.

        BGE models perform better when queries are prefixed with a specific
        instruction. This method adds the prefix automatically.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        prefixed_query = f"{self.BGE_QUERY_PREFIX} {query}"

        embedding = self.model.encode(
            prefixed_query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embedding.tolist()

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension (e.g., 1024 for bge-large-en-v1.5)
        """
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            return self.model.get_sentence_embedding_dimension()

        # Fallback: embed a test string and check dimension
        test_embedding = self.embed_text("test")
        return len(test_embedding)

    def is_gpu_available(self) -> bool:
        """Check if GPU is available and being used.

        Returns:
            True if GPU is available and in use
        """
        return is_accelerated_device(self.device_label)

