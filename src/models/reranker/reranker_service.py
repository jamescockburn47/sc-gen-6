"""Cross-encoder reranker service using mixedbread-ai models.

Provides reranking of query-chunk pairs using cross-encoder models.
Supports GPU acceleration via CUDA (NVIDIA) or ROCm/HIP (AMD).
"""

import warnings
from typing import Optional

import torch
from sentence_transformers import CrossEncoder

from src.config_loader import Settings, get_settings
from src.utils.device_utils import is_accelerated_device, resolve_torch_device

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*were not initialized from the model checkpoint.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")
warnings.filterwarnings("ignore", message=".*some weights of the model checkpoint at.*")
warnings.filterwarnings("ignore", message=".*score.weight.*")


class RerankerService:
    """Cross-encoder reranker for query-chunk pairs.

    Uses mixedbread-ai reranker models to score query-chunk relevance.
    Automatically uses GPU (CUDA/ROCm) when available for faster inference.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        settings: Optional[Settings] = None,
        device: Optional[str] = None,
    ):
        """Initialize reranker service.

        Args:
            model_name: HuggingFace model name. If None, uses default from config.
            settings: Settings instance. If None, loads from config.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
                    Note: ROCm/HIP uses 'cuda' as device string.
        """
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.models.reranker.default
        self.batch_size = self.settings.performance.rerank_batch_size

        # Force CPU for reranker - DirectML is unstable and causes GPU device suspension
        # The reranker is fast enough on CPU and this prevents crashes
        if device is None:
            self.device = "cpu"
            self.device_label = "cpu (forced - DirectML unstable)"
        else:
            self.device, self.device_label = resolve_torch_device(device)

        # Load model
        self.model: Optional[CrossEncoder] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the reranker model on the appropriate device.

        CrossEncoder from sentence-transformers wraps a HuggingFace AutoModelForSequenceClassification.
        We need to explicitly move it to GPU and ensure it stays there during inference.
        """
        if self.model is None:
            try:
                # Load model - CrossEncoder accepts device parameter
                # Use model_kwargs for dtype (newer API)
                model_kwargs = {"dtype": torch.float16} if self.device != "cpu" else {}
                
                # Try loading from local cache first (offline mode)
                try:
                    self.model = CrossEncoder(
                        self.model_name,
                        device=self.device,
                        trust_remote_code=True,
                        model_kwargs=model_kwargs,
                        local_files_only=True,
                    )
                except Exception:
                    # Fallback to standard load
                    self.model = CrossEncoder(
                        self.model_name,
                        device=self.device,
                        trust_remote_code=True,
                        model_kwargs=model_kwargs,
                    )

                # Verify model is on correct device
                if self.device != "cpu" and hasattr(self.model, 'model'):
                    actual_device = next(self.model.model.parameters()).device
                    if actual_device.type == "cpu":
                        # Force move to GPU if it loaded on CPU
                        self.model.model = self.model.model.to(self.device)
                        print(f"Reranker moved to {self.device_label}")
                    else:
                        print(f"Reranker loaded on {self.device_label} ({actual_device})")

            except Exception as e:
                # If GPU fails, fall back to CPU
                if self.device != "cpu":
                    print(f"Warning: Failed to load reranker on {self.device_label}, falling back to CPU")
                    print(f"Error: {str(e)}")
                    self.device = "cpu"
                    self.device_label = "cpu"
                    try:
                        # Try CPU with local files first
                        self.model = CrossEncoder(
                            self.model_name,
                            device="cpu",
                            trust_remote_code=True,
                            local_files_only=True
                        )
                    except Exception:
                        try:
                            self.model = CrossEncoder(
                                self.model_name,
                                device="cpu",
                                trust_remote_code=True,
                            )
                        except Exception as cpu_error:
                            raise RuntimeError(
                                f"Failed to load reranker model {self.model_name} on CPU: {str(cpu_error)}"
                            ) from cpu_error
                else:
                    raise RuntimeError(
                        f"Failed to load reranker model {self.model_name}: {str(e)}"
                    ) from e

    def rerank(
        self, query: str, chunks: list[tuple[str, str]], top_k: Optional[int] = None
    ) -> list[tuple[str, float]]:
        """Rerank chunks for a query.

        Args:
            query: Query text
            chunks: List of tuples (chunk_id, chunk_text)
            top_k: Optional number of top results to return. If None, returns all.

        Returns:
            List of tuples (chunk_id, score) sorted by score descending.
            Scores are typically in range [0, 1] or [-1, 1] depending on model.
        """
        if not chunks:
            return []

        assert self.model is not None, "Model should be loaded in __init__"

        pairs = [(query, chunk_text) for _, chunk_text in chunks]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Combine with chunk IDs and sort
        results = list(zip([chunk_id for chunk_id, _ in chunks], scores.tolist()))
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top_k if specified
        if top_k is not None:
            results = results[:top_k]

        return results

    def is_gpu_available(self) -> bool:
        """Check if GPU is available and being used.

        Returns:
            True if GPU is available and in use
        """
        return is_accelerated_device(self.device_label)
