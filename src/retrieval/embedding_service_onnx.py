"""GPU-accelerated embedding service using ONNX Runtime with DirectML.

Bypasses sentence-transformers' DirectML issues by using ONNX Runtime directly.
Supports AMD, Intel, and NVIDIA GPUs on Windows via DirectML.
"""

import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from src.config_loader import Settings, get_settings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress ONNX Runtime verbose warnings about node assignments
# These are informational and don't affect functionality
ort.set_default_logger_severity(3)  # ERROR level only


class ONNXEmbeddingService:
    """GPU-accelerated embedding service using ONNX Runtime + DirectML.

    This implementation:
    1. Uses ONNX Runtime with DirectML for true GPU acceleration on AMD GPUs
    2. Exports BGE model to ONNX format on first use (cached)
    3. Provides same interface as EmbeddingService for drop-in replacement
    """

    # BGE query prefix for better retrieval performance
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages:"

    # ONNX model cache directory
    ONNX_CACHE_DIR = Path("data/onnx_models")

    def __init__(
        self,
        model_name: Optional[str] = None,
        settings: Optional[Settings] = None,
        device: Optional[str] = None,  # Ignored - always uses DirectML if available
    ):
        """Initialize ONNX embedding service.

        Args:
            model_name: HuggingFace model name. If None, uses default from config.
            settings: Settings instance. If None, loads from config.
            device: Ignored - automatically uses DirectML GPU if available.
        """
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.models.embedding.default
        self.batch_size = self.settings.performance.embed_batch_size

        # Setup ONNX cache
        self.ONNX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Determine execution provider
        self.provider, self.device_label = self._get_best_provider()

        # Load tokenizer and ONNX model
        self.tokenizer = None
        self.session = None
        self._embedding_dim = None
        # Lazy load: self._load_model() is now called on first use

    def _get_best_provider(self) -> tuple[str, str]:
        """Determine the best ONNX Runtime execution provider.

        Returns:
            Tuple of (provider_name, device_label)
        """
        available = ort.get_available_providers()
        print(f"[ONNX] Available providers: {available}")

        # Priority order: DirectML > CUDA > CPU
        if "DmlExecutionProvider" in available:
            try:
                # Test DirectML
                import torch_directml
                gpu_name = torch_directml.device_name(0)
                return "DmlExecutionProvider", f"DirectML GPU ({gpu_name})"
            except Exception:
                return "DmlExecutionProvider", "DirectML GPU"

        if "CUDAExecutionProvider" in available:
            return "CUDAExecutionProvider", "CUDA GPU"

        return "CPUExecutionProvider", "CPU"

    def _get_onnx_path(self) -> Path:
        """Get the path for the ONNX model cache."""
        # Sanitize model name for filesystem
        safe_name = self.model_name.replace("/", "_").replace("\\", "_")
        return self.ONNX_CACHE_DIR / f"{safe_name}.onnx"

    def _export_to_onnx(self) -> Path:
        """Export the HuggingFace model to ONNX format.

        Returns:
            Path to the exported ONNX model
        """
        from transformers import AutoModel
        import torch

        onnx_path = self._get_onnx_path()

        print(f"[ONNX] Exporting {self.model_name} to ONNX format...")
        print(f"[ONNX] This is a one-time operation. Model will be cached at: {onnx_path}")

        # Load the model
        model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        model.eval()

        # Create dummy inputs
        dummy_input_ids = torch.ones(1, 512, dtype=torch.long)
        dummy_attention_mask = torch.ones(1, 512, dtype=torch.long)
        dummy_token_type_ids = torch.zeros(1, 512, dtype=torch.long)

        # Check if model uses token_type_ids
        try:
            model(dummy_input_ids, dummy_attention_mask, dummy_token_type_ids)
            use_token_type_ids = True
        except TypeError:
            use_token_type_ids = False

        # Export to ONNX
        dynamic_axes = {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        }

        input_names = ["input_ids", "attention_mask"]
        inputs = (dummy_input_ids, dummy_attention_mask)

        if use_token_type_ids:
            input_names.append("token_type_ids")
            inputs = (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids)
            dynamic_axes["token_type_ids"] = {0: "batch", 1: "sequence"}

        torch.onnx.export(
            model,
            inputs,
            str(onnx_path),
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
        )

        print(f"[ONNX] Export complete: {onnx_path}")
        return onnx_path

    def _load_model(self) -> None:
        """Load the ONNX model and tokenizer."""
        # Load tokenizer
        print(f"[ONNX] Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Check for cached ONNX model
        onnx_path = self._get_onnx_path()
        if not onnx_path.exists():
            onnx_path = self._export_to_onnx()

        # Create ONNX Runtime session with DirectML
        print(f"[ONNX] Loading ONNX model with {self.provider}...")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Enable memory pattern optimization
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        try:
            self.session = ort.InferenceSession(
                str(onnx_path),
                sess_options,
                providers=[self.provider],
            )
            # Verify which provider is actually being used
            actual_providers = self.session.get_providers()
            if self.provider in actual_providers:
                print(f"[ONNX] Model loaded on {self.device_label}")
            else:
                print(f"[ONNX] Requested {self.provider} but using {actual_providers}")
                if "DmlExecutionProvider" in actual_providers:
                    self.device_label = "DirectML GPU"
                elif "CUDAExecutionProvider" in actual_providers:
                    self.device_label = "CUDA GPU"
                else:
                    self.device_label = "CPU"
        except Exception as e:
            print(f"[ONNX] Warning: Failed to load with {self.provider}: {e}")
            print("[ONNX] Falling back to CPU...")
            self.provider = "CPUExecutionProvider"
            self.device_label = "CPU (fallback)"
            self.session = ort.InferenceSession(
                str(onnx_path),
                sess_options,
                providers=["CPUExecutionProvider"],
            )

        # Get embedding dimension from model config
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_name)
            self._embedding_dim = config.hidden_size
        except Exception:
            self._embedding_dim = 1024  # Default for bge-large

    def _mean_pooling(
        self, token_embeddings: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        """Apply mean pooling to token embeddings.

        Args:
            token_embeddings: Shape (batch, seq_len, hidden_size)
            attention_mask: Shape (batch, seq_len)

        Returns:
            Pooled embeddings of shape (batch, hidden_size)
        """
        # Expand attention mask for broadcasting
        mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)

        # Sum embeddings weighted by mask
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)

        # Sum mask for normalization
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)

        return sum_embeddings / sum_mask

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, a_min=1e-9, a_max=None)

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings using ONNX Runtime.

        Args:
            texts: List of texts to encode

        Returns:
            Normalized embeddings of shape (len(texts), hidden_size)
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        # Prepare inputs
        ort_inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }

        # Add token_type_ids if model expects it
        input_names = [inp.name for inp in self.session.get_inputs()]
        if "token_type_ids" in input_names:
            if "token_type_ids" in encoded:
                ort_inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)
            else:
                ort_inputs["token_type_ids"] = np.zeros_like(encoded["input_ids"])

        # Run inference
        outputs = self.session.run(None, ort_inputs)
        token_embeddings = outputs[0]  # Shape: (batch, seq_len, hidden_size)

        # Mean pooling
        pooled = self._mean_pooling(token_embeddings, encoded["attention_mask"])

        # Normalize
        normalized = self._normalize(pooled)

        return normalized

    def _ensure_model_loaded(self) -> None:
        """Ensure model and tokenizer are loaded."""
        if self.session is None or self.tokenizer is None:
            self._load_model()

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        self._ensure_model_loaded()
        embeddings = self._encode([text])
        return embeddings[0].tolist()

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

        # Process in batches
        self._ensure_model_loaded()
        all_embeddings = []
        for i in range(0, len(valid_texts), self.batch_size):
            batch = valid_texts[i : i + self.batch_size]
            embeddings = self._encode(batch)
            all_embeddings.extend(embeddings.tolist())

        return all_embeddings

    def embed_query(self, query: str) -> list[float]:
        """Embed a query string with BGE query prefix.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not query.strip():
            raise ValueError("Query cannot be empty")

        prefixed_query = f"{self.BGE_QUERY_PREFIX} {query}"
        self._ensure_model_loaded()
        embeddings = self._encode([prefixed_query])
        return embeddings[0].tolist()

        return self._embedding_dim

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model.

        Returns:
            Embedding dimension (e.g., 1024 for bge-large-en-v1.5)
        """
        if self._embedding_dim is None:
            # Try to get from config without loading full model if possible
            try:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(self.model_name)
                self._embedding_dim = config.hidden_size
            except Exception:
                self._ensure_model_loaded()
                
        return self._embedding_dim

    def is_gpu_available(self) -> bool:
        """Check if GPU is available and being used.

        Returns:
            True if GPU is available and in use
        """
        return "GPU" in self.device_label or "DirectML" in self.device_label

