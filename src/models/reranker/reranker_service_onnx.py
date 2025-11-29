"""GPU-accelerated reranker service using ONNX Runtime with DirectML.

Bypasses sentence-transformers CrossEncoder's DirectML issues by using ONNX Runtime directly.
Supports AMD, Intel, and NVIDIA GPUs on Windows via DirectML.
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from src.config_loader import Settings, get_settings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*were not initialized from the model checkpoint.*")
warnings.filterwarnings("ignore", message=".*You should probably TRAIN this model.*")
warnings.filterwarnings("ignore", message=".*TracerWarning.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress ONNX Runtime verbose warnings
ort.set_default_logger_severity(3)  # ERROR level only


class ONNXRerankerService:
    """GPU-accelerated reranker service using ONNX Runtime + DirectML.

    This implementation:
    1. Uses ONNX Runtime with DirectML for true GPU acceleration on AMD GPUs
    2. Exports reranker model to ONNX format on first use (cached)
    3. Provides same interface as RerankerService for drop-in replacement
    """

    # ONNX model cache directory
    ONNX_CACHE_DIR = Path("data/onnx_models")

    def __init__(
        self,
        model_name: Optional[str] = None,
        settings: Optional[Settings] = None,
        device: Optional[str] = None,  # Ignored - always uses DirectML if available
    ):
        """Initialize ONNX reranker service.

        Args:
            model_name: HuggingFace model name. If None, uses default from config.
            settings: Settings instance. If None, loads from config.
            device: Ignored - automatically uses DirectML GPU if available.
        """
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.models.reranker.default
        self.batch_size = self.settings.performance.rerank_batch_size

        # Setup ONNX cache
        self.ONNX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Determine execution provider
        self.provider, self.device_label = self._get_best_provider()

        # Load tokenizer and ONNX model
        self.tokenizer = None
        self.session = None
        self._load_model()

    def _get_best_provider(self) -> tuple[str, str]:
        """Determine the best ONNX Runtime execution provider.

        Returns:
            Tuple of (provider_name, device_label)
        """
        available = ort.get_available_providers()

        # Priority order: DirectML > CUDA > CPU
        if "DmlExecutionProvider" in available:
            try:
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
        safe_name = self.model_name.replace("/", "_").replace("\\", "_")
        return self.ONNX_CACHE_DIR / f"{safe_name}_reranker.onnx"
    
    def _get_optimum_onnx_path(self) -> Path:
        """Get the path for Optimum-exported ONNX model."""
        safe_name = self.model_name.replace("/", "_").replace("\\", "_")
        return self.ONNX_CACHE_DIR / f"{safe_name}_reranker" / "model.onnx"

    def _export_to_onnx(self) -> Path:
        """Export the HuggingFace reranker model to ONNX format.

        Uses Hugging Face Optimum for robust export of modern transformer architectures.

        Returns:
            Path to the exported ONNX model
        """
        onnx_path = self._get_onnx_path()
        onnx_dir = onnx_path.parent / onnx_path.stem

        print(f"[ONNX Reranker] Exporting {self.model_name} to ONNX format...")
        print(f"[ONNX Reranker] This is a one-time operation. Model will be cached.")

        try:
            # Try using Optimum for export (handles DynamicCache and other edge cases)
            from optimum.onnxruntime import ORTModelForSequenceClassification
            
            # Export using Optimum (handles complex models better)
            model = ORTModelForSequenceClassification.from_pretrained(
                self.model_name,
                export=True,
                trust_remote_code=True,
            )
            
            # Save to our cache directory
            onnx_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(onnx_dir))
            
            # Find the actual ONNX file
            actual_onnx = onnx_dir / "model.onnx"
            if actual_onnx.exists():
                print(f"[ONNX Reranker] Export complete: {actual_onnx}")
                return actual_onnx
            else:
                raise FileNotFoundError(f"ONNX file not found at {actual_onnx}")
                
        except Exception as e:
            print(f"[ONNX Reranker] Optimum export failed: {e}")
            print("[ONNX Reranker] Trying manual torch.onnx.export...")
            
            # Fallback to manual export for simpler models
            from transformers import AutoModelForSequenceClassification
            import torch

            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                use_cache=False,  # Disable cache to avoid DynamicCache issues
            )
            model.eval()

            # Create dummy inputs
            dummy_input_ids = torch.ones(1, 128, dtype=torch.long)
            dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)

            dynamic_axes = {
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            }

            torch.onnx.export(
                model,
                (dummy_input_ids, dummy_attention_mask),
                str(onnx_path),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=14,
                do_constant_folding=True,
            )

            print(f"[ONNX Reranker] Export complete: {onnx_path}")
            return onnx_path

    def _load_model(self) -> None:
        """Load the ONNX model and tokenizer."""
        # Load tokenizer
        print(f"[ONNX Reranker] Loading tokenizer for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Check for cached ONNX model (try both paths)
        onnx_path = self._get_onnx_path()
        optimum_path = self._get_optimum_onnx_path()
        
        if optimum_path.exists():
            onnx_path = optimum_path
        elif not onnx_path.exists():
            onnx_path = self._export_to_onnx()

        # Create ONNX Runtime session with DirectML
        print(f"[ONNX Reranker] Loading ONNX model with {self.provider}...")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True

        try:
            self.session = ort.InferenceSession(
                str(onnx_path),
                sess_options,
                providers=[self.provider],
            )
            actual_providers = self.session.get_providers()
            if self.provider in actual_providers:
                print(f"[ONNX Reranker] Model loaded on {self.device_label}")
            else:
                print(f"[ONNX Reranker] Requested {self.provider} but using {actual_providers}")
        except Exception as e:
            print(f"[ONNX Reranker] Warning: Failed to load with {self.provider}: {e}")
            print("[ONNX Reranker] Falling back to CPU...")
            self.provider = "CPUExecutionProvider"
            self.device_label = "CPU (fallback)"
            self.session = ort.InferenceSession(
                str(onnx_path),
                sess_options,
                providers=["CPUExecutionProvider"],
            )

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid function to convert logits to probabilities."""
        return 1 / (1 + np.exp(-x))

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
            Scores are in range [0, 1] (sigmoid of logits).
        """
        if not chunks:
            return []

        chunk_ids = [chunk_id for chunk_id, _ in chunks]
        chunk_texts = [chunk_text for _, chunk_text in chunks]

        # Process in batches
        all_scores = []
        for i in range(0, len(chunk_texts), self.batch_size):
            batch_texts = chunk_texts[i : i + self.batch_size]
            batch_queries = [query] * len(batch_texts)

            # Tokenize query-document pairs
            encoded = self.tokenizer(
                batch_queries,
                batch_texts,
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
            logits = outputs[0]  # Shape: (batch, num_labels) or (batch,)

            # Handle different output shapes
            if len(logits.shape) > 1 and logits.shape[1] > 1:
                # Multi-class: take the "relevant" class (usually index 1)
                scores = self._sigmoid(logits[:, -1])
            else:
                # Binary or single output
                scores = self._sigmoid(logits.flatten())

            all_scores.extend(scores.tolist())

        # Combine with chunk IDs and sort
        results = list(zip(chunk_ids, all_scores))
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
        return "GPU" in self.device_label or "DirectML" in self.device_label

