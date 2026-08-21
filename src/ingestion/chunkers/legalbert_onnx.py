"""LegalBERT encoder for semantic chunking.

Uses PyTorch with GPU acceleration if available, falling back to CPU.
LegalBERT (~110M params) is small enough that CPU inference is fast for
the sentence-encoding workload in the semantic chunker.

Provides batched encoding for dramatic speedup over sentence-by-sentence.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional


class LegalBERTEncoder:
    """GPU-accelerated LegalBERT encoder with automatic CPU fallback.

    Singleton pattern — only one instance is created and cached.
    """

    LEGAL_BERT_MODEL = "nlpaueb/legal-bert-base-uncased"

    _instance: Optional[LegalBERTEncoder] = None

    def __new__(cls) -> LegalBERTEncoder:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.model = None
        self.tokenizer = None
        self.device = None
        self.device_label: str = "not loaded"
        self._initialized = True

    def _load_model(self) -> None:
        """Load model and tokenizer.

        Forces CPU to avoid ROCm segfaults on gfx1151.  LegalBERT is
        only ~110M params so CPU inference is fast enough for the
        sentence-encoding workload in the semantic chunker.
        """
        import os
        # Prevent PyTorch from touching ROCm/CUDA — avoids segfault on gfx1151
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

        import torch
        from transformers import AutoModel, AutoTokenizer

        print("[LegalBERT] Loading model (CPU-only for stability)...")

        self.device = torch.device("cpu")
        self.device_label = "CPU"

        print(f"[LegalBERT] Using device: {self.device_label}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.LEGAL_BERT_MODEL)
        self.model = AutoModel.from_pretrained(
            self.LEGAL_BERT_MODEL, use_safetensors=True,
        )
        self.model.eval()

        print(f"[LegalBERT] Model ready on {self.device_label}")

    def _ensure_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            self._load_model()

    def encode_sentences(self, sentences: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode sentences to embeddings with batched inference.

        Args:
            sentences: List of sentences to encode.
            batch_size: Batch size for inference.

        Returns:
            numpy array of shape ``(len(sentences), 768)`` with mean-pooled
            embeddings.
        """
        if not sentences:
            return np.array([])

        import torch

        self._ensure_loaded()

        all_embeddings: list[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i : i + batch_size]

                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Mean pooling
                token_embs = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embs = torch.sum(token_embs * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_embs / sum_mask

                all_embeddings.append(pooled.float().cpu().numpy())

        return np.vstack(all_embeddings)

    def is_gpu_available(self) -> bool:
        """Check if GPU is being used."""
        return self.device is not None and self.device.type == "cuda"


# Legacy alias for existing imports
ROCmLegalBERT = LegalBERTEncoder


def get_legalbert_encoder() -> LegalBERTEncoder:
    """Get the singleton LegalBERT encoder instance."""
    return LegalBERTEncoder()
