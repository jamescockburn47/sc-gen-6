"""Hardware and locality checks for ingestion."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

from src.config.llm_config import load_llm_config
from src.config_loader import Settings, get_settings


@dataclass
class IngestionEnvironment:
    """Snapshot of local hardware and policy relevant to ingestion."""

    has_gpu: bool
    gpu_name: Optional[str]
    total_vram_gb: Optional[float]
    cpu_threads: int
    local_only: bool
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def verify_local_ingestion(settings: Optional[Settings] = None) -> IngestionEnvironment:
    """Assess whether ingestion can run using only local compute resources."""

    settings = settings or get_settings()
    llm_config = load_llm_config()

    has_gpu, gpu_name, total_vram_gb = _detect_gpu()
    cpu_threads = _cpu_threads()

    warnings: list[str] = []
    notes: list[str] = []

    if has_gpu:
        notes.append(
            f"GPU detected: {gpu_name or 'Unknown'} ({total_vram_gb or '?.?'} GB VRAM)"
        )
    else:
        warnings.append(
            "No compatible GPU detected. Ingestion will run on CPU and may be slower."
        )

    notes.append(f"CPU threads available: {cpu_threads}")

    local_only = _validate_local_environment(warnings)

    # Sanity check for LLM backend (ingestion must remain local regardless)
    backend = llm_config.provider
    if backend not in {"lmstudio", "llama_cpp", "ollama"}:
        warnings.append(
            f"LLM provider '{backend}' is not recognized as a local provider. "
            "Ensure ingestion components do not call remote services."
        )

    return IngestionEnvironment(
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        total_vram_gb=total_vram_gb,
        cpu_threads=cpu_threads,
        local_only=local_only,
        warnings=warnings,
        notes=notes,
    )


# ------------------------------------------------------------------#
# Helpers
# ------------------------------------------------------------------#
def _detect_gpu() -> tuple[bool, Optional[str], Optional[float]]:
    """Detect GPU - supports both CUDA (NVIDIA) and ROCm (AMD)."""
    if torch is None:
        return False, None, None

    try:
        # PyTorch uses cuda API for both NVIDIA CUDA and AMD ROCm
        if torch.cuda.is_available():
            device_index = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(device_index)
            props = torch.cuda.get_device_properties(device_index)
            total_vram_gb = round(props.total_memory / (1024**3), 2)
            return True, gpu_name, total_vram_gb
    except Exception:
        pass

    # Fallback: Check for ROCm via hip module (AMD GPUs)
    try:
        if hasattr(torch, 'hip') or 'rocm' in torch.__config__.show().lower():
            # ROCm is available but device may not be detected yet
            return False, "ROCm available (device not initialized)", None
    except Exception:
        pass

    return False, None, None


def _cpu_threads() -> int:
    if psutil:
        try:
            count = psutil.cpu_count(logical=True)
            if count:
                return count
        except Exception:
            pass
    return os.cpu_count() or 1


def _validate_local_environment(warnings: list[str]) -> bool:
    """Ensure no proxy/remote configuration is in place."""
    local_only = True
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        if os.environ.get(var):
            warnings.append(
                f"Environment variable {var} is set. Disable proxies to keep ingestion local."
            )
            local_only = False
    return local_only

