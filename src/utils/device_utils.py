"""Utilities for selecting torch devices with multi-backend GPU support.

Supported backends (in priority order):
1. CUDA - NVIDIA GPUs (native)
2. ROCm/HIP - AMD GPUs on Linux (via PyTorch ROCm build)
3. DirectML - AMD/Intel/NVIDIA GPUs on Windows/WSL2 (via torch-directml)
4. SYCL - Intel/AMD GPUs via oneAPI (future, scaffolded)
5. CPU - Fallback for all systems

For WSL2 with AMD GPUs:
- Native ROCm requires /dev/kfd which is not available in WSL2
- DirectML uses /dev/dxg (DirectX passthrough) which IS available
- DirectML provides ~80-90% of native ROCm performance
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Any, Tuple, Optional

import torch


DeviceReturn = Tuple[Any, str]


@dataclass
class GPUBackendInfo:
    """Information about an available GPU backend."""
    name: str
    device_string: str
    gpu_name: Optional[str]
    is_available: bool
    priority: int  # Lower is higher priority
    notes: str = ""


def _device_label(device: Any) -> str:
    """Return a lowercase label for torch devices/strings."""
    if isinstance(device, str):
        return device.lower()
    try:
        return str(device).lower()
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _get_cuda_gpu_info() -> str | None:
    """Get CUDA/ROCm GPU name if available."""
    if torch.cuda.is_available():
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            pass
    return None


def _check_directml() -> tuple[bool, str | None]:
    """Check if DirectML is available and get device name.

    DirectML is the recommended backend for AMD GPUs in WSL2 since
    native ROCm requires /dev/kfd which isn't exposed in WSL2.

    Returns:
        Tuple of (is_available, device_name or None)
    """
    try:
        import torch_directml
        device_count = torch_directml.device_count()
        if device_count > 0:
            device_name = torch_directml.device_name(0)
            return True, device_name
        return False, None
    except ImportError:
        return False, None
    except Exception:
        return False, None


def _check_sycl() -> tuple[bool, str | None]:
    """Check if SYCL/oneAPI backend is available (scaffolded for future).

    SYCL via Intel oneAPI can target:
    - Intel GPUs (native)
    - AMD GPUs (via HIP backend)
    - NVIDIA GPUs (via CUDA backend)

    This is scaffolded for future implementation when intel-extension-for-pytorch
    matures for cross-vendor support.

    Returns:
        Tuple of (is_available, device_name or None)
    """
    # SYCL support is scaffolded but not yet implemented
    # Future implementation would use intel_extension_for_pytorch
    try:
        # Placeholder for future SYCL detection
        # import intel_extension_for_pytorch as ipex
        # if ipex.xpu.is_available():
        #     return True, ipex.xpu.get_device_name(0)
        return False, None
    except ImportError:
        return False, None
    except Exception:
        return False, None


def _is_wsl() -> bool:
    """Check if running in Windows Subsystem for Linux."""
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower() or 'wsl' in f.read().lower()
    except Exception:
        return False


def _is_amd_gpu(gpu_name: str | None) -> bool:
    """Check if GPU name indicates an AMD GPU."""
    if not gpu_name:
        return False
    gpu_lower = gpu_name.lower()
    return 'amd' in gpu_lower or 'radeon' in gpu_lower or gpu_name.startswith('gfx')


def get_available_backends() -> list[GPUBackendInfo]:
    """Detect all available GPU backends and return sorted by priority.

    Returns:
        List of GPUBackendInfo sorted by priority (best first)
    """
    backends = []

    # Check CUDA/ROCm (priority 1 for NVIDIA, 2 for AMD ROCm on native Linux)
    if torch.cuda.is_available():
        gpu_name = _get_cuda_gpu_info()
        is_amd = _is_amd_gpu(gpu_name)

        if is_amd:
            backends.append(GPUBackendInfo(
                name="ROCm",
                device_string="cuda",  # ROCm uses cuda API in PyTorch
                gpu_name=gpu_name,
                is_available=True,
                priority=2,
                notes="AMD GPU via ROCm/HIP (native Linux)"
            ))
        else:
            backends.append(GPUBackendInfo(
                name="CUDA",
                device_string="cuda",
                gpu_name=gpu_name,
                is_available=True,
                priority=1,
                notes="NVIDIA GPU via CUDA"
            ))

    # Check DirectML (priority 3 - good for WSL2 with AMD GPUs)
    dml_available, dml_name = _check_directml()
    if dml_available:
        backends.append(GPUBackendInfo(
            name="DirectML",
            device_string="privateuseone:0",  # torch-directml device string
            gpu_name=dml_name,
            is_available=True,
            priority=3,
            notes="GPU via DirectML (Windows/WSL2)"
        ))

    # Check SYCL (priority 4 - scaffolded for future)
    sycl_available, sycl_name = _check_sycl()
    if sycl_available:
        backends.append(GPUBackendInfo(
            name="SYCL",
            device_string="xpu",  # Intel extension device string
            gpu_name=sycl_name,
            is_available=True,
            priority=4,
            notes="GPU via Intel oneAPI/SYCL"
        ))

    # CPU is always available (lowest priority)
    backends.append(GPUBackendInfo(
        name="CPU",
        device_string="cpu",
        gpu_name=None,
        is_available=True,
        priority=100,
        notes="CPU fallback"
    ))

    # Sort by priority
    backends.sort(key=lambda x: x.priority)
    return backends


def resolve_torch_device(
    requested_device: Any | None,
) -> DeviceReturn:
    """Resolve the torch device to use along with a human-friendly label.

    Supports multiple backends in priority order:
    1. CUDA (NVIDIA)
    2. ROCm/HIP (AMD on native Linux)
    3. DirectML (AMD/Intel/NVIDIA on Windows/WSL2)
    4. SYCL (Intel oneAPI - scaffolded)
    5. CPU (fallback)

    Args:
        requested_device: Explicit device string/torch.device, or None for auto.

    Returns:
        Tuple of (device_for_torch, label_string)
    """
    # If explicitly requested, use that
    if requested_device is not None:
        label = _device_label(requested_device)
        # Handle DirectML special case
        if label == "dml" or label == "directml":
            try:
                import torch_directml
                dml_device = torch_directml.device()
                dml_name = torch_directml.device_name(0)
                return dml_device, f"directml ({dml_name})"
            except Exception:
                pass
        return requested_device, label

    # Auto-detect best available backend
    backends = get_available_backends()

    for backend in backends:
        if not backend.is_available or backend.name == "CPU":
            continue

        if backend.name == "DirectML":
            try:
                import torch_directml
                dml_device = torch_directml.device()
                return dml_device, f"directml ({backend.gpu_name})"
            except Exception:
                continue

        elif backend.name in ("CUDA", "ROCm"):
            name_lower = backend.name.lower()
            gpu_info = f" ({backend.gpu_name})" if backend.gpu_name else ""
            return "cuda", f"{name_lower}{gpu_info}"

        elif backend.name == "SYCL":
            # Scaffolded for future
            return "xpu", f"sycl ({backend.gpu_name})"

    return "cpu", "cpu"


def is_accelerated_device(label: str | None) -> bool:
    """Return True if the label indicates GPU acceleration.

    Supports CUDA, ROCm, DirectML, and SYCL backends.
    """
    if not label:
        return False
    label = label.lower()
    return (
        label.startswith("cuda") or
        label.startswith("rocm") or
        label.startswith("directml") or
        label.startswith("sycl") or
        label.startswith("xpu") or
        label.startswith("privateuseone")  # DirectML internal device name
    )


def get_device_summary() -> dict:
    """Get a summary of the current device configuration.

    Returns:
        Dictionary with device information for diagnostics/UI display
    """
    backends = get_available_backends()
    device, label = resolve_torch_device(None)

    # Get active backend info
    active_backend = None
    for b in backends:
        if b.name.lower() in label.lower() or b.device_string in str(device):
            active_backend = b
            break

    # Check for known compatibility issues
    compatibility_notes = []
    if active_backend and active_backend.name == "DirectML":
        compatibility_notes.append(
            "DirectML has limited compatibility with sentence-transformers. "
            "Embeddings may run on CPU while LLM inference uses GPU."
        )

    return {
        "active_device": str(device),
        "active_label": label,
        "active_backend": active_backend.name if active_backend else "Unknown",
        "gpu_name": active_backend.gpu_name if active_backend else None,
        "is_accelerated": is_accelerated_device(label),
        "available_backends": [
            {
                "name": b.name,
                "gpu_name": b.gpu_name,
                "priority": b.priority,
                "notes": b.notes
            }
            for b in backends if b.is_available
        ],
        "platform": platform.system(),
        "is_wsl": _is_wsl(),
        "compatibility_notes": compatibility_notes,
    }


def resolve_device_for_library(library: str, requested_device: Any | None = None) -> DeviceReturn:
    """Resolve device for a specific library, handling compatibility issues.

    Some libraries (like sentence-transformers) have compatibility issues with
    certain backends (like DirectML). This function provides library-specific
    device resolution.

    Args:
        library: Library name ('sentence_transformers', 'transformers', 'llama_cpp')
        requested_device: Explicit device or None for auto

    Returns:
        Tuple of (device_for_torch, label_string)
    """
    if requested_device is not None:
        return resolve_torch_device(requested_device)

    device, label = resolve_torch_device(None)

    # sentence-transformers has issues with DirectML's version_counter
    # Fall back to CPU for reliability
    if library == "sentence_transformers" and "directml" in label.lower():
        return "cpu", "cpu (DirectML incompatible with sentence-transformers)"

    return device, label

