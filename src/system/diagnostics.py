"""System diagnostics for verifying GPU, ROCm, and model setup.

Provides comprehensive checks for hardware acceleration and model availability.
Designed to be called from the UI for user-friendly status reporting.
"""

from __future__ import annotations

import os
import sys
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch


@dataclass
class DiagnosticResult:
    """Result of a single diagnostic check."""

    name: str
    status: str  # "pass", "warn", "fail", "info"
    message: str
    details: Optional[str] = None


@dataclass
class SystemDiagnostics:
    """Complete system diagnostics report."""

    results: list[DiagnosticResult] = field(default_factory=list)
    summary: str = ""
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    pytorch_version: str = ""
    rocm_version: str = ""
    cuda_version: str = ""

    def add(self, result: DiagnosticResult) -> None:
        """Add a diagnostic result."""
        self.results.append(result)

    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.status == "pass")

    def warn_count(self) -> int:
        return sum(1 for r in self.results if r.status == "warn")

    def fail_count(self) -> int:
        return sum(1 for r in self.results if r.status == "fail")


def run_diagnostics() -> SystemDiagnostics:
    """Run all system diagnostics and return results.

    Returns:
        SystemDiagnostics with all check results
    """
    diag = SystemDiagnostics()

    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    diag.add(DiagnosticResult(
        name="Python Version",
        status="pass" if sys.version_info >= (3, 11) else "warn",
        message=f"Python {py_version}",
        details="Python 3.11+ recommended for best performance"
    ))

    # PyTorch version and build
    diag.pytorch_version = torch.__version__
    torch_build_info = []

    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        diag.gpu_available = True
        torch_build_info.append("CUDA/ROCm support")
    if hasattr(torch.version, 'hip') and torch.version.hip:
        diag.rocm_version = torch.version.hip
        torch_build_info.append(f"ROCm {torch.version.hip}")
    if hasattr(torch.version, 'cuda') and torch.version.cuda:
        diag.cuda_version = torch.version.cuda
        torch_build_info.append(f"CUDA {torch.version.cuda}")

    diag.add(DiagnosticResult(
        name="PyTorch",
        status="pass" if torch_build_info else "warn",
        message=f"PyTorch {torch.__version__}",
        details=", ".join(torch_build_info) if torch_build_info else "CPU only build"
    ))

    # GPU Detection
    if torch.cuda.is_available():
        try:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            diag.gpu_name = gpu_name

            # Get memory
            total_mem = torch.cuda.get_device_properties(0).total_memory
            diag.gpu_memory_gb = total_mem / (1024**3)

            # Detect AMD vs NVIDIA
            is_amd = any(x in gpu_name.lower() for x in ["amd", "radeon", "gfx"])

            diag.add(DiagnosticResult(
                name="GPU Detection",
                status="pass",
                message=f"{gpu_name}",
                details=f"{diag.gpu_memory_gb:.1f} GB VRAM, {gpu_count} device(s)"
            ))

            if is_amd:
                diag.add(DiagnosticResult(
                    name="AMD ROCm",
                    status="pass" if diag.rocm_version else "warn",
                    message=f"AMD GPU detected: {gpu_name}",
                    details=f"ROCm version: {diag.rocm_version or 'Not detected (using HIP compatibility)'}"
                ))
        except Exception as e:
            diag.add(DiagnosticResult(
                name="GPU Detection",
                status="fail",
                message="GPU detected but error getting details",
                details=str(e)
            ))
    else:
        diag.add(DiagnosticResult(
            name="GPU Detection",
            status="fail",
            message="No GPU detected by PyTorch",
            details="Install PyTorch with CUDA/ROCm support for GPU acceleration"
        ))

    # Test GPU tensor operations
    if diag.gpu_available:
        try:
            test_tensor = torch.randn(1000, 1000, device="cuda")
            result = torch.mm(test_tensor, test_tensor)
            del test_tensor, result
            torch.cuda.empty_cache()

            diag.add(DiagnosticResult(
                name="GPU Compute Test",
                status="pass",
                message="GPU tensor operations working",
                details="Matrix multiplication test passed"
            ))
        except Exception as e:
            diag.add(DiagnosticResult(
                name="GPU Compute Test",
                status="fail",
                message="GPU compute test failed",
                details=str(e)
            ))

    # Check sentence-transformers
    try:
        import sentence_transformers
        st_version = sentence_transformers.__version__
        diag.add(DiagnosticResult(
            name="Sentence Transformers",
            status="pass",
            message=f"sentence-transformers {st_version}",
            details="Embedding and reranker models available"
        ))
    except ImportError:
        diag.add(DiagnosticResult(
            name="Sentence Transformers",
            status="fail",
            message="sentence-transformers not installed",
            details="pip install sentence-transformers"
        ))

    # Check transformers (for cross-encoder)
    try:
        import transformers
        diag.add(DiagnosticResult(
            name="Transformers",
            status="pass",
            message=f"transformers {transformers.__version__}",
            details="HuggingFace transformers available"
        ))
    except ImportError:
        diag.add(DiagnosticResult(
            name="Transformers",
            status="fail",
            message="transformers not installed",
            details="pip install transformers"
        ))

    # Check ChromaDB
    try:
        import chromadb
        diag.add(DiagnosticResult(
            name="ChromaDB",
            status="pass",
            message=f"chromadb {chromadb.__version__}",
            details="Vector database available"
        ))
    except ImportError:
        diag.add(DiagnosticResult(
            name="ChromaDB",
            status="fail",
            message="chromadb not installed",
            details="pip install chromadb"
        ))

    # Check config profile
    try:
        from src.config_loader import get_config_profile, get_settings
        profile = get_config_profile()
        settings = get_settings()

        diag.add(DiagnosticResult(
            name="Config Profile",
            status="info",
            message=f"Profile: {profile}",
            details=f"Embed batch: {settings.performance.embed_batch_size}, "
                    f"Rerank batch: {settings.performance.rerank_batch_size}, "
                    f"Workers: {settings.performance.max_workers}"
        ))
    except Exception as e:
        diag.add(DiagnosticResult(
            name="Config Profile",
            status="warn",
            message="Could not load config",
            details=str(e)
        ))

    # Check llama.cpp server
    try:
        from src.config.runtime_store import load_runtime_state
        runtime = load_runtime_state()
        llama_cfg = runtime.get("llama_server", {})

        if llama_cfg.get("executable"):
            exe_path = Path(llama_cfg["executable"])
            exe_exists = exe_path.exists()
            model_path = llama_cfg.get("model_path", "")
            model_exists = Path(model_path).exists() if model_path else False

            diag.add(DiagnosticResult(
                name="llama.cpp Server",
                status="pass" if (exe_exists and model_exists) else "warn",
                message=f"Executable: {'✓' if exe_exists else '✗'}, Model: {'✓' if model_exists else '✗'}",
                details=f"Context: {llama_cfg.get('context', 'N/A')}, "
                        f"GPU layers: {llama_cfg.get('gpu_layers', 'N/A')}, "
                        f"Batch: {llama_cfg.get('batch', 'N/A')}"
            ))
        else:
            diag.add(DiagnosticResult(
                name="llama.cpp Server",
                status="info",
                message="Not configured",
                details="Set up in Settings > LLM Configuration"
            ))
    except Exception as e:
        diag.add(DiagnosticResult(
            name="llama.cpp Server",
            status="info",
            message="Could not check llama.cpp config",
            details=str(e)
        ))

    # Generate summary
    total = len(diag.results)
    passed = diag.pass_count()
    warned = diag.warn_count()
    failed = diag.fail_count()

    if failed == 0 and warned == 0:
        diag.summary = f"All {total} checks passed ✓"
    elif failed == 0:
        diag.summary = f"{passed}/{total} passed, {warned} warnings"
    else:
        diag.summary = f"{passed}/{total} passed, {failed} failed, {warned} warnings"

    return diag


def get_gpu_recommendations(diag: SystemDiagnostics) -> list[str]:
    """Get recommendations based on diagnostic results.

    Args:
        diag: SystemDiagnostics from run_diagnostics()

    Returns:
        List of recommendation strings
    """
    recommendations = []

    if not diag.gpu_available:
        recommendations.append(
            "⚠️ No GPU detected. Install PyTorch with ROCm support:\n"
            "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2"
        )

    if diag.gpu_memory_gb >= 64:
        recommendations.append(
            f"✓ {diag.gpu_memory_gb:.0f}GB VRAM detected - You can run 70B+ models!\n"
            "   Recommended: Qwen2.5-72B-Instruct-Q4_K_M or Llama-3.3-70B-Instruct-Q4_K_M"
        )
        recommendations.append(
            "💡 Set SC_CONFIG=96gb environment variable to use optimized settings"
        )

    if diag.gpu_memory_gb >= 32 and diag.gpu_memory_gb < 64:
        recommendations.append(
            f"✓ {diag.gpu_memory_gb:.0f}GB VRAM - Good for 32B models at higher quants\n"
            "   Recommended: Qwen2.5-32B-Instruct-Q6_K"
        )

    if "amd" in diag.gpu_name.lower() or "radeon" in diag.gpu_name.lower():
        recommendations.append(
            "🔧 AMD GPU detected - For best llama.cpp performance:\n"
            "   Build llama.cpp with: -DGGML_HIP_ROCWMMA_FATTN=ON"
        )

    return recommendations


def format_diagnostics_text(diag: SystemDiagnostics) -> str:
    """Format diagnostics as plain text for display.

    Args:
        diag: SystemDiagnostics from run_diagnostics()

    Returns:
        Formatted text string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("SYSTEM DIAGNOSTICS")
    lines.append("=" * 60)
    lines.append("")

    status_icons = {
        "pass": "✓",
        "warn": "⚠",
        "fail": "✗",
        "info": "ℹ"
    }

    for result in diag.results:
        icon = status_icons.get(result.status, "?")
        lines.append(f"{icon} {result.name}: {result.message}")
        if result.details:
            lines.append(f"    {result.details}")
        lines.append("")

    lines.append("-" * 60)
    lines.append(f"Summary: {diag.summary}")
    lines.append("-" * 60)

    recommendations = get_gpu_recommendations(diag)
    if recommendations:
        lines.append("")
        lines.append("RECOMMENDATIONS:")
        for rec in recommendations:
            lines.append(rec)
            lines.append("")

    return "\n".join(lines)
