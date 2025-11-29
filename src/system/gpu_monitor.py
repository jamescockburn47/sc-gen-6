"""Lightweight GPU usage monitor for Windows via performance counters with fallback."""

from __future__ import annotations

import json
import subprocess
import platform
from dataclasses import dataclass
from typing import List, Optional
import requests

# Try importing psutil for fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


COUNTER_USAGE = r"\\GPU Adapter Memory(*)\\Dedicated Usage"
COUNTER_LIMIT = r"\\GPU Adapter Memory(*)\\Dedicated Limit"
COUNTER_UTIL = r"\\GPU Engine(*)\\Utilization Percentage"


@dataclass
class GPUMetric:
    adapter: str
    usage_mb: float
    limit_mb: float
    utilization: float
    is_fallback: bool = False  # True if this is actually System RAM


@dataclass  
class OllamaGPUInfo:
    """GPU info from Ollama's ps endpoint."""
    model_name: str
    gpu_name: str
    vram_used_gb: float
    vram_total_gb: float
    is_loaded: bool


def get_ollama_gpu_info(host: str = "http://localhost:11434") -> Optional[OllamaGPUInfo]:
    """Query Ollama for GPU memory usage of loaded models.
    
    Returns info about the currently loaded model's GPU usage.
    """
    try:
        # Get running models
        response = requests.get(f"{host}/api/ps", timeout=2)
        if response.status_code != 200:
            return None
            
        data = response.json()
        models = data.get("models", [])
        
        if not models:
            # No model loaded - return available GPU info from /api/show
            return OllamaGPUInfo(
                model_name="(none loaded)",
                gpu_name="AMD Radeon 8060S (ROCm)",
                vram_used_gb=0,
                vram_total_gb=96.0,  # Your system's unified memory allocation
                is_loaded=False
            )
        
        # Get first loaded model's info
        model = models[0]
        model_name = model.get("name", "unknown")
        size_vram = model.get("size_vram", 0)  # bytes
        
        # Ollama reports size_vram in bytes
        vram_gb = size_vram / (1024**3) if size_vram else 0
        
        return OllamaGPUInfo(
            model_name=model_name,
            gpu_name="AMD Radeon 8060S (ROCm)",
            vram_used_gb=vram_gb,
            vram_total_gb=96.0,  # Your unified memory config
            is_loaded=True
        )
        
    except Exception:
        return None


def _query_counter(counter: str) -> list[dict]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-Counter -Counter '{ctr}' -ErrorAction SilentlyContinue | "
            "Select -ExpandProperty CounterSamples | "
            "Select Path,CookedValue | ConvertTo-Json -Compress"
        ).format(ctr=counter),
    ]
    try:
        # Increase timeout to 2s, but don't block UI too long
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        if not result.stdout.strip():
            return []
            
        data = json.loads(result.stdout)
        if isinstance(data, dict):
            return [data]
        return data or []
    except Exception:
        return []


def get_gpu_stats() -> List[GPUMetric]:
    """Get GPU statistics, falling back to System RAM if GPU counters fail."""
    
    # 1. Try Windows Performance Counters (Best for NVIDIA/AMD on Windows)
    try:
        usage_samples = _query_counter(COUNTER_USAGE)
        
        if usage_samples:
            limit_samples = _query_counter(COUNTER_LIMIT)
            util_samples = _query_counter(COUNTER_UTIL)

            usage_map = {_extract_instance(s["Path"]): s["CookedValue"] for s in usage_samples if "Path" in s}
            limit_map = {_extract_instance(s["Path"]): s["CookedValue"] for s in limit_samples if "Path" in s}

            util_map = {}
            for sample in util_samples:
                path = sample.get("Path", "")
                instance = _extract_instance(path)
                value = sample.get("CookedValue", 0.0)
                if instance:
                    util_map.setdefault(instance, []).append(value)

            metrics: list[GPUMetric] = []
            for adapter, usage in usage_map.items():
                limit = max(limit_map.get(adapter, 0.0), 1.0)
                util_list = util_map.get(adapter, [])
                avg_util = sum(util_list) / len(util_list) if util_list else 0.0
                
                # Filter out tiny allocations (system reserved)
                if usage > 100 * 1024 * 1024:  # > 100MB
                    metrics.append(
                        GPUMetric(
                            adapter=adapter,
                            usage_mb=usage / (1024 * 1024),
                            limit_mb=limit / (1024 * 1024),
                            utilization=avg_util,
                            is_fallback=False
                        )
                    )
            
            if metrics:
                return metrics

    except Exception:
        pass  # Fall through to fallback

    # 2. Fallback: System RAM (via psutil)
    # Useful for DirectML/iGPUs where VRAM is shared with System RAM
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        
        return [
            GPUMetric(
                adapter="System RAM (Shared)",
                usage_mb=mem.used / (1024 * 1024),
                limit_mb=mem.total / (1024 * 1024),
                utilization=cpu,
                is_fallback=True
            )
        ]

    return []


def _extract_instance(path: str) -> str:
    if "(" in path and ")" in path:
        start = path.find("(") + 1
        end = path.find(")", start)
        return path[start:end]
    return path
