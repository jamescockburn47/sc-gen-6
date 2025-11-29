"""Persistent runtime configuration for LLM provider management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


RUNTIME_FILE = Path("config/llm_runtime.json")

DEFAULT_STATE: Dict[str, Any] = {
    "provider": "ollama",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "model_name": "qwen2.5:32b",
    "llama_server": {
        "executable": "C:/path/to/llama-server.exe",
        "model_path": "C:/models/gpt-oss-20b.Q4_K_M.gguf",
        "context": 32768,
        "gpu_layers": 999,
        "parallel": 2,
        "batch": 1024,
        "timeout": 1800,
        "host": "127.0.0.1",
        "port": 8000,
        "flash_attn": False,
        "extra_args": "",
    },
}


def load_runtime_state() -> Dict[str, Any]:
    """Load runtime state from disk, falling back to defaults."""
    if RUNTIME_FILE.exists():
        try:
            with RUNTIME_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return _merge_dict(DEFAULT_STATE, data)
        except (json.JSONDecodeError, OSError):
            return DEFAULT_STATE.copy()
    return DEFAULT_STATE.copy()


def save_runtime_state(state: Dict[str, Any]) -> None:
    """Persist runtime state to disk."""
    RUNTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged = _merge_dict(DEFAULT_STATE, state)
    with RUNTIME_FILE.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result



