"""Model presets management for llama.cpp."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from src.config.llm_config import load_llm_config
from src.config.runtime_store import load_runtime_state, save_runtime_state
from src.llm.server_manager import manager as llama_manager

PRESETS_FILE = Path("config/model_presets.json")


@dataclass
class ModelPreset:
    """Represents a saved model configuration."""
    label: str
    model_name: str
    path: str
    executable: Optional[str] = None  # Optional override for executable
    vram_gb: Optional[float] = None  # VRAM requirement in GB
    description: Optional[str] = None  # Human-readable description
    provider: str = "llama_cpp"  # Provider: "llama_cpp" or "ollama"


def get_model_presets(force_refresh: bool = False) -> List[ModelPreset]:
    """Load model presets from config file."""
    if not PRESETS_FILE.exists():
        return []

    try:
        with open(PRESETS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [ModelPreset(**item) for item in data]
    except Exception as e:
        print(f"Error loading presets: {e}")
        return []


def apply_model_preset(preset: ModelPreset, restart_server: bool = True) -> None:
    """Apply a model preset to the runtime configuration."""
    
    # 1. Update runtime state
    state = load_runtime_state()
    
    state["provider"] = preset.provider
    state["model_name"] = preset.model_name
    
    # Update llama_server section
    # Update llama_server section only if provider is llama_cpp
    if preset.provider == "llama_cpp":
        if "llama_server" not in state:
            state["llama_server"] = {}
            
        state["llama_server"]["model_path"] = preset.path
        if preset.executable:
            state["llama_server"]["executable"] = preset.executable
        
        # Ensure optimized defaults if missing
        llama_cfg = state["llama_server"]
        if "context" not in llama_cfg:
            llama_cfg["context"] = 32768
        if "batch" not in llama_cfg:
            llama_cfg["batch"] = 4096
        if "gpu_layers" not in llama_cfg:
            llama_cfg["gpu_layers"] = 999
        if "parallel" not in llama_cfg:
            llama_cfg["parallel"] = 4
        if "flash_attn" not in llama_cfg:
            llama_cfg["flash_attn"] = True
        if "extra_args" not in llama_cfg:
            # Default optimization for AMD/High-RAM
            llama_cfg["extra_args"] = "--ubatch-size 512 --threads 12 -cb -no-mmap"
    
    # For Ollama, we might want to update base_url if needed, but usually defaults are fine
    if preset.provider == "ollama":
        # Ensure base_url is set to default if not present
        if "base_url" not in state or not state["base_url"]:
            state["base_url"] = "http://localhost:11434/v1"

    save_runtime_state(state)
    
    # 2. Restart server if requested
    if restart_server and llama_manager.is_running():
        llama_manager.stop()
        # Auto-start will happen on next query or if main window timer picks it up
        # But we can also trigger it here if needed. 
        # For now, we rely on the UI/System to restart it or the user to click "Start"
