"""Tests for LLM runtime store."""

from pathlib import Path

import json
import tempfile

from src.config import runtime_store


def test_load_save_runtime_state(tmp_path: Path, monkeypatch):
    """Ensure runtime state round-trips."""
    runtime_file = tmp_path / "llm_runtime.json"
    monkeypatch.setattr(runtime_store, "RUNTIME_FILE", runtime_file)

    data = runtime_store.load_runtime_state()
    assert data["provider"] == "ollama"

    data["provider"] = "lmstudio"
    data["base_url"] = "http://localhost:1234/v1"
    runtime_store.save_runtime_state(data)

    loaded = runtime_store.load_runtime_state()
    assert loaded["provider"] == "lmstudio"
    assert loaded["base_url"] == "http://localhost:1234/v1"

