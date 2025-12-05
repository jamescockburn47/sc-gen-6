"""Environment-driven configuration for OpenAI-compatible local LLM providers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.config.runtime_store import load_runtime_state, save_runtime_state

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs):
        return False


load_dotenv()


DEFAULT_PROVIDER = "llama_cpp"
DEFAULT_MODEL_NAME = "qwen2.5:32b"
DEFAULTS_BY_PROVIDER = {
    "llama_cpp": {
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key": "local-llama",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
}


@dataclass(slots=True)
class LLMConfig:
    """Runtime configuration for the active LLM provider."""

    provider: str
    base_url: str
    api_key: str
    model_name: str

    def normalized_provider(self) -> str:
        """Return normalized provider key."""
        return self.provider.lower()


def load_llm_config() -> LLMConfig:
    """Load LLM configuration from runtime overrides/env variables."""

    runtime_state = load_runtime_state()

    provider = (
        runtime_state.get("provider")
        or os.getenv("LLM_PROVIDER", DEFAULT_PROVIDER)
    ).strip().lower() or DEFAULT_PROVIDER
    if provider not in DEFAULTS_BY_PROVIDER:
        provider = DEFAULT_PROVIDER

    env_base_url = (
        runtime_state.get("base_url")
        or os.getenv("LLM_BASE_URL", "")
    ).strip()
    env_api_key = (
        runtime_state.get("api_key")
        or os.getenv("LLM_API_KEY", "")
    ).strip()
    default_provider_values = DEFAULTS_BY_PROVIDER[provider]

    # Prevent "sticky" base_url from previous provider
    # If the stored URL matches the default of a DIFFERENT provider, ignore it
    for other_provider, defaults in DEFAULTS_BY_PROVIDER.items():
        if other_provider != provider and env_base_url == defaults["base_url"]:
            env_base_url = ""
            break

    base_url = env_base_url or default_provider_values["base_url"]
    api_key = env_api_key or default_provider_values["api_key"]
    model_name = (
        runtime_state.get("model_name")
        or os.getenv("LLM_MODEL_NAME", DEFAULT_MODEL_NAME)
    ).strip() or DEFAULT_MODEL_NAME

    # Ensure base_url does not end with a trailing slash
    base_url = base_url.rstrip("/")

    return LLMConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
    )


def save_llm_config(config: LLMConfig) -> None:
    """Save LLM configuration to runtime state.
    
    Args:
        config: LLMConfig instance to save
    """
    state = load_runtime_state()
    state["provider"] = config.provider
    state["base_url"] = config.base_url
    state["api_key"] = config.api_key
    state["model_name"] = config.model_name
    save_runtime_state(state)


