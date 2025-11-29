"""Smoke test for the configured local LLM provider."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.llm_config import load_llm_config


def _headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _messages() -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a readiness probe."},
        {"role": "user", "content": "Say 'ready'."},
    ]


def main() -> int:
    cfg = load_llm_config()
    print("=" * 60)
    print("LLM Provider Smoke Test")
    print("=" * 60)
    print(f"Provider : {cfg.provider}")
    print(f"Base URL : {cfg.base_url}")
    print(f"Model    : {cfg.model_name}")

    payload: Dict[str, Any] = {
        "model": cfg.model_name,
        "messages": _messages(),
        "temperature": 0.0,
        "stream": False,
        "response_format": {"type": "text"},
        "stop": ["<|channel|>"],
    }

    endpoint = f"{cfg.base_url.rstrip('/')}/chat/completions"

    try:
        response = requests.post(
            endpoint,
            headers=_headers(cfg.api_key),
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        print(f"Ping response: {content}")
        return 0
    except Exception as exc:
        print(f"Connection failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

