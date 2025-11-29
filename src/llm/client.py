"""Provider abstraction for OpenAI-compatible local LLM servers."""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

import requests

from src.config.llm_config import LLMConfig


class LLMClient:
    """Base client for OpenAI-compatible endpoints."""

    def __init__(self, config: LLMConfig, timeout: float = 1800.0):
        self.config = config
        self.timeout = timeout
        self.session = requests.Session()

    # ------------------------------------------------------------------#
    # Public API
    # ------------------------------------------------------------------#
    def generate_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """Send a non-streaming chat completion request."""
        # Use native Ollama API if configured
        if self.config.provider == "ollama":
            return self._generate_ollama_native(messages, model, temperature, stream=False, **kwargs)

        payload = self._build_payload(messages, model, temperature, stream=False, **kwargs)
        response = self.session.post(
            self._build_url("chat/completions"),
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(response)
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned from LLM response")
        return choices[0].get("message", {}).get("content", "")
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """Convenience method for simple text generation.
        
        Args:
            prompt: The user prompt
            model: Optional model override
            temperature: Temperature for generation
            **kwargs: Additional arguments
            
        Returns:
            Generated text response
        """
        messages = [{"role": "user", "content": prompt}]
        return self.generate_chat_completion(messages, model, temperature, **kwargs)

    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        **kwargs,
    ) -> Iterable[str]:
        """Stream tokens from the chat completion endpoint."""
        # Use native Ollama API if configured
        if self.config.provider == "ollama":
            yield from self._stream_ollama_native(messages, model, temperature, **kwargs)
            return

        payload = self._build_payload(messages, model, temperature, stream=True, **kwargs)
        with self.session.post(
            self._build_url("chat/completions"),
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
            stream=True,
        ) as response:
            self._raise_for_status(response)
            yield from self._iter_stream(response)

    def _generate_ollama_native(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
        stream: bool,
        **kwargs,
    ) -> str:
        """Generate using native Ollama API (/api/chat)."""
        # Native Ollama URL is base_url (e.g. localhost:11434) + /api/chat
        # Config base_url might have /v1, strip it
        base = self.config.base_url.replace("/v1", "").rstrip("/")
        url = f"{base}/api/chat"
        
        payload = {
            "model": model or self.config.model_name,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_ctx": 16384,  # Explicit 16K context
                "num_batch": 2048,  # Improve prompt processing speed
            }
        }
        
        # Add other options if needed
        if "max_tokens" in kwargs:
            payload["options"]["num_predict"] = kwargs["max_tokens"]
        # #region agent log

            
        response = self.session.post(url, json=payload, timeout=self.timeout)

        self._raise_for_status(response)
        
        data = response.json()

        content = data.get("message", {}).get("content", "")

        return content

    def _stream_ollama_native(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
        **kwargs,
    ) -> Iterable[str]:
        """Stream using native Ollama API (/api/chat)."""
        base = self.config.base_url.replace("/v1", "").rstrip("/")
        url = f"{base}/api/chat"
        
        payload = {
            "model": model or self.config.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": 16384,  # Explicit 16K context
                "num_batch": 2048,  # Improve prompt processing speed
            }
        }
        
        if "max_tokens" in kwargs:
            payload["options"]["num_predict"] = kwargs["max_tokens"]
            
        with self.session.post(url, json=payload, timeout=self.timeout, stream=True) as response:
            self._raise_for_status(response)
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        continue

    def list_models(self) -> list[str]:
        """Return the list of available models reported by the provider."""
        response = self.session.get(
            self._build_url("models"),
            headers=self._headers(),
            timeout=15.0,
        )
        self._raise_for_status(response)
        data = response.json()
        models = data.get("data", [])
        return [model.get("id", "") for model in models if model.get("id")]

    # ------------------------------------------------------------------#
    # Helpers
    # ------------------------------------------------------------------#
    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str],
        temperature: float,
        stream: bool,
        **kwargs,
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "model": model or self.config.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        
        # For Ollama: Set context window size to handle large prompts
        # Default Ollama context is only 4096, we need more for RAG
        # Note: num_ctx must be passed at the top level for OpenAI-compatible endpoint
        if self.config.provider == "ollama":
            # Try both locations - Ollama API is inconsistent
            payload["options"] = {"num_ctx": 16384}
            payload["num_ctx"] = 16384  # Also at top level
        
        payload.update(kwargs)
        return payload

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _build_url(self, path: str) -> str:
        base = self.config.base_url.rstrip("/")
        relative = path.lstrip("/")
        return f"{base}/{relative}"

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - HTTP errors wrapped
            # Check if this is a context size error
            if response.status_code == 400:
                try:
                    error_body = response.json()
                    error_msg = error_body.get("error", "")
                    if "context" in error_msg.lower() and ("exceed" in error_msg.lower() or "size" in error_msg.lower()):
                        raise ValueError(
                            f"Request exceeds server context window. "
                            f"Server error: {error_msg}. "
                            f"Try reducing the number of context chunks."
                        ) from exc
                except json.JSONDecodeError:
                    pass  # Fall through to generic error
                except ValueError:
                    raise  # Re-raise our context error

            raise RuntimeError(f"LLM HTTP request failed: {exc}") from exc

    @staticmethod
    def _iter_stream(response: requests.Response) -> Iterable[str]:
        for line in response.iter_lines(decode_unicode=False):
            if not line:
                continue
            if line.startswith(b"data:"):
                _, payload = line.split(b":", 1)
            else:
                payload = line

            payload = payload.strip()
            if not payload or payload == b"[DONE]":
                continue

            try:
                chunk = json.loads(payload.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                yield content


class LmStudioClient(LLMClient):
    """LM Studio OpenAI-compatible client."""

    pass


class LlamaCppClient(LLMClient):
    """llama.cpp llama-server OpenAI-compatible client."""

    pass


class OllamaClient(LLMClient):
    """Ollama OpenAI-compatible client.

    Ollama exposes an OpenAI-compatible API at /v1/* endpoints.
    """

    pass


def get_llm_client(config: LLMConfig) -> LLMClient:
    """Factory that returns the correct client for the configured provider."""
    provider = config.normalized_provider()

    if provider == "lmstudio":
        return LmStudioClient(config)
    elif provider == "ollama":
        return OllamaClient(config)
    return LlamaCppClient(config)


