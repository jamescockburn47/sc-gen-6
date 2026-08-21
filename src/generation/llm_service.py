"""LLM service using Ollama or LM Studio with streaming and citation-aware prompts."""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import Any, Callable, Optional

from src.config.llm_config import LLMConfig, load_llm_config
from src.config_loader import Settings, get_settings
from src.generation.prompts import REFUSAL_TEMPLATE, SYSTEM_LIT_RAG, build_user_prompt
from src.llm.client import LLMClient, get_llm_client

logger = logging.getLogger(__name__)


def estimate_token_count(text: str) -> int:
    """Estimate token count using a conservative heuristic.

    Args:
        text: Text to estimate tokens for

    Returns:
        Estimated token count (approximately 3.5 chars/token for legal/mixed content).
        Using 3.5 rather than 4 to avoid underestimating and hitting context limits.
    """
    # Conservative: 3.5 chars/token for legal documents (citations, numbers, short words)
    return max(1, int(len(text) / 3.5))


def _deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Remove duplicate/near-duplicate chunks by text fingerprint.

    Hybrid retrieval (semantic + BM25) can return the same passage under
    different chunk IDs — especially for short single-page documents where
    every retrieval path surfaces the same content.  We deduplicate by
    comparing the first 200 normalised characters of each chunk; if two
    chunks share the same fingerprint, only the highest-scoring one survives.

    Preserves original ordering (highest score wins within a fingerprint group).
    """
    seen: dict[str, dict] = {}
    for chunk in chunks:
        raw = chunk.get("text") or ""
        # Normalise: collapse whitespace, lower-case, take first 200 chars
        fingerprint = " ".join(raw.split())[:200].lower()
        if not fingerprint:
            continue
        existing = seen.get(fingerprint)
        if existing is None:
            seen[fingerprint] = chunk
        else:
            # Keep whichever has the higher reranker/RRF score
            if chunk.get("score", 0.0) > existing.get("score", 0.0):
                seen[fingerprint] = chunk

    # Restore original relative order (by position in input list)
    order = {id(c): i for i, c in enumerate(chunks)}
    deduped = sorted(seen.values(), key=lambda c: order.get(id(c), 0))
    removed = len(chunks) - len(deduped)
    if removed:
        logger.info("[Dedup] Removed %d duplicate chunk(s) before prompt assembly", removed)
    return deduped


try:
    from PySide6.QtCore import QObject, Signal
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


if QT_AVAILABLE:
    class GenerationProgressSignals(QObject):
        """Progress signals for LLM generation."""

        # Generation started: (model_name)
        generation_started = Signal(str)

        # Generation progress: (tokens_generated, time_elapsed_ms, tokens_per_sec)
        generation_progress = Signal(int, float, float)

        # Generation completed: (total_tokens, total_time_ms, avg_tokens_per_sec)
        generation_completed = Signal(int, float, float)

        # Stage updates for UI labels: (stage_name)
        stage_changed = Signal(str)

        # Batch progress (completed, total) for chunk batching
        batch_progress = Signal(int, int)


class LLMService:
    """LLM service backed by OpenAI-compatible local providers (llama.cpp, LM Studio)."""

    def __init__(self, settings: Optional[Settings] = None, progress_signals: Optional['GenerationProgressSignals'] = None):
        """Initialize LLM service.

        Args:
            settings: Settings instance. If None, loads from config.
            progress_signals: Optional progress signals for UI updates.
        """
        self.settings = settings or get_settings()
        self.progress_signals = progress_signals
        self.llm_env_config: LLMConfig = load_llm_config()
        self.llm_client: LLMClient = get_llm_client(self.llm_env_config)
        self.default_response_format: dict[str, str] = {"type": "text"}
        self.default_stop_sequences: list[str] = ["<|channel|>"]
        self.last_generation_stats: dict[str, float | int] = {}
    
    def supports_thinking(self, model: Optional[str] = None) -> bool:
        """Check if a model supports thinking/reasoning mode.
        
        Args:
            model: Model name. If None, uses default from config.
            
        Returns:
            True if the model supports thinking mode
        """
        model = model or self.get_default_model()
        model_lower = model.lower()
        
        # Check against configured thinking models
        for thinking_model in self.settings.models.llm.thinking_models:
            if thinking_model.lower() in model_lower:
                return True
        return False
    
    def _build_thinking_kwargs(self, model: str) -> dict[str, Any]:
        """Build additional kwargs to enable thinking mode for compatible models.
        
        Args:
            model: Model name
            
        Returns:
            Dict of additional kwargs for the LLM API call
        """
        kwargs = {}
        
        if not self.settings.models.llm.enable_thinking:
            return kwargs
        
        if not self.supports_thinking(model):
            return kwargs
        
        model_lower = model.lower()
        
        # Qwen3 models: Use enable_thinking parameter
        if "qwen3" in model_lower or "qwen-3" in model_lower:
            kwargs["extra_body"] = {
                "enable_thinking": True,
                "thinking_budget": self.settings.models.llm.thinking_budget,
            }
        
        # DeepSeek-R1: Thinking is built-in, but we can increase max_tokens
        elif "deepseek-r1" in model_lower:
            # DeepSeek-R1 uses <think> tags automatically
            kwargs["extra_body"] = {
                "enable_thinking": True,
            }
        
        # Nemotron 3 Nano: Supports reasoning via thinking budget
        elif "nemotron" in model_lower:
            kwargs["extra_body"] = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": self.settings.models.llm.thinking_budget,
                }
            }
        
        # GLM 4.7 Flash: Supports enable_thinking via Ollama extra_body
        elif "glm" in model_lower:
            kwargs["extra_body"] = {
                "enable_thinking": True,
            }
        
        # For other models, no special handling needed
        return kwargs

    # ------------------------------------------------------------------#
    # Helper methods
    # ------------------------------------------------------------------#
    def get_default_model(self) -> str:
        """Get default model from config.

        Returns:
            Default model name
        """
        return self.llm_env_config.model_name or self.settings.models.llm.default

    def get_context_window_size(self) -> int:
        """Get context window size from runtime config.

        Returns:
            Context window size in tokens (default 32768 for safety)
        """
        from src.config.runtime_store import load_runtime_state
        runtime_state = load_runtime_state()
        llama_server_config = runtime_state.get("llama_server", {})
        # Default to 32768 for safety (conservative estimate)
        return llama_server_config.get("context", 32768)

    def get_available_models(self) -> list[str]:
        """Try to retrieve the list of models exposed by the active provider."""
        try:
            models = self.llm_client.list_models()
            if models:
                return models
        except Exception:
            pass
        return self.settings.models.llm.available

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        stream: bool = False,
        cancel_event: Optional[threading.Event] = None,
        initial_stats: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text from prompt (non-streaming).

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt. If None, uses default.
            model: Model name. If None, uses default from config.
            temperature: Sampling temperature (0.0 = deterministic)
            stream: Whether to stream (ignored in non-streaming mode)

        Returns:
            Generated text

        Raises:
            RuntimeError: If generation fails
        """
        request_start = time.time()
        model = model or self.get_default_model()
        system_prompt = system_prompt or SYSTEM_LIT_RAG

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("LLM generation cancelled")

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            request_start = time.time()
            kwargs = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
            }
            if self._should_apply_stop_sequences():
                kwargs["stop"] = self.default_stop_sequences
            if self._should_apply_response_format():
                kwargs["response_format"] = self.default_response_format
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            
            # Add thinking mode kwargs if model supports it
            thinking_kwargs = self._build_thinking_kwargs(model)
            kwargs.update(thinking_kwargs)
            
            response = self.llm_client.generate_chat_completion(**kwargs)
            response = self._post_process_output(response)
            duration_ms = (time.time() - request_start) * 1000
            base_stats = initial_stats.copy() if initial_stats else {}
            base_stats.update({
                "token_count": None,
                "duration_ms": duration_ms,
                "tokens_per_sec": None,
            })
            self.last_generation_stats = base_stats
            return response
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {str(e)}") from e

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        callback: Optional[Callable[[str], None]] = None,
        thinking_callback: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        initial_stats: Optional[dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text with streaming tokens.

        Args:
            callback: Called with each final-answer token.
            thinking_callback: Called with each thinking/reasoning token (can be None).
            max_tokens: Hard cap on total tokens (thinking + answer). None = unlimited.
        """
        model = model or self.get_default_model()
        system_prompt = system_prompt or SYSTEM_LIT_RAG

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("LLM generation cancelled")

        if self.progress_signals:
            self.progress_signals.generation_started.emit(model)
            self.progress_signals.stage_changed.emit("Streaming answer")

        # Separate buffers for answer content vs thinking
        token_buffer = queue.Queue(maxsize=200)
        complete_text = []  # final answer only
        error_occurred = threading.Event()

        start_time = time.time()
        token_count = [0]
        last_progress_time = [start_time]
        self.last_generation_stats = initial_stats.copy() if initial_stats else {}

        def token_consumer():
            while True:
                try:
                    item = token_buffer.get(timeout=1.0)
                    if item is None:  # sentinel
                        break
                    kind, token = item
                    if kind == "content":
                        complete_text.append(token)
                        token_count[0] += 1
                        current_time = time.time()
                        if current_time - last_progress_time[0] > 0.1 or token_count[0] % 10 == 0:
                            elapsed_ms = (current_time - start_time) * 1000
                            tok_per_sec = token_count[0] / max(current_time - start_time, 0.001)
                            if self.progress_signals:
                                self.progress_signals.generation_progress.emit(
                                    token_count[0], elapsed_ms, tok_per_sec
                                )
                            last_progress_time[0] = current_time
                        if callback:
                            callback(token)
                    elif kind == "thinking":
                        if thinking_callback:
                            thinking_callback(token)
                    token_buffer.task_done()
                except queue.Empty:
                    if error_occurred.is_set() or (cancel_event and cancel_event.is_set()):
                        break
                    continue

        consumer_thread = threading.Thread(target=token_consumer, daemon=True)
        consumer_thread.start()

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            kwargs = {
                "messages": messages,
                "model": model,
                "temperature": temperature,
            }
            if self._should_apply_stop_sequences():
                kwargs["stop"] = self.default_stop_sequences
            if self._should_apply_response_format():
                kwargs["response_format"] = self.default_response_format
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            thinking_kwargs = self._build_thinking_kwargs(model)
            kwargs.update(thinking_kwargs)

            stream = self.llm_client.stream_chat_completion(**kwargs)

            for item in stream:
                if cancel_event and cancel_event.is_set():
                    error_occurred.set()
                    token_buffer.put(None)
                    raise RuntimeError("LLM generation cancelled")

                # item is (kind, token) from _iter_stream
                # Fallback: plain string from non-llama_cpp providers
                if isinstance(item, tuple):
                    kind, token = item
                else:
                    kind, token = "content", item

                try:
                    token_buffer.put((kind, token), timeout=0.1)
                except queue.Full:
                    token_buffer.put((kind, token), timeout=1.0)

            if cancel_event and cancel_event.is_set():
                raise RuntimeError("LLM generation cancelled")

            token_buffer.put(None)
            consumer_thread.join(timeout=10.0)

            total_time_ms = (time.time() - start_time) * 1000
            avg_tok_per_sec = token_count[0] / max(time.time() - start_time, 0.001) if token_count[0] > 0 else 0
            if self.progress_signals:
                self.progress_signals.generation_completed.emit(
                    token_count[0], total_time_ms, avg_tok_per_sec
                )
                self.progress_signals.stage_changed.emit("Completed")

            self.last_generation_stats.update({
                "token_count": token_count[0],
                "duration_ms": total_time_ms,
                "tokens_per_sec": avg_tok_per_sec,
            })

            return self._post_process_output("".join(complete_text))

        except Exception as e:
            error_occurred.set()
            token_buffer.put(None)
            raise RuntimeError(f"LLM streaming failed: {str(e)}") from e

    def generate_with_context(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        stream: bool = False,
        callback: Optional[Callable[[str], None]] = None,
        thinking_callback: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate answer with context chunks and automatic prompt composition.

        Automatically composes the prompt with strict citation rules and
        formatted chunks.

        Args:
            query: User query
            chunks: List of chunk dictionaries with 'text' and 'metadata' keys
            system_prompt: Optional system prompt. If None, uses default.
            model: Model name. If None, uses default from config.
            temperature: Sampling temperature (0.0 = deterministic)
            stream: Whether to stream tokens
            callback: Optional callback for streaming (token) -> None

        Returns:
            Generated answer text

        Raises:
            RuntimeError: If generation fails
            ValueError: If chunks list is empty
        """
        if not chunks:
            raise ValueError("Cannot generate answer without context chunks")

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Generation cancelled")

        # Deduplicate chunks by text content before prompt assembly.
        # Hybrid retrieval (semantic + BM25 + reranker) can return the same
        # passage under different chunk IDs, causing the LLM to see the same
        # source text 2-3× and waste its thinking budget on repetition analysis.
        assembly_start = time.time()
        chunks = _deduplicate_chunks(chunks)
        user_prompt = build_user_prompt(query, chunks)
        prompt_build_ms = (time.time() - assembly_start) * 1000

        # Determine system prompt based on mode if provided
        # Note: 'search_mode' is not currently passed to generate_with_context but could be in prompt construction
        # For now, we rely on SYSTEM_LIT_RAG as default
        # If specific modes are needed, they should be passed in or handled by build_user_prompt
        
        # Use provided system_prompt or default
        system_text = system_prompt or SYSTEM_LIT_RAG
        
        # HACK: Detect mode from system_prompt content if passed, or just rely on default
        # The QueryPanel passes specific prompts now, so we should respect them.
        
        total_prompt = system_text + "\n\n" + user_prompt
        estimated_tokens = estimate_token_count(total_prompt)
        context_window = self.get_context_window_size()
        prompt_stats = {
            "prompt_tokens": estimated_tokens,
            "prompt_chars": len(total_prompt),
            "chunk_count": len(chunks),
            "prompt_build_ms": prompt_build_ms,
        }

        # Reserve 25% of context for the response
        max_prompt_tokens = int(context_window * 0.75)

        if estimated_tokens > max_prompt_tokens:
            raise ValueError(
                f"Prompt too large: ~{estimated_tokens:,} tokens estimated, "
                f"but only {max_prompt_tokens:,} tokens available "
                f"(context window: {context_window:,}, 25% reserved for response). "
                f"Try reducing context_to_llm (currently {len(chunks)} chunks) "
                f"or use a model with larger context window."
            )

        # Generate with streaming or non-streaming
        if stream:
            return self.generate_stream(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                callback=callback,
                thinking_callback=thinking_callback,
                cancel_event=cancel_event,
                initial_stats=prompt_stats,
                max_tokens=max_tokens,
            )
        else:
            result = self.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                cancel_event=cancel_event,
                initial_stats=prompt_stats,
                max_tokens=max_tokens,
            )
            self.last_generation_stats = getattr(self, "last_generation_stats", {})
            self.last_generation_stats.update(prompt_stats)
            return result

    # ------------------------------------------------------------------#
    # Internal helpers
    # ------------------------------------------------------------------#
    def _should_apply_response_format(self) -> bool:
        return self.llm_env_config.provider != "llama_cpp"

    def _should_apply_stop_sequences(self) -> bool:
        return self.llm_env_config.provider != "llama_cpp"

    def _post_process_output(self, text: str) -> str:
        """Post-process LLM output based on provider.
        
        For llama.cpp: Extract content from channel markers
        For other providers: Return as-is
        """
        input_len = len(text) if text else 0
        logger.debug("[POST_PROCESS] Input: %d chars, Provider: %s", input_len, self.llm_env_config.provider)
        
        if self.llm_env_config.provider == "llama_cpp":
            result = self._extract_llama_final_channel(text)
            logger.debug("[POST_PROCESS] After channel extraction: %d chars", len(result) if result else 0)
            return result
        
        # For other providers (ollama, etc), return raw output
        logger.debug("[POST_PROCESS] Non-llama provider, returning raw text: %d chars", input_len)
        return text

    @staticmethod
    def _extract_llama_final_channel(text: str) -> str:
        """Extract final channel content from llama.cpp structured output.

        If no final channel is found, extracts all channel content and combines it,
        falling back to raw text if no channels are present.
        """
        logger.debug("[CHANNEL] Input text length: %d chars", len(text))

        # Try to extract from final channel first
        final_marker = "<|channel|>final<|message|>"
        idx = text.rfind(final_marker)
        if idx != -1:
            remainder = text[idx + len(final_marker):]
            end_idx = remainder.find("<|")
            if end_idx != -1:
                remainder = remainder[:end_idx]
            extracted = remainder.strip()
            logger.debug("[CHANNEL] Extracted from final channel: %d chars", len(extracted))
            return extracted

        # No final channel found - try to extract from any channel
        # Pattern: <|channel|>NAME<|message|>CONTENT
        import re
        channel_pattern = r'<\|channel\|>(\w+)<\|message\|>(.*?)(?=<\|channel\|>|$)'
        matches = re.findall(channel_pattern, text, re.DOTALL)

        if matches:
            channel_names = [m[0] for m in matches]
            logger.debug("[CHANNEL] Found %d channels: %s", len(matches), channel_names)

            # Combine all channel contents (analysis, final, etc.)
            combined = []
            for channel_name, content in matches:
                clean_content = content.strip()
                if clean_content:
                    logger.debug("[CHANNEL] Channel '%s': %d chars", channel_name, len(clean_content))
                    combined.append(clean_content)

            if combined:
                result = "\n\n".join(combined)
                logger.debug("[CHANNEL] Combined all channels: %d chars total", len(result))
                return result

        # No channels found at all - return raw text
        logger.debug("[CHANNEL] No channel markers found, returning raw text: %d chars", len(text))
        return text.strip()

    def _filter_llama_stream_token(self, token: str, state: dict[str, Any]) -> Optional[str]:
        state["buffer"] += token
        marker = "<|channel|>final<|message|>"

        if not state["use_filter"]:
            if "<|channel|>" in state["buffer"]:
                state["use_filter"] = True
            else:
                output = state["buffer"]
                state["buffer"] = ""
                return output or None

        if not state["final_started"]:
            idx = state["buffer"].find(marker)
            if idx == -1:
                return None
            state["final_started"] = True
            output = state["buffer"][idx + len(marker):]
            state["buffer"] = ""
            return output or None
        else:
            output = state["buffer"]
            state["buffer"] = ""
            return output or None

    def check_model_available(self, model: str) -> bool:
        """Check if a model is available in the backend.

        Args:
            model: Model name to check

        Returns:
            True if model is available
        """
        try:
            return model in self.llm_client.list_models()
        except Exception:
            return False

    def get_refusal_message(self) -> str:
        """Get standard refusal message when no chunks meet threshold.

        Returns:
            Refusal message text
        """
        return REFUSAL_TEMPLATE

