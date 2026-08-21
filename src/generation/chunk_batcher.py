"""Chunk batching orchestration for parallel LLM generation."""

from __future__ import annotations

import logging
import threading
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from src.generation.llm_service import LLMService, estimate_token_count
import re

logger = logging.getLogger(__name__)


SYNTHESIS_SYSTEM_PROMPT = """You are a litigation analyst. Synthesize the provided batched notes into a comprehensive, client-ready answer.
 
 OUTPUT DIRECTLY with these sections (do NOT use <|channel|> markers or reasoning):
 
 Summary:
 - A detailed executive summary of the key findings.
 
 Key Facts:
 1. Numbered factual statements with citations.
 - INCLUDE ALL RELEVANT DETAILS. Do not summarize if it loses important nuance.
 - Be exhaustive.
 
 Outstanding Issues / Risks:
 - Bullet points (or "None noted.")
 
 Sources:
 - List each unique citation exactly as it appears in the notes
 
 RULES:
 - Reuse citation strings verbatim from the notes - do NOT modify or invent citations
 - Preserve ALL critical factual language.
 - Do NOT truncate or summarize for brevity; prioritize completeness.
 - Maintain precise, authoritative tone
 - Output the final answer directly - no meta-commentary or step-by-step reasoning
 - STOP immediately after the Sources section.
 """


def extract_final_answer_from_synthesis(synthesis_output: str) -> tuple[str, str]:
    """Extract final answer from synthesis output that may contain reasoning.

    The model often generates reasoning/analysis before the final answer.
    This function tries to extract the actual answer sections.

    Args:
        synthesis_output: Raw synthesis output (may include reasoning)

    Returns:
        Tuple of (final_answer, reasoning) where:
        - final_answer: Extracted structured answer with Summary/Key Facts/etc.
        - reasoning: The reasoning/analysis part (for optional display)
    """
    if not synthesis_output:
        return "", ""

    # Strategy 1: Look for the last occurrence of section headers (Summary, Key Facts, etc.)
    # The final answer typically appears after reasoning
    section_pattern = r'(?:^|\n)(Summary:|Key Facts:|Outstanding Issues|Sources:)'
    matches = list(re.finditer(section_pattern, synthesis_output, re.MULTILINE | re.IGNORECASE))

    if matches:
        # Find the last cluster of section headers (indicates final answer)
        # Look for the last Summary: which typically starts the final answer
        summary_matches = [m for m in matches if 'summary' in m.group(1).lower()]

        if summary_matches:
            # Get the last Summary section - this is likely the start of the final answer
            last_summary_idx = summary_matches[-1].start()

            # Extract everything from this point as the final answer
            final_answer = synthesis_output[last_summary_idx:].strip()

            # Everything before is reasoning
            reasoning = synthesis_output[:last_summary_idx].strip()

            return final_answer, reasoning

    # Strategy 2: If no clear sections found, try to split on common reasoning patterns
    # Look for transitions like "Therefore:", "In conclusion:", "Final answer:", etc.
    transition_pattern = r'(?:^|\n)(?:Therefore|In conclusion|Final answer|To summarize)[:\s]'
    transitions = list(re.finditer(transition_pattern, synthesis_output, re.MULTILINE | re.IGNORECASE))

    if transitions:
        # Use the last transition as the split point
        last_transition = transitions[-1].start()
        final_answer = synthesis_output[last_transition:].strip()
        reasoning = synthesis_output[:last_transition].strip()
        return final_answer, reasoning

    # Strategy 3: If all else fails, look for the last substantial paragraph
    # Split by double newlines and take the last few paragraphs
    paragraphs = [p.strip() for p in synthesis_output.split('\n\n') if p.strip()]

    if len(paragraphs) > 2:
        # Assume the last 40% of content is the final answer
        split_idx = int(len(paragraphs) * 0.6)
        reasoning = '\n\n'.join(paragraphs[:split_idx])
        final_answer = '\n\n'.join(paragraphs[split_idx:])
        return final_answer, reasoning

    # Fallback: Return everything as final answer (no reasoning detected)
    return synthesis_output, ""


class ChunkBatchGenerator:
    """Split retrieved chunks into batches and generate answers in parallel."""

    def __init__(self, base_service: LLMService):
        self.base_service = base_service
        self.settings = base_service.settings
        self.last_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------#
    # Public API
    # ------------------------------------------------------------------#
    def should_batch(self, chunk_count: int) -> bool:
        """Return True if chunk batching should be applied."""
        cfg = getattr(self.settings, "generation", None)
        if not cfg or not cfg.enable_batching:
            return False
        threshold = max(cfg.min_chunks_for_batching, cfg.chunk_batch_size)
        result = chunk_count >= threshold
        return result

    def generate(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        model: Optional[str] = None,
        token_callback: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        max_output_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate response by splitting context into chunk batches."""
        self._current_system_prompt = system_prompt  # Store for batch calls
        cfg = getattr(self.settings, "generation", None)
        if not cfg:
            raise RuntimeError("Generation configuration missing; cannot batch.")

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Generation cancelled")

        batches = self._build_batches(chunks, cfg)
        if not batches:
            raise ValueError("No chunk batches available for generation.")

        total_batches = len(batches)
        model_name = model or self.base_service.get_default_model()
        progress = getattr(self.base_service, "progress_signals", None)

        if progress:
            progress.generation_started.emit(model_name)
            progress.stage_changed.emit("Parallel batches")
            progress.batch_progress.emit(0, total_batches)

        start_time = time.time()
        ordered_parts = ["" for _ in range(total_batches)]
        notes_for_synthesis = ["" for _ in range(total_batches)]
        ordered_stats: list[dict[str, Any]] = []
        pending_results: dict[int, tuple[str, dict[str, Any]]] = {}
        next_to_emit = 0
        completed_batches = 0

        logger.debug("[BATCH] Starting parallel generation with %d workers for %d batches",
                     cfg.parallel_workers, total_batches)

        with ThreadPoolExecutor(max_workers=max(1, int(cfg.parallel_workers))) as executor:
            future_map = {
                executor.submit(
                    self._run_single_batch,
                    query,
                    batch["chunks"],
                    model_name,
                    cancel_event,
                ): idx
                for idx, batch in enumerate(batches)
            }

            logger.debug("[BATCH] Submitted %d batch tasks to executor", len(future_map))

            for future in as_completed(future_map):
                if cancel_event and cancel_event.is_set():
                    raise RuntimeError("Generation cancelled")

                batch_index = future_map[future]
                logger.debug("[BATCH] Batch %d/%d completed, retrieving result...",
                             batch_index + 1, total_batches)

                try:
                    # Add timeout to prevent indefinite hanging
                    batch_text, batch_stats = future.result(timeout=600)  # 10 minute timeout
                    logger.debug("[BATCH] Batch %d result: %d chars, %d tokens",
                                 batch_index + 1, len(batch_text), batch_stats.get('token_count', 0))
                except Exception as e:
                    logger.error("[BATCH] ERROR in batch %d: %s", batch_index + 1, str(e))
                    raise RuntimeError(f"Batch {batch_index + 1} failed: {str(e)}") from e

                pending_results[batch_index] = (batch_text, batch_stats)
                completed_batches += 1

                if progress:
                    progress.batch_progress.emit(completed_batches, total_batches)

                while next_to_emit in pending_results:
                    text, stats = pending_results.pop(next_to_emit)
                    formatted = self._format_batch_output(
                        next_to_emit,
                        batches,
                        text,
                        cfg,
                    )
                    ordered_parts[next_to_emit] = formatted
                    notes_for_synthesis[next_to_emit] = formatted
                    ordered_stats.append(stats)
                    next_to_emit += 1

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Generation cancelled")

        final_text = cfg.join_separator.join(part for part in ordered_parts if part).strip()
        total_duration_ms = (time.time() - start_time) * 1000
        approx_tokens = estimate_token_count(final_text) if final_text else 0

        stats = self._aggregate_stats(
            ordered_stats,
            approx_tokens,
            total_duration_ms,
            sum(batch["count"] for batch in batches),
            total_batches,
        )

        self.last_stats = stats.copy()
        self.base_service.last_generation_stats = stats.copy()
        self.base_service.last_generation_stats["batch_count"] = total_batches

        if cfg.enable_synthesis:
            if progress:
                progress.stage_changed.emit("Synthesis")
            synthesized = self._synthesize_notes(
                query=query,
                notes=[note for note in notes_for_synthesis if note.strip()],
                model_name=model_name,
                cancel_event=cancel_event,
            )
            if synthesized:
                # For now, use FULL synthesis output as main display
                # The extraction can be improved based on user feedback
                final_text = synthesized.strip()
                stats["synthesis_full"] = synthesized

                # Also try extraction for comparison
                extracted_answer, reasoning = extract_final_answer_from_synthesis(synthesized)
                if extracted_answer and reasoning:
                    stats["synthesis_extracted"] = extracted_answer
                    stats["synthesis_reasoning"] = reasoning
                    logger.debug("[BATCH] Synthesis: %d chars total, extracted: %d chars, reasoning: %d chars",
                                 len(synthesized), len(extracted_answer), len(reasoning))
                else:
                    logger.debug("[BATCH] Extraction failed, using full synthesis: %d chars", len(synthesized))

        if progress:
            progress.stage_changed.emit("Assembly")
            progress.generation_completed.emit(
                stats.get("token_count", 0) or 0,
                stats.get("duration_ms", total_duration_ms),
                stats.get("tokens_per_sec", 0.0) or 0.0,
            )
            progress.stage_changed.emit("Completed")

        if token_callback and final_text:
            token_callback(final_text)

        return final_text, stats

    # ------------------------------------------------------------------#
    # Helpers
    # ------------------------------------------------------------------#
    def _build_batches(self, chunks: list[dict[str, Any]], cfg) -> list[dict[str, Any]]:
        """Return list of batch descriptors with chunk slices and metadata."""
        batch_size = max(1, int(cfg.chunk_batch_size))
        batches: list[dict[str, Any]] = []

        batch_idx = 0
        for start in range(0, len(chunks), batch_size):
            if cfg.max_batches and batch_idx >= int(cfg.max_batches):
                break
            chunk_slice = chunks[start : start + batch_size]
            batches.append(
                {
                    "chunks": chunk_slice,
                    "start": start + 1,
                    "end": start + len(chunk_slice),
                    "count": len(chunk_slice),
                }
            )
            batch_idx += 1

        return batches

    def describe_plan(self, chunk_count: int) -> dict[str, Any]:
        """Summarize how many batches would be created for a chunk_count."""
        cfg = getattr(self.settings, "generation", None)
        if not cfg or chunk_count <= 0:
            return {"enabled": False, "chunk_count": chunk_count}

        batch_size = max(1, int(cfg.chunk_batch_size))
        threshold = max(cfg.min_chunks_for_batching, batch_size)
        max_batches = int(cfg.max_batches) if cfg.max_batches else None

        plan: dict[str, Any] = {
            "chunk_count": chunk_count,
            "batch_size": batch_size,
            "threshold": threshold,
            "max_batches": max_batches,
        }

        if chunk_count < threshold:
            plan.update(
                {
                    "enabled": False,
                    "batch_count": 1,
                    "covered_chunks": chunk_count,
                    "last_batch_size": chunk_count,
                }
            )
            return plan

        batch_count = math.ceil(chunk_count / batch_size)
        if max_batches:
            max_batches = max(1, max_batches)
            batch_count = min(batch_count, max_batches)

        remainder = chunk_count - batch_size * (batch_count - 1)
        last_batch_size = remainder if remainder > 0 else batch_size
        covered_chunks = (batch_count - 1) * batch_size + min(last_batch_size, batch_size)

        plan.update(
            {
                "enabled": True,
                "batch_count": batch_count,
                "covered_chunks": min(chunk_count, covered_chunks),
                "last_batch_size": min(last_batch_size, batch_size),
            }
        )
        return plan

    def _run_single_batch(
        self,
        query: str,
        batch_chunks: list[dict[str, Any]],
        model_name: str,
        cancel_event: Optional[threading.Event],
    ) -> tuple[str, dict[str, Any]]:
        """Generate answer for a single batch using a dedicated LLMService with streaming.

        Uses streaming mode to prevent blocking HTTP requests and allow llama.cpp
        to serve requests sequentially instead of causing thread pool starvation.
        """
        thread_id = threading.current_thread().name
        logger.debug("[BATCH] [%s] Starting batch generation for %d chunks", thread_id, len(batch_chunks))

        if cancel_event and cancel_event.is_set():
            raise RuntimeError("Generation cancelled")

        try:
            batch_service = LLMService(settings=self.base_service.settings)
            logger.debug("[BATCH] [%s] LLMService created, calling generate_with_context...", thread_id)

            max_tokens = getattr(self.settings.generation, "batch_max_tokens", None)
            system_prompt = getattr(self, "_current_system_prompt", None)

            response = batch_service.generate_with_context(
                query=query,
                chunks=batch_chunks,
                system_prompt=system_prompt,
                model=model_name,
                stream=False,
                cancel_event=cancel_event,
                max_tokens=max_tokens,
            )

            clean_response = batch_service._post_process_output(response) if response else ""
            logger.debug("[BATCH] [%s] Generation completed, response length: %d chars",
                         thread_id, len(clean_response))

            stats = getattr(batch_service, "last_generation_stats", {}) or {}
            if not stats.get("token_count") and clean_response:
                stats = {**stats, "token_count": estimate_token_count(clean_response)}

            logger.debug("[BATCH] [%s] Batch finished successfully", thread_id)
            return clean_response.strip(), stats

        except Exception as e:
            logger.error("[BATCH] [%s] ERROR during batch generation: %s", thread_id, str(e))
            import traceback
            traceback.print_exc()
            raise

    def _format_batch_output(
        self,
        batch_index: int,
        batches: list[dict[str, Any]],
        text: str,
        cfg,
    ) -> str:
        """Apply optional headers to each batch output."""
        body = text.strip()
        if not body:
            return ""

        if not cfg.show_batch_headers:
            return body

        spec = batches[batch_index]
        header = cfg.header_template.format(
            index=batch_index + 1,
            total=len(batches),
            start=spec["start"],
            end=spec["end"],
        )
        return f"{header}\n{body}"

    def _aggregate_stats(
        self,
        batch_stats: list[dict[str, Any]],
        approx_tokens: int,
        total_duration_ms: float,
        chunk_count: int,
        batch_count: int,
    ) -> dict[str, Any]:
        """Merge batch-level statistics for downstream analytics."""
        prompt_tokens = sum((stat.get("prompt_tokens") or 0) for stat in batch_stats if stat)
        prompt_build_ms = sum((stat.get("prompt_build_ms") or 0) for stat in batch_stats if stat)

        tokens_per_sec = (
            approx_tokens / (total_duration_ms / 1000)
            if approx_tokens and total_duration_ms > 0
            else None
        )

        return {
            "token_count": approx_tokens,
            "duration_ms": total_duration_ms,
            "tokens_per_sec": tokens_per_sec,
            "prompt_tokens": prompt_tokens or None,
            "prompt_build_ms": prompt_build_ms or None,
            "chunk_count": chunk_count,
            "batch_count": batch_count,
        }

    def _synthesize_notes(
        self,
        query: str,
        notes: list[str],
        model_name: str,
        cancel_event: Optional[threading.Event],
        max_tokens: Optional[int] = None,
    ) -> str:
        """Combine batch notes into a single structured response.

        Args:
            max_tokens: Maximum tokens for synthesis. If None, uses unlimited (no truncation).
        """
        if not notes:
            return ""

        combined_notes = "\n\n".join(
            f"Note {idx + 1}:\n{note.strip()}"
            for idx, note in enumerate(notes)
            if note.strip()
        )

        if not combined_notes:
            return ""

        user_prompt = (
            f"Query:\n{query}\n\n"
            "Batched notes from the retrieval phase:\n"
            f"{combined_notes}\n\n"
            "Reformat these notes into the required structure. Reuse the existing citation "
            "strings exactly as they appear; do not invent new citations."
        )

        try:
            # Use configured max tokens or default to 8192 for comprehensive answers
            synthesis_limit = max_tokens if max_tokens is not None else getattr(self.settings.generation, "synthesis_max_tokens", 8192)
            
            logger.debug("[SYNTHESIS] Starting synthesis...")
            logger.debug("[SYNTHESIS] Combined notes: %d chars, max_tokens: %s", len(combined_notes), synthesis_limit)
            
            synthesized = self.base_service.generate(
                prompt=user_prompt,
                system_prompt=SYNTHESIS_SYSTEM_PROMPT,
                model=model_name,
                temperature=0.15,
                cancel_event=cancel_event,
                max_tokens=synthesis_limit,
            )
            
            logger.debug("[SYNTHESIS] Raw generation returned: %d chars", len(synthesized) if synthesized else 0)
            
            if not synthesized or len(synthesized.strip()) == 0:
                logger.warning("[SYNTHESIS] Synthesis returned empty! Query: %s...", query[:100])
                logger.debug("[SYNTHESIS] FALLBACK: Returning combined batch notes instead")
                return combined_notes
            
            logger.debug("[SYNTHESIS] Synthesis successful: %d chars", len(synthesized))
            return synthesized
        except Exception as exc:
            logger.error("[BATCH] Synthesis failed: %s", exc)
            logger.debug("[SYNTHESIS] FALLBACK: Returning combined notes")
            return combined_notes


