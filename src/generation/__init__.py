"""Generation components: LLM service, citation enforcement, prompt templates."""

from src.generation.citation import (
    CitationParser,
    CitationVerifier,
    CitationVerificationResult,
    verify_citations,
)
from src.generation.llm_service import LLMService
from src.generation.prompts import (
    REFUSAL_TEMPLATE,
    SYSTEM_LIT_RAG,
    USER_TEMPLATE,
    build_user_prompt,
    format_chunk_for_prompt,
)

__all__ = [
    "LLMService",
    "CitationParser",
    "CitationVerifier",
    "CitationVerificationResult",
    "verify_citations",
    "SYSTEM_LIT_RAG",
    "USER_TEMPLATE",
    "REFUSAL_TEMPLATE",
    "build_user_prompt",
    "format_chunk_for_prompt",
]
