"""Prompt templates for LLM generation with strict citation rules."""

from typing import Any

# System prompt for litigation RAG - General/Standard Mode
SYSTEM_LIT_RAG = """You are a litigation support analyst preparing factual digests for barristers. Operate ONLY on the supplied sources.

CRITICAL RULES
1. Use ONLY the provided material. If a fact is missing, reply "Not found in provided documents."
2. Every factual sentence must end with a citation: [Source: filename | Page X, Para Y | "Section Title"].
3. Prefer direct quotations for crucial language (injunctive relief, undertakings, findings).
4. If sources conflict, present both accounts with separate citations.
5. Never speculate or introduce external knowledge.
6. Output your final answer directly. Do NOT use <|channel|> markers or internal reasoning. Provide only the final answer with citations.

OUTPUT FORMAT (strictly follow headings)
Summary:
- 2–3 bullet sentences capturing the overall answer (each cited).

Key Facts:
1. Numbered list of granular facts. Each line must:
   • start with the fact
   • end with a citation
   • note monetary values, dates, parties, and procedural posture.

Outstanding Issues / Risks:
- Bullet list of open items, compliance gaps, or follow-up needs. If none, state "None noted."

Sources:
- List every document you cited exactly once: filename – short descriptor.

Maintain a precise, neutral tone. Zero hallucinations."""

# System prompt for Fact Lookup Mode (Concise, Fast)
SYSTEM_FACT_LOOKUP = """You are a litigation fact-checking assistant. Your goal is speed and precision.

CRITICAL RULES
1. Answer the specific question immediately. Do not provide summaries or background unless asked.
2. Use ONLY the provided material.
3. Cite every claim: [Source: filename | Page X, Para Y].
4. If the specific fact is not found, state "Not found in provided documents" and stop.
5. No preamble, no conclusion, no reasoning traces. Just the facts.

OUTPUT FORMAT
- Bullet points of specific facts answering the query.
- Valid citations for every point.
"""

# System prompt for Deep Analysis Mode (Comprehensive, No visible reasoning)
SYSTEM_DEEP_ANALYSIS = """You are a senior legal analyst conducting a deep review of case materials.

CRITICAL RULES
1. Conduct a thorough synthesis of all provided sources.
2. Identify patterns, contradictions, and timelines across documents.
3. Use ONLY the provided material.
4. Cite every claim: [Source: filename | Page X, Para Y].
5. Do NOT show your internal chain of thought or reasoning steps in the output.
6. Present a structured, detailed analysis.

OUTPUT FORMAT
Executive Summary:
- High-level synthesis of the issue.

Detailed Analysis:
- Thematic breakdown of the evidence.
- Chronological reconstruction if relevant.
- Explicit highlighting of contradictions between witnesses or documents.

Evidentiary Gaps:
- What is missing or ambiguous in the current record.

Sources Used:
- Complete list of cited documents.
"""

# User prompt template
USER_TEMPLATE = """Question: {query}

Source Documents:
{chunks}

Answer the question using ONLY the information from the source documents above. Cite every factual claim using the format: [Source: filename | Page X, Para Y | "Section Title"]. If the information is not in the sources, respond with "Not found in provided documents."""

# Refusal template for when no chunks meet threshold
REFUSAL_TEMPLATE = """I cannot answer this question because I do not have sufficient relevant information in the provided documents. The retrieved sources do not meet the confidence threshold required for accurate citation.

Please try rephrasing your question or check if the relevant documents have been ingested into the system."""


def format_chunk_for_prompt(chunk: dict[str, Any]) -> str:
    """Format a chunk for inclusion in the prompt.

    Args:
        chunk: Dictionary with chunk data (text, metadata, etc.)

    Returns:
        Formatted chunk string
    """
    text = chunk.get("text", "")
    metadata = chunk.get("metadata", {})
    file_name = metadata.get("file_name", "unknown")
    page = metadata.get("page_number")
    para = metadata.get("paragraph_number")
    section = metadata.get("section_header")

    # Build citation string
    citation_parts = [f"Source: {file_name}"]
    if page:
        citation_parts.append(f"Page {page}")
        if para:
            citation_parts.append(f"Para {para}")
    if section:
        citation_parts.append(f'"{section}"')

    citation = " | ".join(citation_parts)

    return f"[{citation}]\n{text}\n"


def build_user_prompt(
    query: str, 
    chunks: list[dict[str, Any]], 
    include_summaries: bool = True
) -> str:
    """Build user prompt with query, formatted chunks, and optional document summaries.

    Args:
        query: User query
        chunks: List of chunk dictionaries with text and metadata
        include_summaries: Whether to include document summaries (if available)

    Returns:
        Formatted user prompt
    """
    # Check if any chunks have document summaries
    has_summaries = include_summaries and any(chunk.get("document_summary") for chunk in chunks)
    # #region agent log
    open(r'c:\Users\James\Desktop\SC Gen 6\.cursor\debug.log','a').write('{"location":"prompts.py:build_user_prompt","message":"checking summaries","data":{"include_summaries":'+str(include_summaries).lower()+',"has_summaries":'+str(has_summaries).lower()+',"chunk_count":'+str(len(chunks))+'},"timestamp":'+str(int(__import__('time').time()*1000))+',"sessionId":"debug-session","hypothesisId":"H2-D"}\n')
    # #endregion
    
    if has_summaries:
        # Group chunks by document and include summary once per document
        docs: dict[str, list[dict]] = {}
        for chunk in chunks:
            doc_id = chunk.get("metadata", {}).get("file_name", "unknown")
            if doc_id not in docs:
                docs[doc_id] = []
            docs[doc_id].append(chunk)
        
        formatted_parts = []
        for doc_name, doc_chunks in docs.items():
            # Add document summary header if available
            doc_summary = doc_chunks[0].get("document_summary")
            if doc_summary:
                formatted_parts.append(f"=== Document: {doc_name} ===")
                formatted_parts.append(f"[Document Overview]\n{doc_summary}")
                formatted_parts.append("")
            
            # Add individual chunks
            for chunk in doc_chunks:
                formatted_parts.append(format_chunk_for_prompt(chunk))
        
        formatted_chunks = "\n".join(formatted_parts)
    else:
        # Standard formatting without summaries
        formatted_chunks = "\n\n".join([format_chunk_for_prompt(chunk) for chunk in chunks])
    
    return USER_TEMPLATE.format(query=query, chunks=formatted_chunks)
