"""Diagnostic script to analyze context token breakdown.

Run with: python diagnose_context.py

This will analyze a sample query and show exactly where tokens are coming from.
"""

import sys
sys.path.insert(0, ".")

from src.config_loader import get_settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.prompts import build_user_prompt, SYSTEM_LIT_RAG, format_chunk_for_prompt


def count_tokens(text: str) -> int:
    """Estimate tokens (conservative: 3 chars per token)."""
    return len(text) // 3


def main():
    print("=" * 70)
    print("CONTEXT TOKEN BREAKDOWN DIAGNOSTIC")
    print("=" * 70)
    
    settings = get_settings()
    retriever = HybridRetriever(settings)
    
    # Use a sample query
    query = "smith metadata issue"
    print(f"\nQuery: '{query}'")
    print("-" * 70)
    
    # Retrieve chunks
    chunks = retriever.retrieve(query, context_to_llm=25)
    print(f"\nRetrieved {len(chunks)} chunks")
    
    # Analyze individual chunks
    print("\n" + "=" * 70)
    print("PER-CHUNK BREAKDOWN")
    print("=" * 70)
    
    total_text_chars = 0
    total_summary_chars = 0
    docs_with_summaries = set()
    
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        summary = chunk.get("document_summary", "")
        file_name = chunk.get("metadata", {}).get("file_name", "unknown")
        
        text_chars = len(text)
        summary_chars = len(summary) if summary else 0
        
        total_text_chars += text_chars
        if summary:
            docs_with_summaries.add(file_name)
            total_summary_chars += summary_chars
        
        print(f"Chunk {i+1}: {text_chars:,} chars | Summary: {summary_chars:,} chars | Doc: {file_name[:40]}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"\nChunk text (raw):      {total_text_chars:,} chars (~{count_tokens(str(total_text_chars)):,} tokens)")
    print(f"Document summaries:    {total_summary_chars:,} chars (~{count_tokens(str(total_summary_chars)):,} tokens)")
    print(f"Unique docs with summaries: {len(docs_with_summaries)}")
    
    # Build prompt to see formatted size
    print("\n" + "=" * 70)
    print("PROMPT CONSTRUCTION")
    print("=" * 70)
    
    # With summaries
    prompt_with_summaries = build_user_prompt(query, chunks, include_summaries=True)
    # Without summaries
    prompt_without_summaries = build_user_prompt(query, chunks, include_summaries=False)
    
    system_chars = len(SYSTEM_LIT_RAG)
    prompt_with_chars = len(prompt_with_summaries)
    prompt_without_chars = len(prompt_without_summaries)
    
    total_with = system_chars + prompt_with_chars
    total_without = system_chars + prompt_without_chars
    
    print(f"\nSystem prompt:         {system_chars:,} chars (~{count_tokens(str(system_chars)):,} tokens)")
    print(f"User prompt (w/ sum):  {prompt_with_chars:,} chars (~{count_tokens(str(prompt_with_chars)):,} tokens)")
    print(f"User prompt (no sum):  {prompt_without_chars:,} chars (~{count_tokens(str(prompt_without_chars)):,} tokens)")
    print(f"\nTotal WITH summaries:  {total_with:,} chars (~{count_tokens(str(total_with)):,} tokens)")
    print(f"Total WITHOUT sum:     {total_without:,} chars (~{count_tokens(str(total_without)):,} tokens)")
    
    overhead = prompt_with_chars - prompt_without_chars
    print(f"\nSummary OVERHEAD:      {overhead:,} chars (~{count_tokens(str(overhead)):,} tokens)")
    
    # Show average per chunk
    avg_chunk = total_text_chars / len(chunks) if chunks else 0
    avg_summary = total_summary_chars / len(docs_with_summaries) if docs_with_summaries else 0
    
    print(f"\nAvg chunk size:        {avg_chunk:,.0f} chars (~{count_tokens(str(int(avg_chunk))):,} tokens)")
    print(f"Avg doc summary size:  {avg_summary:,.0f} chars (~{count_tokens(str(int(avg_summary))):,} tokens)")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if overhead > 30000:
        print("\n⚠️  Document summaries add significant overhead!")
        print("    Consider: Shorter summaries or disabling include_summaries")
    
    if avg_chunk > 2000:
        print("\n⚠️  Chunks are large - consider reducing chunk_size in config")
    
    if len(docs_with_summaries) > 5:
        print(f"\n⚠️  {len(docs_with_summaries)} unique documents - each adds summary overhead")
        print("    Consider: reduce context_to_llm to pull from fewer docs")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
