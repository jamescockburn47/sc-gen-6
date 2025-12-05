"""Test parallel generation pipeline."""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import get_settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_service import LLMService


def test_parallel_generation():
    print("=" * 60)
    print("PARALLEL GENERATION TEST")
    print("=" * 60)
    
    settings = get_settings()
    print(f"\nConfig:")
    print(f"  parallel_workers: {settings.generation.parallel_workers}")
    print(f"  chunk_batch_size: {settings.generation.chunk_batch_size}")
    print(f"  enable_batching: {settings.generation.enable_batching}")
    
    # Initialize services
    print("\nInitializing services...")
    retriever = HybridRetriever(settings=settings)
    llm_service = LLMService(settings=settings)
    
    # Warm up
    _ = retriever.retrieve('test', semantic_top_n=5, keyword_top_n=5, rerank_top_k=3)
    
    query = "What are the key facts of this case and who are the main parties involved?"
    
    # Retrieve chunks
    print(f"\nQuery: '{query}'")
    print("Retrieving chunks...")
    
    start = time.time()
    chunks = retriever.retrieve(query, context_to_llm=12)
    retrieval_time = (time.time() - start) * 1000
    print(f"  Retrieved {len(chunks)} chunks in {retrieval_time:.0f}ms")
    
    # Test generation
    print("\nGenerating response...")
    
    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk['metadata'].get('file_name', 'Unknown')
        page = chunk['metadata'].get('page_number', '?')
        context_parts.append(f"[Source {i}: {source}, Page {page}]\n{chunk['text'][:500]}...")
    
    context = "\n\n---\n\n".join(context_parts)
    
    system_prompt = """You are a legal research assistant. Answer based ONLY on the provided sources.
Cite sources using [Source N] format."""
    
    prompt = f"""Based on the following sources, answer this question: {query}

SOURCES:
{context}

ANSWER:"""
    
    # Time generation
    start = time.time()
    response = ""
    token_count = 0
    
    print("\nStreaming response:")
    print("-" * 40)
    for token in llm_service.generate_stream(prompt, system_prompt=system_prompt):
        response += token
        token_count += 1
        print(token, end="", flush=True)
    
    generation_time = (time.time() - start) * 1000
    tokens_per_sec = token_count / (generation_time / 1000) if generation_time > 0 else 0
    
    print("\n" + "-" * 40)
    print(f"\nGeneration stats:")
    print(f"  Tokens: {token_count}")
    print(f"  Time: {generation_time:.0f}ms")
    print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")
    
    print(f"\nTotal pipeline time:")
    print(f"  Retrieval: {retrieval_time:.0f}ms")
    print(f"  Generation: {generation_time:.0f}ms")
    print(f"  Total: {retrieval_time + generation_time:.0f}ms")
    
    return True


if __name__ == "__main__":
    test_parallel_generation()



