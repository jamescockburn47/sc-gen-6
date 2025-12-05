"""Benchmark full query pipeline with specified model."""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import get_settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_service import LLMService


def benchmark_query(model_name: str, query: str, context_chunks: int = 10):
    """Run full query benchmark."""
    
    print("=" * 70)
    print(f"QUERY BENCHMARK: {model_name}")
    print("=" * 70)
    
    settings = get_settings()
    
    # Initialize services
    print("\n[1/4] Initializing services...")
    init_start = time.time()
    retriever = HybridRetriever(settings=settings)
    llm_service = LLMService(settings=settings)
    init_time = (time.time() - init_start) * 1000
    print(f"       Init time: {init_time:.0f}ms")
    
    # Warm up retriever
    _ = retriever.retrieve('test', semantic_top_n=5, keyword_top_n=5, rerank_top_k=3)
    
    print(f"\n[2/4] Query: '{query}'")
    print(f"       Model: {model_name}")
    print(f"       Context chunks: {context_chunks}")
    
    # Phase 1: Retrieval
    print("\n[3/4] Retrieval phase...")
    retrieval_start = time.time()
    chunks = retriever.retrieve(query, context_to_llm=context_chunks)
    retrieval_time = (time.time() - retrieval_start) * 1000
    print(f"       Retrieved: {len(chunks)} chunks")
    print(f"       Time: {retrieval_time:.0f}ms")
    
    # Build context
    context_parts = []
    total_context_chars = 0
    for i, chunk in enumerate(chunks, 1):
        source = chunk['metadata'].get('file_name', 'Unknown')
        page = chunk['metadata'].get('page_number', '?')
        text = chunk['text']
        total_context_chars += len(text)
        context_parts.append(f"[Source {i}: {source} | Page {page}]\n{text}")
    
    context = "\n\n---\n\n".join(context_parts)
    print(f"       Context size: {total_context_chars:,} chars (~{total_context_chars//4:,} tokens)")
    
    # Phase 2: Generation
    print("\n[4/4] Generation phase...")
    
    system_prompt = """You are a legal research assistant. Answer based ONLY on the provided sources.
Always cite your sources using [Source N] format. Be precise and factual."""
    
    prompt = f"""Based on the following sources, answer this question: {query}

SOURCES:
{context}

ANSWER:"""
    
    generation_start = time.time()
    response = ""
    token_count = 0
    first_token_time = None
    
    print("\n" + "-" * 70)
    for token in llm_service.generate_stream(
        prompt, 
        system_prompt=system_prompt,
        model=model_name,
        temperature=0.1,
    ):
        if first_token_time is None:
            first_token_time = (time.time() - generation_start) * 1000
        response += token
        token_count += 1
        print(token, end="", flush=True)
    
    generation_time = (time.time() - generation_start) * 1000
    tokens_per_sec = token_count / (generation_time / 1000) if generation_time > 0 else 0
    
    print("\n" + "-" * 70)
    
    # Summary
    total_time = retrieval_time + generation_time
    
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\nModel: {model_name}")
    print(f"Query: '{query[:50]}...'")
    print(f"\nTiming Breakdown:")
    print(f"  Retrieval:        {retrieval_time:>8.0f}ms")
    print(f"  Time to 1st token:{first_token_time:>8.0f}ms")
    print(f"  Generation:       {generation_time:>8.0f}ms")
    print(f"  ─────────────────────────────")
    print(f"  TOTAL:            {total_time:>8.0f}ms ({total_time/1000:.1f}s)")
    print(f"\nGeneration Stats:")
    print(f"  Tokens generated: {token_count}")
    print(f"  Speed:            {tokens_per_sec:.1f} tokens/sec")
    print(f"  Response length:  {len(response)} chars")
    print(f"\nContext Stats:")
    print(f"  Chunks used:      {len(chunks)}")
    print(f"  Context size:     {total_context_chars:,} chars")
    
    return {
        "model": model_name,
        "retrieval_ms": retrieval_time,
        "ttft_ms": first_token_time,
        "generation_ms": generation_time,
        "total_ms": total_time,
        "tokens": token_count,
        "tokens_per_sec": tokens_per_sec,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark query pipeline")
    parser.add_argument("--model", "-m", default="mistral-nemo:12b-instruct-2407-q8_0",
                       help="Model to use")
    parser.add_argument("--query", "-q", 
                       default="What are the key facts of this case and who are the main parties involved?",
                       help="Query to run")
    parser.add_argument("--chunks", "-c", type=int, default=10,
                       help="Number of context chunks")
    
    args = parser.parse_args()
    benchmark_query(args.model, args.query, args.chunks)



