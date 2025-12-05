"""Test parallel optimizations for RAG pipeline.

Verifies that parallel retrieval and generation are working correctly.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import get_settings


def test_config_values():
    """Verify optimized config values are loaded."""
    settings = get_settings()
    
    print("=" * 60)
    print("OPTIMIZED CONFIGURATION VALUES")
    print("=" * 60)
    
    print("\nPerformance Settings:")
    print(f"  embed_batch_size:  {settings.performance.embed_batch_size} (optimal: 64)")
    print(f"  rerank_batch_size: {settings.performance.rerank_batch_size} (optimal: 32)")
    print(f"  max_workers:       {settings.performance.max_workers} (optimal: 8)")
    
    print("\nGeneration Settings:")
    print(f"  chunk_batch_size:  {settings.generation.chunk_batch_size} (optimal: 4)")
    print(f"  max_batches:       {settings.generation.max_batches} (optimal: 8)")
    print(f"  parallel_workers:  {settings.generation.parallel_workers} (optimal: 8)")
    
    print("\nSummary Settings:")
    print(f"  parallel_summaries: {settings.summary.parallel_summaries} (optimal: 8)")
    
    # Verify values
    issues = []
    if settings.performance.max_workers < 8:
        issues.append(f"max_workers={settings.performance.max_workers} < 8")
    if settings.generation.parallel_workers < 8:
        issues.append(f"generation.parallel_workers={settings.generation.parallel_workers} < 8")
    if settings.summary.parallel_summaries < 8:
        issues.append(f"summary.parallel_summaries={settings.summary.parallel_summaries} < 8")
    
    if issues:
        print(f"\n[WARNING] Sub-optimal settings: {', '.join(issues)}")
    else:
        print("\n[OK] All settings optimized for your hardware!")


def test_parallel_retrieval():
    """Test that parallel retrieval is working."""
    print("\n" + "=" * 60)
    print("PARALLEL RETRIEVAL TEST")
    print("=" * 60)
    
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.vector_store import VectorStore
    from src.retrieval.fts5_index import FTS5IndexCompat
    
    settings = get_settings()
    
    # Check if indexes exist
    vs = VectorStore(settings=settings)
    fts5 = FTS5IndexCompat(settings=settings)
    
    vs_stats = vs.stats()
    fts5_stats = fts5.stats()
    
    print(f"\nVector Store: {vs_stats.get('total_chunks', 0)} chunks")
    print(f"FTS5 Index: {fts5_stats.get('total_chunks', 0)} chunks")
    
    if vs_stats.get('total_chunks', 0) == 0:
        print("\n[SKIP] No documents indexed - cannot test retrieval")
        return
    
    # Test retrieval
    retriever = HybridRetriever(settings=settings)
    
    query = "What are the key facts of this case?"
    
    # Sequential baseline (simulated)
    print(f"\nTesting query: '{query}'")
    
    start = time.time()
    results = retriever.retrieve(query, semantic_top_n=20, keyword_top_n=20, rerank_top_k=10)
    elapsed = time.time() - start
    
    print(f"\nRetrieval completed:")
    print(f"  Results: {len(results)} chunks")
    print(f"  Time: {elapsed*1000:.0f}ms")
    
    if elapsed < 2.0:
        print(f"  [OK] Fast retrieval ({elapsed*1000:.0f}ms)")
    else:
        print(f"  [WARNING] Slow retrieval ({elapsed*1000:.0f}ms)")


def test_ollama_parallelism():
    """Test Ollama can handle parallel requests."""
    print("\n" + "=" * 60)
    print("OLLAMA PARALLELISM TEST")
    print("=" * 60)
    
    import concurrent.futures
    from src.llm.client import LLMClient
    from src.config.llm_config import load_llm_config
    
    config = load_llm_config()
    
    def test_request(i):
        client = LLMClient(config)
        messages = [{'role': 'user', 'content': f'Say "test {i}" in one word'}]
        start = time.time()
        client.generate_chat_completion(messages, temperature=0.0)
        return time.time() - start
    
    # Test 4 concurrent requests
    print("\nTesting 4 concurrent requests...")
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(test_request, i) for i in range(4)]
        times = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    total = time.time() - start
    sequential_est = sum(times)
    
    print(f"\n  4 requests completed in: {total:.1f}s")
    print(f"  Individual times: {[f'{t:.1f}s' for t in times]}")
    print(f"  If sequential: ~{sequential_est:.1f}s")
    print(f"  Speedup: {sequential_est/total:.1f}x")
    
    if total < sequential_est * 0.7:
        print(f"  [OK] Ollama IS processing in parallel!")
    else:
        print(f"  [WARNING] Ollama may be processing sequentially")


def main():
    print("=" * 60)
    print("SC Gen 6 - Optimization Verification")
    print("=" * 60)
    
    test_config_values()
    test_parallel_retrieval()
    test_ollama_parallelism()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()



