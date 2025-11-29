"""Test parallel batch generation with ChunkBatchGenerator."""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import get_settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_service import LLMService
from src.generation.chunk_batcher import ChunkBatchGenerator


def test_batch_generation():
    print("=" * 60)
    print("PARALLEL BATCH GENERATION TEST")
    print("=" * 60)
    
    settings = get_settings()
    print(f"\nBatch Config:")
    print(f"  enable_batching: {settings.generation.enable_batching}")
    print(f"  min_chunks_for_batching: {settings.generation.min_chunks_for_batching}")
    print(f"  chunk_batch_size: {settings.generation.chunk_batch_size}")
    print(f"  parallel_workers: {settings.generation.parallel_workers}")
    
    # Initialize services
    print("\nInitializing services...")
    retriever = HybridRetriever(settings=settings)
    llm_service = LLMService(settings=settings)
    batch_generator = ChunkBatchGenerator(llm_service)
    
    # Warm up
    _ = retriever.retrieve('test', semantic_top_n=5, keyword_top_n=5, rerank_top_k=3)
    
    query = "Summarize all the key evidence and claims in this case"
    
    # Get many chunks to trigger batching
    print(f"\nQuery: '{query}'")
    print("Retrieving many chunks to trigger batching...")
    
    chunks = retriever.retrieve(query, context_to_llm=16)  # 16 chunks
    print(f"  Retrieved {len(chunks)} chunks")
    
    # Check if batching will be triggered
    will_batch = len(chunks) >= settings.generation.min_chunks_for_batching
    expected_batches = (len(chunks) + settings.generation.chunk_batch_size - 1) // settings.generation.chunk_batch_size
    
    print(f"\nBatching decision:")
    print(f"  Chunks: {len(chunks)} >= {settings.generation.min_chunks_for_batching} (min)? {will_batch}")
    print(f"  Expected batches: {expected_batches} (batch_size={settings.generation.chunk_batch_size})")
    print(f"  Parallel workers: {settings.generation.parallel_workers}")
    
    if not will_batch:
        print("\n[SKIP] Not enough chunks to trigger batching")
        print(f"  Need at least {settings.generation.min_chunks_for_batching} chunks")
        return
    
    # Run batch generation
    print("\nStarting batch generation...")
    print("-" * 40)
    
    start = time.time()
    
    # Track tokens for display
    tokens_received = []
    def on_token(token):
        tokens_received.append(token)
        print(token, end="", flush=True)
    
    # Use batch generation with token callback
    response_text, stats = batch_generator.generate(
        query=query,
        chunks=chunks,
        token_callback=on_token,
        cancel_event=None,
    )
    
    generation_time = (time.time() - start) * 1000
    
    print("\n" + "-" * 40)
    
    # Stats are returned from generate()
    print(f"\nBatch Generation Stats:")
    print(f"  Total batches: {stats.get('total_batches', 'N/A')}")
    print(f"  Batch times: {stats.get('batch_times', [])}")
    print(f"  Parallel workers config: {settings.generation.parallel_workers}")
    print(f"  Total time: {generation_time:.0f}ms")
    print(f"  Response length: {len(response_text)} chars")
    
    total_batches = stats.get('total_batches', 0)
    if total_batches > 1:
        print(f"\n[OK] Parallel batch generation completed ({total_batches} batches)!")
    else:
        print("\n[INFO] Single batch - no parallelism needed")


if __name__ == "__main__":
    test_batch_generation()

