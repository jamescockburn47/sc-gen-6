#!/usr/bin/env python3
"""Benchmark script for SC Gen 6 RAG system.

Runs 8 competition law queries and records:
- Latency (seconds)
- Errors
- Number of chunks retrieved
- Top 3 chunk previews
- Answer preview

Results saved as timestamped JSON in benchmarks/ directory.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import get_settings
from src.config.llm_config import load_llm_config


BENCHMARK_QUERIES = [
    "What is the test for abuse of dominant position under section 18 Competition Act 1998?",
    "What is the limitation period for follow-on competition damages claims in the CAT?",
    "What are the requirements for a collective proceedings order under section 47B Competition Act 1998?",
    "Explain the pass-on defence in competition law damages claims",
    "What did the Supreme Court decide in Merricks v Mastercard about certification?",
    "How is the relevant market defined for Chapter I and Chapter II prohibitions?",
    "What is the standard of proof for binding infringement decisions in follow-on claims?",
    "Describe the disclosure obligations in CAT proceedings",
]


def run_benchmark(
    label: str = "default",
    include_generation: bool = True,
    output_dir: str = "benchmarks",
) -> dict:
    """Run the benchmark suite.
    
    Args:
        label: Label for this benchmark run
        include_generation: Whether to test LLM generation (requires model)
        output_dir: Directory to save results
        
    Returns:
        Dictionary with benchmark results
    """
    settings = get_settings()
    llm_config = load_llm_config()
    
    # Initialize retrieval pipeline
    print("=== SC Gen 6 RAG Benchmark ===")
    print(f"Label: {label}")
    print(f"Provider: {llm_config.provider}")
    print(f"Model: {llm_config.model_name}")
    print(f"Embedding: {settings.models.embedding.default}")
    print(f"Generation: {'enabled' if include_generation else 'disabled'}")
    print()
    
    from src.retrieval import get_embedding_service, HybridRetriever, VectorStore, FTS5IndexCompat
    from src.models.reranker import get_reranker_service
    
    print("[INIT] Loading services...")
    init_start = time.time()
    
    embed_svc = get_embedding_service(settings=settings)
    vs = VectorStore(settings=settings)
    fts = FTS5IndexCompat(settings=settings)
    reranker = get_reranker_service(settings=settings)
    
    retriever = HybridRetriever(
        embedding_service=embed_svc,
        vector_store=vs,
        keyword_index=fts,
        reranker_service=reranker,
        settings=settings,
    )
    
    llm_service = None
    if include_generation:
        from src.generation.llm_service import LLMService
        llm_service = LLMService(settings=settings)
    
    init_time = time.time() - init_start
    print(f"[INIT] Services loaded in {init_time:.1f}s")
    print(f"[INIT] Vector store: {vs.collection.count()} chunks")
    print()
    
    # Run queries
    results = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "provider": llm_config.provider,
            "model": llm_config.model_name,
            "embedding_model": settings.models.embedding.default,
            "reranker_model": settings.models.reranker.default,
            "vector_store_count": vs.collection.count(),
            "generation_enabled": include_generation,
        },
        "init_time_s": round(init_time, 2),
        "queries": [],
    }
    
    total_retrieval_time = 0
    total_generation_time = 0
    errors = 0
    
    for i, query in enumerate(BENCHMARK_QUERIES, 1):
        print(f"[{i}/8] {query[:60]}...")
        result = {
            "query": query,
            "retrieval_error": None,
            "generation_error": None,
            "retrieval_time_s": 0,
            "generation_time_s": 0,
            "total_time_s": 0,
            "chunks_retrieved": 0,
            "top_chunks": [],
            "answer_preview": "",
        }
        
        # Retrieval
        try:
            ret_start = time.time()
            chunks = retriever.retrieve(query)
            ret_time = time.time() - ret_start
            result["retrieval_time_s"] = round(ret_time, 2)
            total_retrieval_time += ret_time
            
            # Extract chunk info
            if isinstance(chunks, list):
                result["chunks_retrieved"] = len(chunks)
                for j, chunk in enumerate(chunks[:3]):
                    if isinstance(chunk, dict):
                        text = chunk.get("text", "")[:200]
                        chunk_id = chunk.get("chunk_id", "")
                    else:
                        text = getattr(chunk, "text", str(chunk))[:200]
                        chunk_id = getattr(chunk, "chunk_id", "")
                    result["top_chunks"].append({
                        "rank": j + 1,
                        "chunk_id": chunk_id,
                        "text_preview": text,
                    })
            
            print(f"       Retrieval: {ret_time:.1f}s, {result['chunks_retrieved']} chunks")
            
        except Exception as e:
            result["retrieval_error"] = str(e)
            errors += 1
            print(f"       RETRIEVAL ERROR: {e}")
        
        # Generation
        if include_generation and llm_service and result["retrieval_error"] is None:
            try:
                gen_start = time.time()
                answer = llm_service.generate_with_context(
                    query=query,
                    chunks=chunks,
                    stream=False,
                )
                gen_time = time.time() - gen_start
                result["generation_time_s"] = round(gen_time, 2)
                total_generation_time += gen_time
                
                if isinstance(answer, str):
                    result["answer_preview"] = answer[:500]
                elif isinstance(answer, dict):
                    result["answer_preview"] = str(answer.get("text", answer))[:500]
                else:
                    result["answer_preview"] = str(answer)[:500]
                
                print(f"       Generation: {gen_time:.1f}s")
                print(f"       Answer: {result['answer_preview'][:100]}...")
                
            except Exception as e:
                result["generation_error"] = str(e)
                errors += 1
                print(f"       GENERATION ERROR: {e}")
        
        result["total_time_s"] = round(
            result["retrieval_time_s"] + result["generation_time_s"], 2
        )
        results["queries"].append(result)
        print()
    
    # Summary
    retrieval_times = [q["retrieval_time_s"] for q in results["queries"] if q["retrieval_error"] is None]
    generation_times = [q["generation_time_s"] for q in results["queries"] if q["generation_error"] is None and q["generation_time_s"] > 0]
    
    results["summary"] = {
        "total_queries": len(BENCHMARK_QUERIES),
        "retrieval_errors": sum(1 for q in results["queries"] if q["retrieval_error"]),
        "generation_errors": sum(1 for q in results["queries"] if q["generation_error"]),
        "avg_retrieval_time_s": round(sum(retrieval_times) / max(len(retrieval_times), 1), 2),
        "avg_generation_time_s": round(sum(generation_times) / max(len(generation_times), 1), 2) if generation_times else 0,
        "avg_total_time_s": round(
            (sum(retrieval_times) + sum(generation_times)) / max(len(BENCHMARK_QUERIES), 1), 2
        ),
        "avg_chunks_retrieved": round(
            sum(q["chunks_retrieved"] for q in results["queries"]) / max(len(BENCHMARK_QUERIES), 1), 1
        ),
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{label}_{timestamp}.json"
    filepath = output_path / filename
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    print("=" * 60)
    print(f"BENCHMARK RESULTS: {label}")
    print(f"  Queries: {results['summary']['total_queries']}")
    print(f"  Retrieval errors: {results['summary']['retrieval_errors']}")
    print(f"  Generation errors: {results['summary']['generation_errors']}")
    print(f"  Avg retrieval: {results['summary']['avg_retrieval_time_s']}s")
    print(f"  Avg generation: {results['summary']['avg_generation_time_s']}s")
    print(f"  Avg total: {results['summary']['avg_total_time_s']}s")
    print(f"  Avg chunks: {results['summary']['avg_chunks_retrieved']}")
    print(f"  Results saved to: {filepath}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SC Gen 6 RAG Benchmark")
    parser.add_argument("--label", default="default", help="Label for this run")
    parser.add_argument("--no-generation", action="store_true", help="Skip LLM generation")
    parser.add_argument("--output-dir", default="benchmarks", help="Output directory")
    
    args = parser.parse_args()
    
    run_benchmark(
        label=args.label,
        include_generation=not args.no_generation,
        output_dir=args.output_dir,
    )
