"""Test reranking quality with truncation vs full text."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import get_settings
from src.retrieval.hybrid_retriever import HybridRetriever


def test_truncation_quality():
    settings = get_settings()
    retriever = HybridRetriever(settings=settings)
    
    # Warm up
    _ = retriever.retrieve('test', semantic_top_n=5, keyword_top_n=5, rerank_top_k=3)
    
    query = 'What are the key facts involving the defendant?'
    
    print("=" * 60)
    print("RERANKING QUALITY: Truncated vs Full Text")
    print("=" * 60)
    
    # Test with truncation (current config: 512 chars)
    print(f"\nTruncation setting: {settings.retrieval.rerank_max_chars} chars")
    r1 = retriever.retrieve(query, rerank_top_k=15)
    ids_truncated = [r['chunk_id'] for r in r1]
    
    # Test with full text (set very high truncation)
    settings.retrieval.rerank_max_chars = 10000
    r2 = retriever.retrieve(query, rerank_top_k=15)
    ids_full = [r['chunk_id'] for r in r2]
    
    # Compare ranking overlap
    overlap = len(set(ids_truncated) & set(ids_full))
    same_top_3 = ids_truncated[:3] == ids_full[:3]
    same_top_5 = ids_truncated[:5] == ids_full[:5]
    
    print(f"\nRanking Quality Comparison:")
    print(f"  Overlap in top 15: {overlap}/15 ({overlap/15*100:.0f}%)")
    print(f"  Same top 3 results: {same_top_3}")
    print(f"  Same top 5 results: {same_top_5}")
    
    print(f"\nTop 5 (truncated 512 chars):")
    for i, r in enumerate(r1[:5], 1):
        print(f"  {i}. [{r['score']:.3f}] {r['metadata']['file_name']}")
    
    print(f"\nTop 5 (full text):")
    for i, r in enumerate(r2[:5], 1):
        print(f"  {i}. [{r['score']:.3f}] {r['metadata']['file_name']}")
    
    if overlap >= 12:  # 80%+ overlap
        print("\n[OK] Truncation has minimal impact on ranking quality")
    elif overlap >= 10:  # 66%+ overlap
        print("\n[WARNING] Truncation has moderate impact - consider increasing rerank_max_chars")
    else:
        print("\n[CRITICAL] Truncation significantly impacts ranking - increase rerank_max_chars")


if __name__ == "__main__":
    test_truncation_quality()



