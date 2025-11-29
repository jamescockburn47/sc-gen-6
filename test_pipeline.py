"""Full diagnostic test of the RAG pipeline.

Tests:
1. Document ingestion
2. Vector search
3. Keyword search
4. Hybrid retrieval
5. Query generation
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.config_loader import get_settings
from src.retrieval.vector_store import VectorStore
from src.retrieval.fts5_index import FTS5Index
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_engine import QueryEngine

def test_pipeline():
    print("=" * 80)
    print("RAG PIPELINE DIAGNOSTIC TEST")
    print("=" * 80)
    
    settings = get_settings()
    
    # Test 1: Vector Store
    print("\n[1/5] Testing Vector Store (ChromaDB)...")
    try:
        vector_store = VectorStore(settings=settings)
        stats = vector_store.stats()
        print(f"✓ Vector store loaded")
        print(f"  - Total chunks: {stats.get('total_chunks', 0)}")
        print(f"  - Unique files: {stats.get('unique_files', 0)}")
        if stats.get('unique_documents'):
            print(f"  - Documents: {stats['unique_documents']}")
    except Exception as e:
        print(f"✗ Vector store error: {e}")
        return
    
    # Test 2: FTS5 Index
    print("\n[2/5] Testing FTS5 Keyword Index...")
    try:
        fts5 = FTS5Index(settings=settings)
        fts5_stats = fts5.stats()
        print(f"✓ FTS5 index loaded")
        print(f"  - Total chunks: {fts5_stats.get('total_chunks', 0)}")
    except Exception as e:
        print(f"✗ FTS5 error: {e}")
        return
    
    # Test 3: Hybrid Retriever
    print("\n[3/5] Testing Hybrid Retriever...")
    try:
        retriever = HybridRetriever(settings=settings)
        print(f"✓ Hybrid retriever initialized")
        print(f"  - Reranker device: {getattr(retriever.reranker_service, 'device_label', 'Unknown')}")
    except Exception as e:
        print(f"✗ Retriever error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Retrieval
    print("\n[4/5] Testing Retrieval...")
    test_query = "What is this document about?"
    try:
        results = retriever.retrieve(query=test_query)
        print(f"✓ Retrieved {len(results)} results")
        if results:
            print(f"  - Top result: {results[0].get('text', '')[:100]}...")
            print(f"  - From: {results[0].get('metadata', {}).get('file_name', 'Unknown')}")
    except Exception as e:
        print(f"✗ Retrieval error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 5: Query Engine
    print("\n[5/5] Testing Query Engine...")
    try:
        engine = QueryEngine(settings)
        print(f"✓ Query engine initialized")
        
        print(f"\nRunning test query: '{test_query}'")
        print("-" * 80)
        
        response_text = ""
        source_count = 0
        
        for chunk in engine.query(test_query):
            if "token" in chunk:
                response_text += chunk["token"]
                print(chunk["token"], end="", flush=True)
            elif "source" in chunk:
                source_count += 1
            elif "error" in chunk:
                print(f"\n✗ Generation error: {chunk['error']}")
                return
        
        print("\n" + "-" * 80)
        print(f"✓ Query completed")
        print(f"  - Response length: {len(response_text)} chars")
        print(f"  - Sources used: {source_count}")
        
    except Exception as e:
        print(f"✗ Query engine error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED ✓")
    print("=" * 80)

if __name__ == "__main__":
    test_pipeline()
