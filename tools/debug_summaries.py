"""Debug summary integration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import get_settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.summary_store import SummaryStore


def debug_summaries():
    settings = get_settings()
    print(f"use_summaries setting: {settings.retrieval.use_summaries}")
    
    summary_store = SummaryStore(settings=settings)
    
    # Check what summaries exist
    stats = summary_store.stats()
    print(f"\nSummary store stats: {stats}")
    
    # Get a sample summary
    conn = summary_store._get_conn()
    cursor = conn.execute("SELECT document_id, file_name, content FROM document_summaries LIMIT 1")
    row = cursor.fetchone()
    if row:
        print(f"\nSample summary from store:")
        print(f"  document_id: {row[0][:80]}...")
        print(f"  file_name: {row[1]}")
        print(f"  content: {row[2][:100]}...")
    
    # Initialize retriever
    retriever = HybridRetriever(settings=settings)
    
    # Warm up
    _ = retriever.retrieve("test", semantic_top_n=5, keyword_top_n=5, rerank_top_k=3)
    
    # Retrieve chunks
    results = retriever.retrieve("key facts", context_to_llm=3, skip_reranking=True)
    
    print(f"\n\nRetrieved {len(results)} chunks")
    if results:
        r = results[0]
        print(f"\nChunk structure:")
        print(f"  Top-level keys: {list(r.keys())}")
        print(f"  metadata keys: {list(r.get('metadata', {}).keys())}")
        
        # Check document_id sources
        top_level_doc_id = r.get("document_id", "NOT FOUND")
        meta_doc_id = r.get("metadata", {}).get("document_id", "NOT FOUND")
        
        print(f"\n  result['document_id']: {top_level_doc_id[:50] if top_level_doc_id != 'NOT FOUND' else 'NOT FOUND'}...")
        print(f"  result['metadata']['document_id']: {meta_doc_id[:50] if meta_doc_id != 'NOT FOUND' else 'NOT FOUND'}...")
        
        # Check if enhance_with_summaries is looking in the right place
        print(f"\n  has 'document_summary': {'document_summary' in r}")
    
    # Manually test enhance_with_summaries
    print("\n\nManually testing enhance_with_summaries:")
    
    # First, let's check what document IDs are in the summary store
    cursor = conn.execute("SELECT DISTINCT document_id FROM document_summaries LIMIT 5")
    stored_doc_ids = [row[0] for row in cursor.fetchall()]
    print(f"  Sample document_ids in summary store: {stored_doc_ids[:2]}")
    
    # And what's in the chunks
    if results:
        chunk_doc_ids = [r.get("metadata", {}).get("document_id", r.get("document_id", "NONE")) for r in results]
        print(f"  Document IDs from chunks: {chunk_doc_ids}")


if __name__ == "__main__":
    debug_summaries()


