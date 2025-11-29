"""Clear all existing documents from the RAG system.

This script will:
1. Clear ChromaDB vector store
2. Clear FTS5 keyword index
3. Clear BM25 index (if exists)
4. Clear document catalog
5. Clear summary store
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.config_loader import get_settings
from src.retrieval.vector_store import VectorStore
from src.retrieval.fts5_index import FTS5Index
from src.retrieval.bm25_index import BM25Index
from src.documents.catalog import DocumentCatalog
from src.retrieval.summary_store import SummaryStore

def clear_all_documents():
    """Clear all documents from all stores."""
    settings = get_settings()
    
    print("Clearing all documents from RAG system...")
    print("=" * 60)
    
    # 1. Clear vector store
    try:
        print("\n[1/5] Clearing ChromaDB vector store...")
        vector_store = VectorStore(settings=settings)
        vector_store.reset()
        print("✓ ChromaDB cleared")
    except Exception as e:
        print(f"✗ Error clearing ChromaDB: {e}")
    
    # 2. Clear FTS5 index
    try:
        print("\n[2/5] Clearing FTS5 keyword index...")
        fts5_index = FTS5Index(settings=settings)
        fts5_index.reset()
        print("✓ FTS5 index cleared")
    except Exception as e:
        print(f"✗ Error clearing FTS5: {e}")
    
    # 3. Clear BM25 index
    try:
        print("\n[3/5] Clearing BM25 index...")
        bm25_index = BM25Index(settings=settings)
        bm25_index.reset()
        print("✓ BM25 index cleared")
    except Exception as e:
        print(f"✗ Error clearing BM25: {e}")
    
    # 4. Clear document catalog
    try:
        print("\n[4/5] Clearing document catalog...")
        catalog_path = Path("data/documents/catalog.json")
        if catalog_path.exists():
            catalog_path.unlink()
            print("✓ Document catalog cleared")
        else:
            print("✓ Document catalog already empty")
    except Exception as e:
        print(f"✗ Error clearing catalog: {e}")
    
    # 5. Clear summary store
    try:
        print("\n[5/5] Clearing summary store...")
        summary_store = SummaryStore(settings=settings)
        summary_store.reset()
        print("✓ Summary store cleared")
    except Exception as e:
        print(f"✗ Error clearing summaries: {e}")
    
    print("\n" + "=" * 60)
    print("All documents cleared! Ready for re-ingestion.")

if __name__ == "__main__":
    clear_all_documents()
