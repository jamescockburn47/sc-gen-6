"""Fix catalog records by marking documents that exist in indexes as indexed."""

from datetime import datetime
from src.documents.catalog import DocumentCatalog
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_index import BM25Index


def main():
    # Load catalog
    catalog = DocumentCatalog()
    records = catalog.list_records()
    print(f"Catalog: {len(records)} records")

    # Get indexed docs from vector store
    vs = VectorStore()
    vs_stats = vs.stats(include_documents=True)
    indexed_files = set(vs_stats.get("unique_documents", []))
    print(f"Vector store: {len(indexed_files)} unique docs, {vs_stats.get('total_chunks', 0)} chunks")

    # Load BM25
    bm25_files = set()
    try:
        bm25 = BM25Index()
        bm25.load()
        bm25_files = set(c.file_name for c in bm25.chunks if hasattr(c, "file_name"))
        print(f"BM25: {len(bm25_files)} unique docs, {len(bm25.chunks)} chunks")
    except Exception as e:
        print(f"BM25 error: {e}")

    # Combine all indexed files
    all_indexed = indexed_files | bm25_files
    print(f"Total unique indexed files: {len(all_indexed)}")

    # Update catalog records for indexed docs
    updated = 0
    timestamp = datetime.now().isoformat()
    for record in records:
        if record.file_name in all_indexed:
            if not record.indexed:
                record.indexed = True
                record.ingested_at = timestamp
                catalog.update_record(record)
                updated += 1
                print(f"  Updated: {record.file_name}")

    print(f"\nUpdated {updated} records to indexed=True")
    
    # Show summary
    records = catalog.list_records()
    indexed_count = sum(1 for r in records if r.indexed)
    print(f"Now: {indexed_count}/{len(records)} documents marked as indexed")


if __name__ == "__main__":
    main()


