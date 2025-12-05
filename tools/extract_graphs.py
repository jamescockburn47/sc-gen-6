"""Extract graph entities from already-ingested documents."""

from pathlib import Path
from src.documents.catalog import DocumentCatalog
from src.graph.extraction import GraphExtractionService
from src.ingestion.ingestion_pipeline import IngestionPipeline


def main():
    catalog = DocumentCatalog()
    records = catalog.list_records()
    
    # Filter to documents with graph enabled
    graph_docs = [r for r in records if r.include_in_graph]
    print(f"Found {len(graph_docs)} documents with Graph: On (of {len(records)} total)")
    
    if not graph_docs:
        print("No documents have graph extraction enabled.")
        return
    
    pipeline = IngestionPipeline()
    extractor = GraphExtractionService()
    
    extracted = 0
    failed = 0
    
    for record in graph_docs:
        file_path = Path(record.file_path)
        if not file_path.exists():
            print(f"  SKIP: File not found - {record.file_name}")
            failed += 1
            continue
        
        try:
            # Re-parse the document
            parsed = pipeline.parse_document(file_path)
            if not parsed:
                print(f"  FAIL: Could not parse - {record.file_name}")
                failed += 1
                continue
            
            # Extract and queue graph
            extractor.queue_update(parsed)
            extracted += 1
            print(f"  OK: {record.file_name}")
            
        except Exception as e:
            print(f"  ERROR: {record.file_name} - {e}")
            failed += 1
    
    print(f"\nDone! Extracted: {extracted}, Failed: {failed}")
    print(f"Pending reviews in: data/graph/pending/")
    print(f"Click 'Review Case Graph' in the app to accept extractions.")


if __name__ == "__main__":
    main()



