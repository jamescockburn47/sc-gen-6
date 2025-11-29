"""Clear old graph data and re-extract with improved extraction."""

from pathlib import Path
import shutil

from src.documents.catalog import DocumentCatalog
from src.graph.extraction import GraphExtractionService
from src.graph.storage import GraphStore
from src.ingestion.ingestion_pipeline import IngestionPipeline


def main():
    print("=" * 60)
    print("GRAPH RE-EXTRACTION WITH INTELLIGENT PATTERNS")
    print("=" * 60)
    
    # Clear old graph data
    print("\n1. Clearing old graph data...")
    graph_path = Path("data/graph")
    
    # Clear pending
    pending_path = graph_path / "pending"
    if pending_path.exists():
        for f in pending_path.glob("*.json"):
            f.unlink()
        print(f"   Cleared pending folder")
    
    # Clear accepted graph
    accepted_path = graph_path / "graph.jsonl"
    if accepted_path.exists():
        accepted_path.unlink()
        print(f"   Cleared accepted graph")
    
    # Load catalog and extract
    print("\n2. Loading document catalog...")
    catalog = DocumentCatalog()
    records = catalog.list_records()
    
    # Filter to documents with graph enabled
    graph_docs = [r for r in records if r.include_in_graph]
    print(f"   Found {len(graph_docs)} documents with Graph: On")
    
    if not graph_docs:
        print("\nNo documents have graph extraction enabled.")
        return
    
    # Initialize services
    pipeline = IngestionPipeline()
    extractor = GraphExtractionService()
    
    print(f"\n3. Extracting entities from {len(graph_docs)} documents...")
    extracted = 0
    failed = 0
    total_parties = 0
    total_events = 0
    
    for i, record in enumerate(graph_docs):
        file_path = Path(record.file_path)
        
        if not file_path.exists():
            print(f"   [{i+1}/{len(graph_docs)}] SKIP: {record.file_name} (file not found)")
            failed += 1
            continue
        
        try:
            # Re-parse
            parsed = pipeline.parse_document(file_path)
            if not parsed:
                failed += 1
                continue
            
            # Extract
            update = extractor.extract(parsed)
            parties = [n for n in update.nodes if n.node_type == "party"]
            events = [n for n in update.nodes if n.node_type == "event"]
            
            total_parties += len(parties)
            total_events += len(events)
            
            # Queue for review
            extractor.store.queue_update(update)
            extracted += 1
            
            print(f"   [{i+1}/{len(graph_docs)}] {record.file_name}: {len(parties)} parties, {len(events)} events")
            
        except Exception as e:
            print(f"   [{i+1}/{len(graph_docs)}] ERROR: {record.file_name} - {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Documents processed: {extracted}")
    print(f"Documents failed: {failed}")
    print(f"Total parties found: {total_parties}")
    print(f"Total events found: {total_events}")
    print(f"\nPending files in: data/graph/pending/")
    print(f"Click 'Review Case Graph' in app to review and approve.")


if __name__ == "__main__":
    main()


