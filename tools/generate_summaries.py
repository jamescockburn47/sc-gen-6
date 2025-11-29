"""Generate document summaries for all indexed documents.

Run this script to generate summaries for your document corpus.
Uses a dedicated smaller/faster model for efficient bulk processing.

Usage:
    python tools/generate_summaries.py                    # Use default model
    python tools/generate_summaries.py --model qwen2.5:3b # Use specific model
    python tools/generate_summaries.py --estimate         # Just show time estimate
    python tools/generate_summaries.py --parallel 4       # Use 4 parallel workers
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config_loader import get_settings
from src.retrieval.vector_store import VectorStore
from src.retrieval.fts5_index import FTS5Index
from src.retrieval.summary_store import SummaryStore
from src.generation.summarizer import SummarizerService, estimate_summarization_time


def get_documents_to_summarize(summary_store: SummaryStore, fts5_index: FTS5Index) -> list[dict]:
    """Get documents that need summaries.
    
    Args:
        summary_store: Summary store to check existing summaries
        fts5_index: FTS5 index to get document data
        
    Returns:
        List of document dicts needing summaries
    """
    # Get all unique documents from FTS5 index
    conn = fts5_index._get_conn()
    cursor = conn.execute("""
        SELECT DISTINCT document_id, file_name, document_type
        FROM chunks
    """)
    
    all_docs = []
    for row in cursor.fetchall():
        doc_id = row[0]
        file_name = row[1]
        doc_type = row[2] or "unknown"
        
        # Check if summary already exists
        existing = summary_store.get_document_summaries(
            document_id=doc_id,
            summary_level="document",
        )
        
        if not existing:
            # Get document text from chunks
            chunk_cursor = conn.execute("""
                SELECT chunk_text FROM chunks 
                WHERE document_id = ? 
                ORDER BY id
            """, (doc_id,))
            
            text_parts = [r[0] for r in chunk_cursor.fetchall()]
            full_text = "\n\n".join(text_parts)
            
            all_docs.append({
                "document_id": doc_id,
                "file_name": file_name,
                "doc_type": doc_type,
                "text": full_text,
            })
    
    return all_docs


def main():
    parser = argparse.ArgumentParser(
        description="Generate document summaries using a dedicated summarization model."
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="Model to use for summarization (default: from config)",
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)",
    )
    parser.add_argument(
        "--estimate", "-e",
        action="store_true",
        help="Only show time estimate, don't generate",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Regenerate all summaries (even existing ones)",
    )
    parser.add_argument(
        "--types", "-t",
        type=str,
        default="overview",
        help="Summary types to generate (comma-separated: overview,key_points,entities,timeline)",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt",
    )
    
    args = parser.parse_args()
    
    settings = get_settings()
    model = args.model or settings.summary.summarization_model
    summary_types = [t.strip() for t in args.types.split(",")]
    
    print("=" * 60)
    print("SC Gen 6 - Document Summary Generation")
    print("=" * 60)
    print()
    
    # Initialize services
    print("Loading indexes...")
    fts5_index = FTS5Index(settings=settings)
    summary_store = SummaryStore(settings=settings)
    
    # Get documents needing summaries
    if args.force:
        # Get all documents
        conn = fts5_index._get_conn()
        cursor = conn.execute("""
            SELECT DISTINCT document_id, file_name, document_type
            FROM chunks
        """)
        
        documents = []
        for row in cursor.fetchall():
            chunk_cursor = conn.execute("""
                SELECT chunk_text FROM chunks 
                WHERE document_id = ? 
                ORDER BY id
            """, (row[0],))
            
            text_parts = [r[0] for r in chunk_cursor.fetchall()]
            full_text = "\n\n".join(text_parts)
            
            documents.append({
                "document_id": row[0],
                "file_name": row[1],
                "doc_type": row[2] or "unknown",
                "text": full_text,
            })
    else:
        documents = get_documents_to_summarize(summary_store, fts5_index)
    
    if not documents:
        print("[OK] All documents already have summaries!")
        print()
        print(f"Summary store stats:")
        stats = summary_store.stats()
        print(f"  Total summaries: {stats['total_summaries']}")
        print(f"  Documents with summaries: {stats['documents_with_summaries']}")
        return
    
    print(f"Documents to summarize: {len(documents)}")
    print(f"Summary types: {summary_types}")
    print(f"Model: {model}")
    print(f"Parallel workers: {args.parallel}")
    print()
    
    # Show time estimate
    estimate = estimate_summarization_time(
        num_documents=len(documents),
        model=model,
        summary_types=len(summary_types),
    )
    
    print("Time Estimate:")
    print(f"  Total summaries: {estimate['total_summaries']}")
    print(f"  Model speed: ~{estimate['estimated_speed_tok_s']} tok/s")
    print(f"  Time per summary: ~{estimate['seconds_per_summary']:.1f}s")
    print(f"  Estimated total time: {estimate['formatted_time']}")
    print()
    
    if args.estimate:
        return
    
    # Confirm before proceeding
    if not args.yes:
        response = input(f"Generate summaries for {len(documents)} documents? [y/N] ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    print()
    print("Starting summarization...")
    print("-" * 60)
    
    # Create summarizer service
    summarizer = SummarizerService(settings=settings)
    
    # Check/pull model
    print(f"Checking model availability: {model}")
    success, message = summarizer.ensure_model_available(model)
    if not success:
        print(f"[ERROR] {message}")
        return
    print(f"[OK] {message}")
    print()
    
    # Progress tracking
    import time
    start_time = time.time()
    
    def on_progress(completed: int, total: int, current_file: str):
        elapsed = time.time() - start_time
        if completed > 0:
            eta = (elapsed / completed) * (total - completed)
            eta_str = f"{int(eta // 60)}m {int(eta % 60)}s"
        else:
            eta_str = "calculating..."
        
        print(f"\r[{completed}/{total}] {current_file[:40]:40s} | ETA: {eta_str}    ", end="", flush=True)
    
    # Generate summaries (use parallel for multiple workers)
    try:
        if args.parallel > 1:
            # Progress handled differently for parallel
            print(f"Starting parallel summarization with {args.parallel} workers...")
            summaries = summarizer.summarize_documents_parallel(
                documents=documents,
                summary_types=summary_types,
                model=model,
                max_workers=args.parallel,
            )
        else:
            summaries = summarizer.summarize_documents(
                documents=documents,
                summary_types=summary_types,
                model=model,
                on_progress=on_progress,
            )
        
        total_time = time.time() - start_time
        
        print()
        print()
        print("-" * 60)
        print(f"[DONE] Generated {len(summaries)} summaries in {int(total_time // 60)}m {int(total_time % 60)}s")
        print()
        
        # Show final stats
        stats = summary_store.stats()
        print("Summary Store Stats:")
        print(f"  Total summaries: {stats['total_summaries']}")
        print(f"  Documents with summaries: {stats['documents_with_summaries']}")
        print(f"  By type: {stats['by_type']}")
        
    except KeyboardInterrupt:
        print()
        print()
        print("Cancelled by user.")
        summarizer.cancel()


if __name__ == "__main__":
    main()

