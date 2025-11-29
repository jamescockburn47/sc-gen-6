"""Migrate BM25 pickle index to FTS5 SQLite index.

Run this script to migrate existing BM25 data to the new FTS5 format.
The old BM25 files will be preserved as backups.
"""

import pickle
from pathlib import Path
import shutil
from datetime import datetime

from src.config_loader import get_settings
from src.retrieval.fts5_index import FTS5Index


def migrate():
    """Migrate BM25 pickle files to FTS5 SQLite database."""
    settings = get_settings()
    index_path = Path(settings.paths.bm25_index)
    
    # Check for existing BM25 files
    bm25_index_file = index_path / "bm25_index.pkl"
    bm25_chunks_file = index_path / "bm25_chunks.pkl"
    
    if not bm25_chunks_file.exists():
        print("No BM25 chunks file found. Nothing to migrate.")
        return
    
    print(f"Found BM25 index at: {index_path}")
    
    # Load existing chunks
    print("Loading BM25 chunks...")
    try:
        with open(bm25_chunks_file, "rb") as f:
            data = pickle.load(f)
            chunks = data.get("chunks", [])
    except Exception as e:
        print(f"Error loading BM25 chunks: {e}")
        return
    
    print(f"Loaded {len(chunks)} chunks from BM25 index")
    
    if not chunks:
        print("No chunks to migrate.")
        return
    
    # Create backup of old files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = index_path / f"bm25_backup_{timestamp}"
    backup_dir.mkdir(exist_ok=True)
    
    if bm25_index_file.exists():
        shutil.copy(bm25_index_file, backup_dir / bm25_index_file.name)
    if bm25_chunks_file.exists():
        shutil.copy(bm25_chunks_file, backup_dir / bm25_chunks_file.name)
    
    print(f"Created backup at: {backup_dir}")
    
    # Create FTS5 index and add chunks
    print("Creating FTS5 index...")
    fts5_index = FTS5Index(index_path=index_path, settings=settings)
    
    print(f"Adding {len(chunks)} chunks to FTS5 index...")
    fts5_index.add_chunks(chunks)
    
    # Verify migration
    stats = fts5_index.stats()
    print(f"\nMigration complete!")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Documents: {stats['document_count']}")
    print(f"  Files: {stats['file_count']}")
    print(f"  Index path: {stats['index_path']}")
    
    # Test search
    print("\nTesting FTS5 search...")
    test_results = fts5_index.search("witness statement", top_k=3)
    print(f"  Test query 'witness statement': {len(test_results)} results")
    
    print("\n✓ Migration successful!")
    print(f"  Old BM25 files backed up to: {backup_dir}")
    print(f"  New FTS5 database: {fts5_index.db_path}")
    
    # Update config suggestion
    print("\nTo use FTS5 by default, ensure your config.yaml has:")
    print("  retrieval:")
    print("    keyword_backend: fts5")


if __name__ == "__main__":
    migrate()


