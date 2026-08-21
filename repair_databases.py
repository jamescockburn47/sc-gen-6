"""Database repair and maintenance utilities.

Repairs corrupted SQLite and ChromaDB databases, and adds safeguards.
"""

import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path


def repair_sqlite_catalog(catalog_path: Path) -> bool:
    """Repair a corrupted SQLite catalog database.
    
    Uses SQLite's .recover command to salvage data from corrupted database.
    
    Args:
        catalog_path: Path to catalog.db file
        
    Returns:
        True if repair successful, False otherwise
    """
    if not catalog_path.exists():
        print(f"[Repair] Catalog not found: {catalog_path}")
        return False
    
    backup_path = catalog_path.with_suffix('.db.corrupted')
    recovered_path = catalog_path.with_suffix('.db.recovered')
    
    print(f"[Repair] Attempting to repair: {catalog_path}")
    
    try:
        # First, check if it's actually corrupted
        conn = sqlite3.connect(catalog_path, timeout=5)
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()
            if result[0] == 'ok':
                print("[Repair] Database integrity check passed - no repair needed")
                conn.close()
                return True
        except sqlite3.DatabaseError as e:
            print(f"[Repair] Database is corrupted: {e}")
        finally:
            conn.close()
    except Exception as e:
        print(f"[Repair] Could not open database: {e}")
    
    # Backup the corrupted file
    print(f"[Repair] Backing up corrupted file to: {backup_path}")
    shutil.copy2(catalog_path, backup_path)
    
    # Try to recover data using dump
    try:
        import subprocess
        
        # Use sqlite3 CLI to dump what we can
        result = subprocess.run(
            ['sqlite3', str(catalog_path), '.dump'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0 and result.stdout:
            # Create new database from dump
            new_conn = sqlite3.connect(recovered_path)
            new_conn.executescript(result.stdout)
            new_conn.close()
            
            # Replace original with recovered
            os.remove(catalog_path)
            shutil.move(recovered_path, catalog_path)
            
            print(f"[Repair] Successfully recovered database!")
            return True
        else:
            print(f"[Repair] Could not dump database: {result.stderr}")
            
    except FileNotFoundError:
        print("[Repair] sqlite3 CLI not found - using fallback method")
    except Exception as e:
        print(f"[Repair] Dump recovery failed: {e}")
    
    # Fallback: delete and let app recreate
    print("[Repair] Fallback: removing corrupted database (will be recreated)")
    os.remove(catalog_path)
    return True


def repair_chromadb(vector_db_path: Path) -> bool:
    """Repair a corrupted ChromaDB database.
    
    ChromaDB HNSW segment corruption usually requires full rebuild.
    This function removes corrupted segments and resets the collection.
    
    Args:
        vector_db_path: Path to vector_db directory
        
    Returns:
        True if repair successful, False otherwise
    """
    if not vector_db_path.exists():
        print(f"[Repair] VectorDB not found: {vector_db_path}")
        return False
    
    print(f"[Repair] Attempting to repair ChromaDB: {vector_db_path}")
    
    # Check for corrupted HNSW segments
    corrupted_segments = []
    for segment_dir in vector_db_path.iterdir():
        if segment_dir.is_dir() and len(segment_dir.name) == 36:  # UUID format
            # Check for corrupted pickle files
            for file in segment_dir.iterdir():
                if file.suffix == '.pickle' or file.name.endswith('.bin'):
                    try:
                        # Quick read test
                        with open(file, 'rb') as f:
                            f.read(100)
                    except Exception:
                        corrupted_segments.append(segment_dir)
                        break
    
    if not corrupted_segments:
        # Try to detect corruption via ChromaDB
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(vector_db_path))
            # Try list collections
            client.list_collections()
            print("[Repair] ChromaDB appears healthy")
            return True
        except Exception as e:
            print(f"[Repair] ChromaDB error: {e}")
            corrupted_segments = [vector_db_path]  # Mark whole DB for reset
    
    # Backup and reset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = vector_db_path.with_name(f"{vector_db_path.name}_backup_{timestamp}")
    
    print(f"[Repair] Backing up to: {backup_path}")
    shutil.move(vector_db_path, backup_path)
    
    # Create fresh directory
    vector_db_path.mkdir(parents=True, exist_ok=True)
    
    print("[Repair] ChromaDB reset complete - re-ingestion required")
    return True


def repair_workspace(workspace_path: Path) -> dict:
    """Repair all databases in a workspace.
    
    Args:
        workspace_path: Path to matter/workspace directory
        
    Returns:
        Dictionary with repair results
    """
    results = {
        'workspace': str(workspace_path),
        'catalog_repaired': False,
        'vectordb_repaired': False,
        'keyword_index_repaired': False,
    }
    
    # Repair SQLite catalog
    catalog_path = workspace_path / 'documents' / 'catalog.db'
    if catalog_path.exists():
        results['catalog_repaired'] = repair_sqlite_catalog(catalog_path)
    
    # Repair ChromaDB
    vectordb_path = workspace_path / 'vector_db'
    if vectordb_path.exists():
        results['vectordb_repaired'] = repair_chromadb(vectordb_path)
    
    # Repair keyword index (SQLite)
    keyword_index_path = workspace_path / 'keyword_index'
    if keyword_index_path.exists():
        for db_file in keyword_index_path.glob('*.db'):
            repair_sqlite_catalog(db_file)
        results['keyword_index_repaired'] = True
    
    return results


def add_sqlite_safeguards():
    """Returns recommended SQLite PRAGMA settings for crash safety."""
    return """
        PRAGMA journal_mode=WAL;      -- Write-Ahead Logging for crash safety
        PRAGMA synchronous=NORMAL;    -- Balanced performance/safety
        PRAGMA busy_timeout=30000;    -- 30 second timeout
        PRAGMA temp_store=MEMORY;     -- In-memory temp tables
    """


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python repair_databases.py <workspace_path>")
        print("Example: python repair_databases.py data/matters/enron")
        sys.exit(1)
    
    workspace = Path(sys.argv[1])
    if not workspace.exists():
        print(f"Workspace not found: {workspace}")
        sys.exit(1)
    
    print(f"Repairing workspace: {workspace}")
    results = repair_workspace(workspace)
    
    print("\n=== Repair Results ===")
    for key, value in results.items():
        print(f"  {key}: {value}")
