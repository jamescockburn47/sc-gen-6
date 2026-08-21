"""Enterprise-grade SQLite connection utilities.

Provides a centralized factory for creating SQLite connections with
proper reliability settings to prevent database corruption.
"""

import sqlite3
import time
from pathlib import Path
from typing import Optional, Any, Callable
import functools


# Enterprise-grade PRAGMA settings
PRAGMA_SETTINGS = {
    "journal_mode": "WAL",           # Write-Ahead Logging for crash safety
    "synchronous": "NORMAL",         # Good balance of safety/performance  
    "busy_timeout": 30000,           # 30s wait for locks
    "foreign_keys": "ON",            # Enforce referential integrity
    "journal_size_limit": 67108864,  # 64MB WAL file limit
}


def get_safe_connection(
    db_path: str | Path,
    timeout: float = 30.0,
    row_factory: bool = True,
) -> sqlite3.Connection:
    """Create SQLite connection with enterprise-grade reliability settings.
    
    Args:
        db_path: Path to the database file
        timeout: Connection timeout in seconds (default 30s)
        row_factory: If True, sets row_factory to sqlite3.Row for dict-like access
        
    Returns:
        Configured SQLite connection
    """
    conn = sqlite3.connect(str(db_path), timeout=timeout)
    
    if row_factory:
        conn.row_factory = sqlite3.Row
    
    # Apply enterprise-grade PRAGMA settings
    for pragma, value in PRAGMA_SETTINGS.items():
        conn.execute(f"PRAGMA {pragma}={value}")
    
    return conn


def check_integrity(db_path: str | Path) -> tuple[bool, str]:
    """Run SQLite integrity check on a database.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        Tuple of (is_ok, message)
    """
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        result = conn.execute("PRAGMA integrity_check").fetchone()
        conn.close()
        
        is_ok = result[0] == "ok"
        return is_ok, result[0]
    except sqlite3.DatabaseError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error checking database: {e}"


def safe_execute_with_retry(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple = (),
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> sqlite3.Cursor:
    """Execute SQL with retry on SQLITE_BUSY.
    
    Uses exponential backoff for retries.
    
    Args:
        conn: SQLite connection
        sql: SQL statement
        params: Query parameters
        max_retries: Maximum retry attempts
        base_delay: Initial delay between retries (doubles each retry)
        
    Returns:
        Cursor with results
        
    Raises:
        sqlite3.OperationalError: If all retries fail
    """
    delay = base_delay
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) or "SQLITE_BUSY" in str(e):
                last_error = e
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                continue
            raise
    
    raise sqlite3.OperationalError(
        f"Database locked after {max_retries} retries: {last_error}"
    )


def with_retry(max_retries: int = 3, base_delay: float = 0.1):
    """Decorator for database methods that should retry on SQLITE_BUSY.
    
    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay between retries
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) or "SQLITE_BUSY" in str(e):
                        last_error = e
                        if attempt < max_retries:
                            time.sleep(delay)
                            delay *= 2
                        continue
                    raise
            
            raise sqlite3.OperationalError(
                f"Database locked after {max_retries} retries: {last_error}"
            )
        return wrapper
    return decorator


def repair_database(db_path: str | Path, backup_suffix: str = ".corrupt") -> bool:
    """Attempt to repair a corrupted database.
    
    Uses the .dump and restore method:
    1. Dump what can be recovered
    2. Rename corrupted file
    3. Create fresh database and import dump
    
    Args:
        db_path: Path to the database
        backup_suffix: Suffix for corrupted backup file
        
    Returns:
        True if repair succeeded, False otherwise
    """
    import subprocess
    import shutil
    from datetime import datetime
    
    db_path = Path(db_path)
    
    if not db_path.exists():
        return False
    
    # Check if actually corrupted
    is_ok, msg = check_integrity(db_path)
    if is_ok:
        return True  # Nothing to repair
    
    print(f"[DB REPAIR] Database corrupted: {msg}")
    
    # Create backup with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_suffix(f".{timestamp}{backup_suffix}")
    
    try:
        # Try to dump what we can
        dump_path = db_path.with_suffix(".sql")
        result = subprocess.run(
            ["sqlite3", str(db_path), ".dump"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0 and result.stdout.strip():
            # Move corrupted file
            shutil.move(str(db_path), str(backup_path))
            
            # Remove WAL/SHM files
            for ext in ["-wal", "-shm"]:
                wal_path = Path(str(db_path) + ext)
                if wal_path.exists():
                    wal_path.unlink()
            
            # Create new database from dump
            dump_path.write_text(result.stdout, encoding="utf-8")
            subprocess.run(
                ["sqlite3", str(db_path)],
                input=result.stdout,
                text=True,
                timeout=60
            )
            dump_path.unlink()
            
            # Verify repair
            is_ok, _ = check_integrity(db_path)
            if is_ok:
                print(f"[DB REPAIR] Successfully repaired. Backup: {backup_path}")
                return True
        
        # Dump failed - just backup and let fresh DB be created
        shutil.move(str(db_path), str(backup_path))
        for ext in ["-wal", "-shm"]:
            wal_path = Path(str(db_path) + ext)
            if wal_path.exists():
                wal_path.unlink()
        
        print(f"[DB REPAIR] Could not recover data. Fresh DB will be created. Backup: {backup_path}")
        return True
        
    except Exception as e:
        print(f"[DB REPAIR] Repair failed: {e}")
        return False
