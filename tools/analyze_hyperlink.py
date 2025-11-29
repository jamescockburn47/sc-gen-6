"""Analyze Hyperlink's database schema to understand how it does local RAG."""

import sqlite3
from pathlib import Path


def main():
    db_path = Path.home() / "AppData/Roaming/Hyperlink/prod/database.sqlite"
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print("=" * 60)
    print("HYPERLINK DATABASE SCHEMA")
    print("=" * 60)
    
    print(f"\nTables ({len(tables)}):")
    for t in tables:
        print(f"  - {t[0]}")
    
    # Get schema and sample data for each table
    print("\n" + "=" * 60)
    print("TABLE DETAILS")
    print("=" * 60)
    
    for (table_name,) in tables:
        print(f"\n### {table_name} ###")
        
        # Schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print("Columns:")
        for col in columns:
            print(f"  {col[1]:30} {col[2]}")
        
        # Row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"Rows: {count}")
        
        # Sample data (first 2 rows)
        if count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
            rows = cursor.fetchall()
            print("Sample:")
            for row in rows:
                # Truncate long values
                truncated = tuple(str(v)[:100] + "..." if len(str(v)) > 100 else v for v in row)
                print(f"  {truncated}")
    
    conn.close()


if __name__ == "__main__":
    main()


