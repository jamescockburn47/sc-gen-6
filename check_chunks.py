"""Quick chunk size diagnostic"""
import sqlite3
import os

for db_path in ['data/vectors.db', 'data/keyword_index/fts5_index.db']:
    print(f"\n{'='*60}")
    print(f"Checking: {db_path}")
    if not os.path.exists(db_path):
        print("  File not found")
        continue
        
    conn = sqlite3.connect(db_path)

# List tables
cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print(f"Tables: {tables}")

# Find the right table
for table in tables:
    try:
        cur = conn.execute(f"SELECT * FROM {table} LIMIT 1")
        cols = [d[0] for d in cur.description]
        if 'text' in cols or 'content' in cols:
            print(f"\nTable '{table}' has text column")
            text_col = 'text' if 'text' in cols else 'content'
            cur = conn.execute(f"SELECT AVG(LENGTH({text_col})), MAX(LENGTH({text_col})), MIN(LENGTH({text_col})), COUNT(*) FROM {table}")
            r = cur.fetchone()
            print(f"  Avg: {r[0]:,.0f} chars, Max: {r[1]:,}, Min: {r[2]:,}, Count: {r[3]:,}")
            
            # Sample some chunks
            cur = conn.execute(f"SELECT LENGTH({text_col}) as len FROM {table} ORDER BY len DESC LIMIT 10")
            print(f"  Top 10 longest: {[r[0] for r in cur.fetchall()]}")
    except Exception as e:
        print(f"Error on {table}: {e}")

conn.close()
