"""Clear all document data from Enron workspace databases."""
import sqlite3

print("Clearing Enron workspace data...")

# Clear catalog
catalog = sqlite3.connect(r'data\matters\enron\documents\catalog.db')
catalog.execute('DELETE FROM documents')
catalog.commit()
print('  Catalog: cleared')

# Clear FTS5 index
fts = sqlite3.connect(r'data\matters\enron\keyword_index\fts5_index.db')
fts.execute('DELETE FROM chunks')
fts.commit()
print('  FTS5 index: cleared')

# Clear vector DB if exists
try:
    import shutil
    from pathlib import Path
    vector_path = Path(r'data\matters\enron\vector_db')
    if vector_path.exists():
        shutil.rmtree(vector_path)
        vector_path.mkdir()
        print('  Vector DB: cleared')
except Exception as e:
    print(f'  Vector DB: {e}')

# Clear summaries if exists
try:
    summaries_path = Path(r'data\matters\enron\summaries')
    if summaries_path.exists():
        for f in summaries_path.glob('*.db*'):
            f.unlink()
        print('  Summaries: cleared')
except Exception as e:
    print(f'  Summaries: {e}')

print("\nDone! Enron workspace is now clean.")
