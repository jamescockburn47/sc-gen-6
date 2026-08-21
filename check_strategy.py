"""Check chunking strategy from Chroma directly (no app dependencies)."""
import chromadb

# Connect to ChromaDB
client = chromadb.PersistentClient(path="data/chroma_db")
collection = client.get_collection("legal_chunks")

# Query a few chunks
results = collection.get(limit=30, include=["metadatas"])

print(f"Checking {len(results['ids'])} chunks for strategy metadata:\n")

strategy_counts = {}
parent_id_count = 0

for i, meta in enumerate(results['metadatas']):
    strategy = meta.get('strategy', 'unknown')
    parent_id = meta.get('parent_id', None)
    section = meta.get('section_header', None)
    
    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    if parent_id:
        parent_id_count += 1
    
    if i < 8:  # Show first 8 details
        print(f"  [{i}] strategy={strategy}, parent_id={parent_id}, section={section}")

print(f"\n--- Summary (sample of {len(results['ids'])} chunks) ---")
print(f"Strategy breakdown:")
for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
    print(f"  {strategy}: {count}")
print(f"\nChunks with parent_id: {parent_id_count}/{len(results['ids'])}")
