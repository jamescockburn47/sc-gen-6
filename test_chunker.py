"""Test script for AgenticChunker."""
import sys
sys.path.insert(0, '.')

from src.schema import ParsedDocument
from src.ingestion.chunkers.agentic_chunker import AgenticChunker

doc = ParsedDocument(
    file_path='test.txt',
    file_name='test_contract.txt',
    document_type='contract',
    text='''ARTICLE 1 - DEFINITIONS
1.1 Confidential Information means proprietary data.
1.2 Deliverables means work product.

ARTICLE 2 - SCOPE OF SERVICES  
2.1 Provider shall perform Services described in SOW.

ARTICLE 3 - TERMINATION
3.1 Either party may terminate with 60 days notice.
''',
    metadata={}
)

print("Creating AgenticChunker...")
chunker = AgenticChunker()
print("Chunking document...")
chunks = chunker.chunk_document(doc)
print(f"Chunks created: {len(chunks)}")
for i, c in enumerate(chunks):
    section = c.metadata.get("section_header", "N/A")
    parent = c.metadata.get("parent_id", "N/A")
    print(f"  [{i}] Section: {section}, parent_id: {parent}, len: {len(c.text)}")
