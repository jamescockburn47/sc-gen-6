import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.getcwd())

from src.ingestion.chunkers.adaptive_chunker import AdaptiveChunker
from src.schema import ParsedDocument

def test_deterministic_ids():
    print("Testing deterministic chunk IDs...")
    
    doc = ParsedDocument(
        file_path="test_doc.txt",
        file_name="test_doc.txt",
        document_type="witness_statement",
        text="This is a test document.\n\nIt has multiple paragraphs.\n\nTo test chunking.",
    )
    
    chunker = AdaptiveChunker()
    
    # Run 1
    chunks1 = chunker.chunk_document(doc)
    ids1 = [c.chunk_id for c in chunks1]
    print(f"Run 1 IDs: {ids1}")
    
    # Run 2
    chunks2 = chunker.chunk_document(doc)
    ids2 = [c.chunk_id for c in chunks2]
    print(f"Run 2 IDs: {ids2}")
    
    if ids1 == ids2:
        print("SUCCESS: IDs are deterministic!")
    else:
        print("FAILURE: IDs are NOT deterministic!")
        sys.exit(1)

if __name__ == "__main__":
    test_deterministic_ids()
