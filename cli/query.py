"""CLI utility for quick querying - shows retrieved chunks without generation."""

import argparse
import sys
from pathlib import Path

from src.config_loader import get_settings
from src.retrieval.hybrid_retriever import HybridRetriever


def format_chunk(chunk: dict, index: int) -> str:
    """Format a chunk for display.

    Args:
        chunk: Chunk dictionary
        index: Chunk index (1-based)

    Returns:
        Formatted string
    """
    metadata = chunk.get("metadata", {})
    score = chunk.get("score", 0.0)
    file_name = metadata.get("file_name", "unknown")
    page = metadata.get("page_number", "N/A")
    para = metadata.get("paragraph_number")
    doc_type = metadata.get("document_type", "unknown")

    lines = [
        f"{index}. [{chunk.get('chunk_id', 'unknown')}] Score: {score:.3f}",
        f"   File: {file_name} | Type: {doc_type} | Page: {page}",
    ]

    if para:
        lines[1] += f" | Para: {para}"

    # Add text preview (first 200 chars)
    text = chunk.get("text", "")
    preview = text[:200] + "..." if len(text) > 200 else text
    lines.append(f"   Text: {preview}")
    lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point for query CLI."""
    parser = argparse.ArgumentParser(
        description="Query SC Gen 6 RAG system - shows retrieved chunks"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Query text",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of results per retriever (default: 50)",
    )
    parser.add_argument(
        "--rerank-k",
        type=int,
        default=20,
        help="Number of candidates to rerank (default: 20)",
    )
    parser.add_argument(
        "--context-m",
        type=int,
        default=5,
        help="Number of chunks to return (default: 5)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.70,
        help="Confidence threshold (default: 0.70)",
    )
    parser.add_argument(
        "--doc-type",
        type=str,
        choices=[
            "witness_statement",
            "pleading",
            "statute",
            "contract",
            "disclosure",
            "email",
            "court_filing",
        ],
        help="Filter by document type",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output",
    )

    args = parser.parse_args()

    # Initialize retriever
    print("Initializing retriever...")
    settings = get_settings()
    retriever = HybridRetriever(settings=settings)

    # Retrieve chunks
    print(f"\nQuerying: {args.query}")
    print("Retrieving relevant chunks...\n")

    try:
        chunks = retriever.retrieve(
            query=args.query,
            semantic_top_n=args.top_n,
            keyword_top_n=args.top_n,
            rerank_top_k=args.rerank_k,
            context_to_llm=args.context_m,
            confidence_threshold=args.confidence,
            doc_type_filter=args.doc_type,
        )

        if not chunks:
            print("No relevant chunks found.")
            print("\nSuggestions:")
            print("  - Try rephrasing your query")
            print("  - Lower the confidence threshold (--confidence)")
            print("  - Check if documents have been ingested")
            sys.exit(1)

        # Display results
        print("=" * 60)
        print(f"Retrieved {len(chunks)} chunk(s)")
        print("=" * 60)
        print()

        for i, chunk in enumerate(chunks, 1):
            print(format_chunk(chunk, i))

        # Summary statistics
        if args.verbose:
            scores = [c.get("score", 0.0) for c in chunks]
            avg_score = sum(scores) / len(scores) if scores else 0.0
            print("=" * 60)
            print(f"Average confidence score: {avg_score:.3f}")
            print(f"Highest score: {max(scores):.3f}")
            print(f"Lowest score: {min(scores):.3f}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()




