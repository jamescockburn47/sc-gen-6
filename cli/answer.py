"""CLI utility for full answer generation with citations."""

import argparse
import sys
from pathlib import Path

from src.config_loader import get_settings
from src.generation.citation import CitationVerifier, verify_citations
from src.generation.llm_service import LLMService
from src.retrieval.hybrid_retriever import HybridRetriever


def format_citation(citation: dict) -> str:
    """Format a citation for display.

    Args:
        citation: Citation dictionary from parser

    Returns:
        Formatted citation string
    """
    file_name = citation.get("file_name", "unknown")
    page = citation.get("page_number")
    para = citation.get("paragraph_number")
    section = citation.get("section_title")

    parts = [f"Source: {file_name}"]
    if page:
        parts.append(f"Page {page}")
        if para:
            parts.append(f"Para {para}")
    if section:
        parts.append(f'"{section}"')

    return " | ".join(parts)


def main():
    """Main entry point for answer CLI."""
    parser = argparse.ArgumentParser(
        description="Get full answer with citations from SC Gen 6 RAG system"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Query text",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="LLM model to use (default: from config)",
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
        help="Number of chunks to LLM (default: 5)",
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
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature (default: 0.0)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens as they're generated",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output including verification details",
    )

    args = parser.parse_args()

    # Initialize services
    print("Initializing services...")
    settings = get_settings()
    retriever = HybridRetriever(settings=settings)
    llm_service = LLMService(settings=settings)

    # Retrieve chunks
    print(f"\nQuery: {args.query}")
    print("Retrieving relevant chunks...")

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
            print("\nNo relevant chunks found.")
            print("\nSuggestions:")
            print("  - Try rephrasing your query")
            print("  - Lower the confidence threshold (--confidence)")
            print("  - Check if documents have been ingested")
            sys.exit(1)

        # Check if we should refuse
        verifier = CitationVerifier(chunks, args.confidence)
        should_refuse, reason = verifier.should_refuse_before_generation()
        if should_refuse:
            print(f"\nCannot answer: {reason}")
            print(llm_service.get_refusal_message())
            sys.exit(1)

        print(f"Found {len(chunks)} relevant chunk(s)")
        print("Generating answer...\n")

        # Generate answer
        if args.stream:
            print("=" * 60)
            print("Answer:")
            print("=" * 60)
            print()

            def stream_callback(token: str):
                print(token, end="", flush=True)

            response = llm_service.generate_with_context(
                query=args.query,
                chunks=chunks,
                model=args.model,
                temperature=args.temperature,
                stream=True,
                callback=stream_callback,
            )
            print("\n")
        else:
            response = llm_service.generate_with_context(
                query=args.query,
                chunks=chunks,
                model=args.model,
                temperature=args.temperature,
                stream=False,
            )

            print("=" * 60)
            print("Answer:")
            print("=" * 60)
            print()
            print(response)
            print()

        # Verify citations
        verification_result = verify_citations(
            response, chunks, confidence_threshold=args.confidence
        )

        # Display verification results
        print("=" * 60)
        print("Citation Verification:")
        print("=" * 60)
        print(f"Verified: {'✓' if verification_result.verified else '✗'}")
        print(f"All sentences cited: {'✓' if verification_result.all_cited else '✗'}")
        print(f"Citations found: {len(verification_result.citations)}")

        if verification_result.missing:
            print(f"\nMissing citations: {len(verification_result.missing)}")
            if args.verbose:
                for missing in verification_result.missing:
                    print(f"  - {missing}")

        if verification_result.uncited_sentences:
            print(f"\nUncited sentences: {len(verification_result.uncited_sentences)}")
            if args.verbose:
                for sentence in verification_result.uncited_sentences[:5]:
                    print(f"  - {sentence[:100]}...")

        # Show citations
        if verification_result.citations:
            print("\nCitations:")
            for i, citation in enumerate(verification_result.citations, 1):
                # Citation is a Citation object with format() method
                print(f"  {i}. {citation.format()}")

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            with output_path.open("w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("Query:\n")
                f.write("=" * 60 + "\n")
                f.write(args.query + "\n\n")
                f.write("=" * 60 + "\n")
                f.write("Answer:\n")
                f.write("=" * 60 + "\n")
                f.write(response + "\n\n")
                f.write("=" * 60 + "\n")
                f.write("Citations:\n")
                f.write("=" * 60 + "\n")
                for citation in verification_result.citations:
                    f.write(f"- {citation.format()}\n")
            print(f"\nResults saved to: {output_path}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

