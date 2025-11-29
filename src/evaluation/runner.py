"""Evaluation runner for LegalBench-RAG style evaluation.

Runs evaluation queries, collects metrics, and generates reports.
"""

import time
from pathlib import Path
from typing import Any, Optional

from src.config_loader import Settings, get_settings
from src.evaluation.dataset import EvaluationDataset, EvaluationQuery, create_synthetic_dataset
from src.evaluation.metrics import (
    EvaluationResult,
    EvaluationMetrics,
    aggregate_metrics,
    format_metrics_report,
)
from src.generation.citation import CitationParser, CitationVerifier
from src.generation.llm_service import LLMService
from src.retrieval.hybrid_retriever import HybridRetriever


class EvaluationRunner:
    """Runner for evaluation queries."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize evaluation runner.

        Args:
            settings: Settings instance. If None, loads from config.
        """
        self.settings = settings or get_settings()
        self.retriever: Optional[HybridRetriever] = None
        self.llm_service: Optional[LLMService] = None
        self.citation_parser = CitationParser()

    def initialize_services(self):
        """Initialize retrieval and LLM services."""
        if not self.retriever:
            self.retriever = HybridRetriever(settings=self.settings)
        if not self.llm_service:
            self.llm_service = LLMService(settings=self.settings)

    def evaluate_query(
        self, query: EvaluationQuery, dataset: EvaluationDataset
    ) -> EvaluationResult:
        """Evaluate a single query.

        Args:
            query: Evaluation query with ground truth
            dataset: Dataset containing documents

        Returns:
            EvaluationResult with metrics
        """
        if not self.retriever or not self.llm_service:
            raise RuntimeError("Services not initialized. Call initialize_services() first.")

        start_time = time.time()

        # Retrieve chunks
        chunks = self.retriever.retrieve(
            query=query.query,
            semantic_top_n=self.settings.retrieval.semantic_top_n,
            keyword_top_n=self.settings.retrieval.keyword_top_n,
            rerank_top_k=self.settings.retrieval.rerank_top_k,
            context_to_llm=self.settings.retrieval.context_to_llm,
            confidence_threshold=self.settings.retrieval.confidence_threshold,
        )

        # Check if should refuse
        verifier = CitationVerifier(
            chunks, self.settings.retrieval.confidence_threshold
        )
        should_refuse, _ = verifier.should_refuse_before_generation()

        if should_refuse and query.should_refuse:
            # Correct refusal
            latency_ms = (time.time() - start_time) * 1000
            return EvaluationResult(
                query_id=query.query_id,
                query=query.query,
                ground_truth_chunks=query.ground_truth_chunks,
                retrieved_chunks=[c.get("chunk_id") for c in chunks],
                ground_truth_citations=query.ground_truth_citations,
                actual_citations=[],
                response="",
                verification_result=None,  # type: ignore
                latency_ms=latency_ms,
                refused=True,
                should_refuse=True,
            )

        if should_refuse and not query.should_refuse:
            # Incorrect refusal
            latency_ms = (time.time() - start_time) * 1000
            return EvaluationResult(
                query_id=query.query_id,
                query=query.query,
                ground_truth_chunks=query.ground_truth_chunks,
                retrieved_chunks=[c.get("chunk_id") for c in chunks],
                ground_truth_citations=query.ground_truth_citations,
                actual_citations=[],
                response=self.llm_service.get_refusal_message(),
                verification_result=None,  # type: ignore
                latency_ms=latency_ms,
                refused=True,
                should_refuse=False,
            )

        # Generate response
        response = self.llm_service.generate_with_context(
            query=query.query,
            chunks=chunks,
            stream=False,
        )

        # Parse citations from response
        parsed_citations = self.citation_parser.parse_citations(response)

        # Convert to Citation objects
        actual_citations = []
        for parsed in parsed_citations:
            # Try to match with chunks
            file_name = parsed["file_name"]
            page = parsed["page_number"]
            para = parsed["paragraph_number"]

            # Find matching chunk
            chunk_id = None
            confidence = 1.0
            for chunk in chunks:
                metadata = chunk.get("metadata", {})
                if (
                    metadata.get("file_name") == file_name
                    and metadata.get("page_number") == page
                    and metadata.get("paragraph_number") == para
                ):
                    chunk_id = chunk.get("chunk_id")
                    confidence = chunk.get("score", 1.0)
                    break

            from src.schema import Citation

            actual_citations.append(
                Citation(
                    file_name=file_name,
                    page_number=page or 1,
                    paragraph_number=para,
                    section_title=parsed.get("section_title"),
                    chunk_id=chunk_id,
                    confidence=confidence,
                )
            )

        # Verify citations
        verification_result = CitationVerifier(
            chunks, self.settings.retrieval.confidence_threshold
        ).verify(response)

        latency_ms = (time.time() - start_time) * 1000

        return EvaluationResult(
            query_id=query.query_id,
            query=query.query,
            ground_truth_chunks=query.ground_truth_chunks,
            retrieved_chunks=[c.get("chunk_id") for c in chunks],
            ground_truth_citations=query.ground_truth_citations,
            actual_citations=actual_citations,
            response=response,
            verification_result=verification_result,
            latency_ms=latency_ms,
            refused=False,
            should_refuse=query.should_refuse,
        )

    def evaluate_dataset(
        self, dataset: EvaluationDataset, verbose: bool = False
    ) -> tuple[list[EvaluationResult], EvaluationMetrics]:
        """Evaluate entire dataset.

        Args:
            dataset: Evaluation dataset
            verbose: Whether to print progress

        Returns:
            Tuple of (results list, aggregated metrics)
        """
        if not self.retriever or not self.llm_service:
            self.initialize_services()

        results = []

        for i, query in enumerate(dataset.queries, 1):
            if verbose:
                print(f"Evaluating query {i}/{len(dataset.queries)}: {query.query_id}")

            try:
                result = self.evaluate_query(query, dataset)
                results.append(result)

                if verbose:
                    print(f"  Latency: {result.latency_ms:.1f} ms")
                    print(f"  Retrieved: {len(result.retrieved_chunks)} chunks")
                    print(f"  Citations: {len(result.actual_citations)}")

            except Exception as e:
                if verbose:
                    print(f"  Error: {str(e)}")
                # Create error result
                results.append(
                    EvaluationResult(
                        query_id=query.query_id,
                        query=query.query,
                        ground_truth_chunks=query.ground_truth_chunks,
                        retrieved_chunks=[],
                        ground_truth_citations=query.ground_truth_citations,
                        actual_citations=[],
                        response=f"Error: {str(e)}",
                        verification_result=None,  # type: ignore
                        latency_ms=0.0,
                        refused=False,
                        should_refuse=query.should_refuse,
                    )
                )

        # Aggregate metrics
        metrics = aggregate_metrics(results, k=20)

        return results, metrics

    def save_results(
        self,
        results: list[EvaluationResult],
        metrics: EvaluationMetrics,
        output_path: Path,
    ):
        """Save evaluation results to file.

        Args:
            results: List of evaluation results
            metrics: Aggregated metrics
            output_path: Path to save results
        """
        import json

        output_data = {
            "metrics": {
                "recall_at_k": metrics.recall_at_k,
                "citation_accuracy": metrics.citation_accuracy,
                "citation_precision": metrics.citation_precision,
                "citation_recall": metrics.citation_recall,
                "refusal_accuracy": metrics.refusal_accuracy,
                "latency_p50_ms": metrics.latency_p50_ms,
                "latency_p95_ms": metrics.latency_p95_ms,
                "latency_p99_ms": metrics.latency_p99_ms,
                "num_queries": metrics.num_queries,
            },
            "results": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "ground_truth_chunks": r.ground_truth_chunks,
                    "retrieved_chunks": r.retrieved_chunks,
                    "ground_truth_citations": [
                        {
                            "file_name": c.file_name,
                            "page_number": c.page_number,
                            "paragraph_number": c.paragraph_number,
                        }
                        for c in r.ground_truth_citations
                    ],
                    "actual_citations": [
                        {
                            "file_name": c.file_name,
                            "page_number": c.page_number,
                            "paragraph_number": c.paragraph_number,
                        }
                        for c in r.actual_citations
                    ],
                    "latency_ms": r.latency_ms,
                    "refused": r.refused,
                    "should_refuse": r.should_refuse,
                }
                for r in results
            ],
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)


def main():
    """Main entry point for evaluation runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation on RAG system")
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        help="Dataset to use ('synthetic' or path to JSON file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file path for results",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print verbose progress"
    )

    args = parser.parse_args()

    # Load dataset
    if args.dataset == "synthetic":
        dataset = create_synthetic_dataset()
    else:
        from src.evaluation.dataset import load_dataset_from_json

        dataset = load_dataset_from_json(args.dataset)

    # Run evaluation
    runner = EvaluationRunner()
    runner.initialize_services()

    print(f"Evaluating dataset: {dataset.name}")
    print(f"Number of queries: {len(dataset.queries)}")
    print()

    results, metrics = runner.evaluate_dataset(dataset, verbose=args.verbose)

    # Print metrics
    print(format_metrics_report(metrics))

    # Save results
    output_path = Path(args.output)
    runner.save_results(results, metrics, output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()




