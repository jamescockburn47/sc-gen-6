"""Evaluation metrics for RAG system.

Implements LegalBench-RAG style metrics:
- Retrieval recall@K
- Citation accuracy (pinpoint page/para)
- Refusal correctness
- Latency (P95 end-to-end)
"""

import statistics
from dataclasses import dataclass
from typing import Any, Optional

from src.generation.citation import CitationVerificationResult
from src.schema import Citation


@dataclass
class EvaluationResult:
    """Result of a single evaluation query.

    Attributes:
        query_id: Unique identifier for the query
        query: Query text
        ground_truth_chunks: List of chunk IDs that should be retrieved
        retrieved_chunks: List of chunk IDs that were retrieved
        ground_truth_citations: List of Citation objects expected in answer
        actual_citations: List of Citation objects found in answer
        response: Generated response text
        verification_result: Citation verification result
        latency_ms: End-to-end latency in milliseconds
        refused: Whether the system refused to answer
        should_refuse: Whether the system should have refused
    """

    query_id: str
    query: str
    ground_truth_chunks: list[str]
    retrieved_chunks: list[str]
    ground_truth_citations: list[Citation]
    actual_citations: list[Citation]
    response: str
    verification_result: Optional[CitationVerificationResult]
    latency_ms: float
    refused: bool
    should_refuse: bool


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics.

    Attributes:
        recall_at_k: Recall@K score (K=20 by default)
        citation_accuracy: Fraction of citations that match ground truth exactly
        citation_precision: Fraction of citations that are correct
        citation_recall: Fraction of ground truth citations found
        refusal_accuracy: Fraction of correct refusal decisions
        latency_p50_ms: Median latency in milliseconds
        latency_p95_ms: 95th percentile latency in milliseconds
        latency_p99_ms: 99th percentile latency in milliseconds
        num_queries: Total number of queries evaluated
    """

    recall_at_k: float
    citation_accuracy: float
    citation_precision: float
    citation_recall: float
    refusal_accuracy: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    num_queries: int


def compute_recall_at_k(
    retrieved_chunks: list[str], ground_truth_chunks: list[str], k: int = 20
) -> float:
    """Compute recall@K for retrieval.

    Args:
        retrieved_chunks: List of retrieved chunk IDs (ordered by relevance)
        ground_truth_chunks: List of ground truth chunk IDs
        k: Number of top results to consider (default 20)

    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not ground_truth_chunks:
        return 1.0

    # Take top K retrieved chunks
    top_k_retrieved = set(retrieved_chunks[:k])
    ground_truth_set = set(ground_truth_chunks)

    # Count how many ground truth chunks are in top K
    relevant_retrieved = len(top_k_retrieved & ground_truth_set)

    return relevant_retrieved / len(ground_truth_set)


def compute_citation_accuracy(
    actual_citations: list[Citation], ground_truth_citations: list[Citation]
) -> tuple[float, float, float]:
    """Compute citation accuracy metrics.

    Args:
        actual_citations: List of actual citations from response
        ground_truth_citations: List of expected citations

    Returns:
        Tuple of (accuracy, precision, recall)
        - accuracy: Exact match fraction (both lists identical)
        - precision: Fraction of actual citations that match ground truth
        - recall: Fraction of ground truth citations found in actual
    """
    if not ground_truth_citations and not actual_citations:
        return 1.0, 1.0, 1.0

    if not ground_truth_citations:
        return 0.0, 0.0, 1.0  # No citations expected, but some provided

    if not actual_citations:
        return 0.0, 1.0, 0.0  # Citations expected, but none provided

    # Normalize citations for comparison (file_name, page, para)
    def normalize_citation(cit: Citation) -> tuple[str, int, int | None]:
        return (cit.file_name.lower(), cit.page_number, cit.paragraph_number)

    actual_set = {normalize_citation(c) for c in actual_citations}
    ground_truth_set = {normalize_citation(c) for c in ground_truth_citations}

    # Exact match (both lists identical)
    accuracy = 1.0 if actual_set == ground_truth_set else 0.0

    # Precision: fraction of actual citations that are correct
    correct_citations = len(actual_set & ground_truth_set)
    precision = correct_citations / len(actual_set) if actual_set else 0.0

    # Recall: fraction of ground truth citations found
    recall = correct_citations / len(ground_truth_set) if ground_truth_set else 0.0

    return accuracy, precision, recall


def compute_refusal_accuracy(results: list[EvaluationResult]) -> float:
    """Compute refusal accuracy.

    Args:
        results: List of evaluation results

    Returns:
        Fraction of correct refusal decisions (0.0 to 1.0)
    """
    if not results:
        return 0.0

    correct_refusals = sum(
        1 for r in results if r.refused == r.should_refuse
    )
    return correct_refusals / len(results)


def compute_latency_percentiles(latencies_ms: list[float]) -> tuple[float, float, float]:
    """Compute latency percentiles.

    Args:
        latencies_ms: List of latency measurements in milliseconds

    Returns:
        Tuple of (p50, p95, p99) in milliseconds
    """
    if not latencies_ms:
        return 0.0, 0.0, 0.0

    sorted_latencies = sorted(latencies_ms)
    n = len(sorted_latencies)

    p50 = sorted_latencies[int(n * 0.50)]
    p95 = sorted_latencies[int(n * 0.95)] if n > 1 else sorted_latencies[-1]
    p99 = sorted_latencies[int(n * 0.99)] if n > 1 else sorted_latencies[-1]

    return p50, p95, p99


def aggregate_metrics(results: list[EvaluationResult], k: int = 20) -> EvaluationMetrics:
    """Aggregate metrics across all evaluation results.

    Args:
        results: List of evaluation results
        k: K value for recall@K (default 20)

    Returns:
        Aggregated EvaluationMetrics
    """
    if not results:
        return EvaluationMetrics(
            recall_at_k=0.0,
            citation_accuracy=0.0,
            citation_precision=0.0,
            citation_recall=0.0,
            refusal_accuracy=0.0,
            latency_p50_ms=0.0,
            latency_p95_ms=0.0,
            latency_p99_ms=0.0,
            num_queries=0,
        )

    # Compute recall@K for each query
    recall_scores = [
        compute_recall_at_k(r.retrieved_chunks, r.ground_truth_chunks, k)
        for r in results
    ]
    avg_recall = statistics.mean(recall_scores) if recall_scores else 0.0

    # Compute citation metrics
    citation_accuracies = []
    citation_precisions = []
    citation_recalls = []

    for result in results:
        if result.actual_citations or result.ground_truth_citations:
            acc, prec, rec = compute_citation_accuracy(
                result.actual_citations, result.ground_truth_citations
            )
            citation_accuracies.append(acc)
            citation_precisions.append(prec)
            citation_recalls.append(rec)

    avg_citation_accuracy = (
        statistics.mean(citation_accuracies) if citation_accuracies else 0.0
    )
    avg_citation_precision = (
        statistics.mean(citation_precisions) if citation_precisions else 0.0
    )
    avg_citation_recall = (
        statistics.mean(citation_recalls) if citation_recalls else 0.0
    )

    # Refusal accuracy
    refusal_accuracy = compute_refusal_accuracy(results)

    # Latency percentiles
    latencies = [r.latency_ms for r in results]
    p50, p95, p99 = compute_latency_percentiles(latencies)

    return EvaluationMetrics(
        recall_at_k=avg_recall,
        citation_accuracy=avg_citation_accuracy,
        citation_precision=avg_citation_precision,
        citation_recall=avg_citation_recall,
        refusal_accuracy=refusal_accuracy,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        num_queries=len(results),
    )


def format_metrics_report(metrics: EvaluationMetrics) -> str:
    """Format metrics as a human-readable report.

    Args:
        metrics: EvaluationMetrics to format

    Returns:
        Formatted report string
    """
    report_lines = [
        "=" * 60,
        "Evaluation Metrics Report",
        "=" * 60,
        "",
        f"Number of Queries: {metrics.num_queries}",
        "",
        "Retrieval Metrics:",
        f"  Recall@20: {metrics.recall_at_k:.3f}",
        "",
        "Citation Metrics:",
        f"  Accuracy: {metrics.citation_accuracy:.3f}",
        f"  Precision: {metrics.citation_precision:.3f}",
        f"  Recall: {metrics.citation_recall:.3f}",
        "",
        "Refusal Metrics:",
        f"  Accuracy: {metrics.refusal_accuracy:.3f}",
        "",
        "Latency Metrics:",
        f"  P50: {metrics.latency_p50_ms:.1f} ms",
        f"  P95: {metrics.latency_p95_ms:.1f} ms",
        f"  P99: {metrics.latency_p99_ms:.1f} ms",
        "",
        "=" * 60,
    ]

    return "\n".join(report_lines)

