"""Tests for evaluation metrics and runner."""

import pytest

from src.evaluation.dataset import create_synthetic_dataset
from src.evaluation.metrics import (
    EvaluationResult,
    EvaluationMetrics,
    compute_citation_accuracy,
    compute_latency_percentiles,
    compute_recall_at_k,
    compute_refusal_accuracy,
    format_metrics_report,
)
from src.schema import Citation


def test_compute_recall_at_k():
    """Test recall@K computation."""
    ground_truth = ["chunk_1", "chunk_2", "chunk_3"]
    retrieved = ["chunk_1", "chunk_4", "chunk_2", "chunk_5"]

    # Recall@3: should find 2 out of 3 ground truth chunks
    recall = compute_recall_at_k(retrieved, ground_truth, k=3)
    assert recall == pytest.approx(2 / 3)

    # Recall@5: should find all 3 ground truth chunks
    recall = compute_recall_at_k(retrieved, ground_truth, k=5)
    # Retrieved has chunk_1 and chunk_2 (2 items). Ground truth has 3 items. Recall is 2/3.
    # Note: The original test expectation of 1.0 was incorrect because chunk_3 is missing from retrieved.
    assert recall == pytest.approx(2 / 3)

    # Perfect retrieval
    recall = compute_recall_at_k(ground_truth, ground_truth, k=3)
    assert recall == 1.0

    # No ground truth
    recall = compute_recall_at_k(retrieved, [], k=3)
    assert recall == 1.0


def test_compute_citation_accuracy():
    """Test citation accuracy computation."""
    ground_truth = [
        Citation(file_name="doc1.pdf", page_number=1, paragraph_number=1),
        Citation(file_name="doc2.pdf", page_number=2, paragraph_number=3),
    ]

    # Perfect match
    actual = [
        Citation(file_name="doc1.pdf", page_number=1, paragraph_number=1),
        Citation(file_name="doc2.pdf", page_number=2, paragraph_number=3),
    ]
    acc, prec, rec = compute_citation_accuracy(actual, ground_truth)
    assert acc == 1.0
    assert prec == 1.0
    assert rec == 1.0

    # Partial match
    actual = [
        Citation(file_name="doc1.pdf", page_number=1, paragraph_number=1),
        Citation(file_name="doc3.pdf", page_number=5, paragraph_number=1),
    ]
    acc, prec, rec = compute_citation_accuracy(actual, ground_truth)
    assert acc == 0.0
    assert prec == 0.5  # 1 out of 2 correct
    assert rec == 0.5  # 1 out of 2 found

    # No citations
    acc, prec, rec = compute_citation_accuracy([], [])
    assert acc == 1.0
    assert prec == 1.0
    assert rec == 1.0


def test_compute_refusal_accuracy():
    """Test refusal accuracy computation."""
    results = [
        EvaluationResult(
            query_id="q1",
            query="test",
            ground_truth_chunks=[],
            retrieved_chunks=[],
            ground_truth_citations=[],
            actual_citations=[],
            response="",
            verification_result=None,  # type: ignore
            latency_ms=100.0,
            refused=True,
            should_refuse=True,  # Correct refusal
        ),
        EvaluationResult(
            query_id="q2",
            query="test",
            ground_truth_chunks=[],
            retrieved_chunks=[],
            ground_truth_citations=[],
            actual_citations=[],
            response="",
            verification_result=None,  # type: ignore
            latency_ms=100.0,
            refused=False,
            should_refuse=False,  # Correct non-refusal
        ),
        EvaluationResult(
            query_id="q3",
            query="test",
            ground_truth_chunks=[],
            retrieved_chunks=[],
            ground_truth_citations=[],
            actual_citations=[],
            response="",
            verification_result=None,  # type: ignore
            latency_ms=100.0,
            refused=True,
            should_refuse=False,  # Incorrect refusal
        ),
    ]

    accuracy = compute_refusal_accuracy(results)
    assert accuracy == pytest.approx(2 / 3)  # 2 out of 3 correct


def test_compute_latency_percentiles():
    """Test latency percentile computation."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    p50, p95, p99 = compute_latency_percentiles(latencies)

    assert p50 == 60.0  # Median (upper median for even n=10, index 5)
    assert p95 >= 90.0  # 95th percentile
    assert p99 >= 100.0  # 99th percentile

    # Single value
    p50, p95, p99 = compute_latency_percentiles([100.0])
    assert p50 == 100.0
    assert p95 == 100.0
    assert p99 == 100.0


def test_create_synthetic_dataset():
    """Test synthetic dataset creation."""
    dataset = create_synthetic_dataset()

    assert dataset.dataset_id is not None
    assert len(dataset.queries) > 0
    assert len(dataset.documents) > 0

    # Check query structure
    query = dataset.queries[0]
    assert query.query_id is not None
    assert query.query is not None
    assert isinstance(query.ground_truth_chunks, list)
    assert isinstance(query.ground_truth_citations, list)

    # Check document structure
    doc = list(dataset.documents.values())[0]
    assert "file_name" in doc
    assert "chunks" in doc
    assert len(doc["chunks"]) > 0


def test_format_metrics_report():
    """Test metrics report formatting."""
    metrics = EvaluationMetrics(
        recall_at_k=0.85,
        citation_accuracy=0.90,
        citation_precision=0.88,
        citation_recall=0.92,
        refusal_accuracy=0.95,
        latency_p50_ms=150.0,
        latency_p95_ms=300.0,
        latency_p99_ms=500.0,
        num_queries=10,
    )

    report = format_metrics_report(metrics)

    assert "Evaluation Metrics Report" in report
    assert "10" in report  # num_queries
    assert "0.850" in report  # recall_at_k
    assert "150.0" in report  # latency_p50


def test_acceptance_criteria_metrics():
    """Test acceptance criteria: all metrics computed correctly."""
    # Create test results
    results = [
        EvaluationResult(
            query_id="q1",
            query="test query",
            ground_truth_chunks=["chunk_1", "chunk_2"],
            retrieved_chunks=["chunk_1", "chunk_3", "chunk_2"],
            ground_truth_citations=[
                Citation(file_name="doc.pdf", page_number=1, paragraph_number=1)
            ],
            actual_citations=[
                Citation(file_name="doc.pdf", page_number=1, paragraph_number=1)
            ],
            response="Answer with citation",
            verification_result=None,  # type: ignore
            latency_ms=100.0,
            refused=False,
            should_refuse=False,
        )
    ]

    from src.evaluation.metrics import aggregate_metrics

    metrics = aggregate_metrics(results, k=20)

    assert metrics.recall_at_k > 0.0
    assert metrics.citation_accuracy > 0.0
    assert metrics.latency_p50_ms > 0.0
    assert metrics.num_queries == 1

    print("✓ Acceptance criteria PASSED: all metrics computed correctly")




