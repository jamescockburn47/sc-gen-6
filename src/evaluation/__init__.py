"""Evaluation harness for RAG system metrics and benchmarking."""

from src.evaluation.dataset import (
    EvaluationDataset,
    EvaluationQuery,
    create_synthetic_dataset,
    load_dataset_from_json,
)
from src.evaluation.metrics import (
    EvaluationMetrics,
    EvaluationResult,
    aggregate_metrics,
    compute_citation_accuracy,
    compute_latency_percentiles,
    compute_recall_at_k,
    compute_refusal_accuracy,
    format_metrics_report,
)
from src.evaluation.runner import EvaluationRunner

__all__ = [
    "EvaluationRunner",
    "EvaluationDataset",
    "EvaluationQuery",
    "EvaluationResult",
    "EvaluationMetrics",
    "create_synthetic_dataset",
    "load_dataset_from_json",
    "aggregate_metrics",
    "compute_recall_at_k",
    "compute_citation_accuracy",
    "compute_refusal_accuracy",
    "compute_latency_percentiles",
    "format_metrics_report",
]
