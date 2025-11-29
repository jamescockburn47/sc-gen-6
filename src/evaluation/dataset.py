"""Synthetic dataset generator for evaluation.

Creates test datasets with queries, documents, ground truth chunks,
and expected citations for LegalBench-RAG style evaluation.
"""

import uuid
from dataclasses import dataclass
from typing import Any

from src.schema import Citation, DocumentType


@dataclass
class EvaluationQuery:
    """A single evaluation query with ground truth.

    Attributes:
        query_id: Unique identifier
        query: Query text
        ground_truth_chunks: List of chunk IDs that should be retrieved
        ground_truth_citations: List of expected citations in answer
        should_refuse: Whether system should refuse to answer
        context: Additional context for the query
    """

    query_id: str
    query: str
    ground_truth_chunks: list[str]
    ground_truth_citations: list[Citation]
    should_refuse: bool
    context: dict[str, Any]


@dataclass
class EvaluationDataset:
    """Evaluation dataset with queries and documents.

    Attributes:
        dataset_id: Unique identifier for dataset
        name: Human-readable name
        queries: List of evaluation queries
        documents: Dictionary mapping document_id to document content
    """

    dataset_id: str
    name: str
    queries: list[EvaluationQuery]
    documents: dict[str, dict[str, Any]]


def create_synthetic_dataset() -> EvaluationDataset:
    """Create a synthetic evaluation dataset.

    Returns:
        EvaluationDataset with test queries and documents
    """
    dataset_id = str(uuid.uuid4())
    queries = []
    documents = {}

    # Document 1: Witness Statement
    doc1_id = "doc_witness_001"
    documents[doc1_id] = {
        "file_name": "witness_statement_john_smith.pdf",
        "document_type": "witness_statement",
        "chunks": [
            {
                "chunk_id": "chunk_001",
                "text": "I, John Smith, testify that on January 15, 2023, I attended a meeting at the company headquarters.",
                "page_number": 1,
                "paragraph_number": 1,
            },
            {
                "chunk_id": "chunk_002",
                "text": "During the meeting, we discussed pricing strategies for the new product line.",
                "page_number": 1,
                "paragraph_number": 2,
            },
            {
                "chunk_id": "chunk_003",
                "text": "The meeting concluded at approximately 3:00 PM.",
                "page_number": 1,
                "paragraph_number": 3,
            },
        ],
    }

    # Document 2: Contract
    doc2_id = "doc_contract_001"
    documents[doc2_id] = {
        "file_name": "supply_agreement_2023.pdf",
        "document_type": "contract",
        "chunks": [
            {
                "chunk_id": "chunk_004",
                "text": "This agreement is entered into on March 1, 2023, between Company A and Company B.",
                "page_number": 1,
                "paragraph_number": 1,
            },
            {
                "chunk_id": "chunk_005",
                "text": "The term of this agreement shall be five years, commencing on the effective date.",
                "page_number": 2,
                "paragraph_number": 5,
            },
            {
                "chunk_id": "chunk_006",
                "text": "Payment terms: Net 30 days from invoice date.",
                "page_number": 3,
                "paragraph_number": 10,
            },
        ],
    }

    # Document 3: Pleading
    doc3_id = "doc_pleading_001"
    documents[doc3_id] = {
        "file_name": "statement_of_claim.pdf",
        "document_type": "pleading",
        "chunks": [
            {
                "chunk_id": "chunk_007",
                "text": "The Plaintiff alleges that the Defendant breached the contract on June 1, 2023.",
                "page_number": 1,
                "paragraph_number": 3,
            },
            {
                "chunk_id": "chunk_008",
                "text": "Damages are sought in the amount of £500,000.",
                "page_number": 2,
                "paragraph_number": 5,
            },
        ],
    }

    # Query 1: Simple retrieval (should find chunk_001)
    queries.append(
        EvaluationQuery(
            query_id="q001",
            query="Who testified about attending a meeting on January 15, 2023?",
            ground_truth_chunks=["chunk_001"],
            ground_truth_citations=[
                Citation(
                    file_name="witness_statement_john_smith.pdf",
                    page_number=1,
                    paragraph_number=1,
                )
            ],
            should_refuse=False,
            context={"expected_answer_contains": "John Smith"},
        )
    )

    # Query 2: Multiple chunks (should find chunk_001 and chunk_002)
    queries.append(
        EvaluationQuery(
            query_id="q002",
            query="What happened at the meeting on January 15, 2023?",
            ground_truth_chunks=["chunk_001", "chunk_002"],
            ground_truth_citations=[
                Citation(
                    file_name="witness_statement_john_smith.pdf",
                    page_number=1,
                    paragraph_number=1,
                ),
                Citation(
                    file_name="witness_statement_john_smith.pdf",
                    page_number=1,
                    paragraph_number=2,
                ),
            ],
            should_refuse=False,
            context={"expected_answer_contains": ["meeting", "pricing strategies"]},
        )
    )

    # Query 3: Contract query (should find chunk_004)
    queries.append(
        EvaluationQuery(
            query_id="q003",
            query="When was the supply agreement entered into?",
            ground_truth_chunks=["chunk_004"],
            ground_truth_citations=[
                Citation(
                    file_name="supply_agreement_2023.pdf",
                    page_number=1,
                    paragraph_number=1,
                )
            ],
            should_refuse=False,
            context={"expected_answer_contains": "March 1, 2023"},
        )
    )

    # Query 4: Should refuse (no relevant information)
    queries.append(
        EvaluationQuery(
            query_id="q004",
            query="What is the capital of France?",
            ground_truth_chunks=[],
            ground_truth_citations=[],
            should_refuse=True,
            context={"expected_answer": "Not found in provided documents"},
        )
    )

    # Query 5: Multiple documents (should find chunks from different docs)
    queries.append(
        EvaluationQuery(
            query_id="q005",
            query="What contracts or agreements are mentioned?",
            ground_truth_chunks=["chunk_004", "chunk_007"],
            ground_truth_citations=[
                Citation(
                    file_name="supply_agreement_2023.pdf",
                    page_number=1,
                    paragraph_number=1,
                ),
                Citation(
                    file_name="statement_of_claim.pdf",
                    page_number=1,
                    paragraph_number=3,
                ),
            ],
            should_refuse=False,
            context={"expected_answer_contains": ["supply agreement", "contract"]},
        )
    )

    return EvaluationDataset(
        dataset_id=dataset_id,
        name="Synthetic Litigation Dataset",
        queries=queries,
        documents=documents,
    )


def load_dataset_from_json(json_path: str) -> EvaluationDataset:
    """Load evaluation dataset from JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        EvaluationDataset

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    import json
    from pathlib import Path

    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {json_path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Parse dataset structure
    dataset_id = data.get("dataset_id", str(uuid.uuid4()))
    name = data.get("name", "Unnamed Dataset")

    # Parse documents
    documents = {}
    for doc_id, doc_data in data.get("documents", {}).items():
        documents[doc_id] = doc_data

    # Parse queries
    queries = []
    for q_data in data.get("queries", []):
        # Parse citations
        citations = []
        for cit_data in q_data.get("ground_truth_citations", []):
            citations.append(
                Citation(
                    file_name=cit_data["file_name"],
                    page_number=cit_data["page_number"],
                    paragraph_number=cit_data.get("paragraph_number"),
                    section_title=cit_data.get("section_title"),
                )
            )

        queries.append(
            EvaluationQuery(
                query_id=q_data["query_id"],
                query=q_data["query"],
                ground_truth_chunks=q_data.get("ground_truth_chunks", []),
                ground_truth_citations=citations,
                should_refuse=q_data.get("should_refuse", False),
                context=q_data.get("context", {}),
            )
        )

    return EvaluationDataset(
        dataset_id=dataset_id, name=name, queries=queries, documents=documents
    )




