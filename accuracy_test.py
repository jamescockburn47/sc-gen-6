#!/usr/bin/env python3
"""Accuracy test for the SC Gen 6 RAG system.

Ingests synthetic test documents with known facts, then queries the system
and evaluates whether responses are factually correct, grounded in sources,
and free from hallucination.
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from src.config_loader import get_settings
from src.config.llm_config import load_llm_config

# ── Ground-truth test cases ──────────────────────────────────────────
# Each case has a question, a list of required facts (must appear),
# a list of forbidden facts (must NOT appear — hallucination traps),
# and the expected source document.

TEST_CASES = [
    {
        "id": "T01",
        "question": "What was the total amount misappropriated from client accounts in Phoenix v Meridian?",
        "required_facts": ["47.3 million", "£47.3"],
        "forbidden_facts": ["50 million", "45 million", "100 million"],
        "expected_source": "phoenix_v_meridian",
        "category": "quantum",
    },
    {
        "id": "T02",
        "question": "How many wire transfers were identified by the forensic accountant Gerald Thornton?",
        "required_facts": ["48 "],  # trailing space avoids partial matches like "4800"
        "forbidden_facts": ["52 transfers", "36 transfers"],
        "expected_source": "thornton",
        "category": "factual_detail",
    },
    {
        "id": "T03",
        "question": "What was the name of the yacht purchased with misappropriated funds and what was its purchase price?",
        "required_facts": ["Artemis", "6.2 million"],
        "forbidden_facts": ["Poseidon", "Neptune"],
        "expected_source": "phoenix_v_meridian",
        "category": "asset_tracing",
    },
    {
        "id": "T04",
        "question": "What percentage of Meridian's shares did James Harrington hold?",
        "required_facts": ["72%"],
        "forbidden_facts": ["51%", "80%", "100%", "65%"],
        "expected_source": "phoenix_v_meridian",
        "category": "factual_detail",
    },
    {
        "id": "T05",
        "question": "What was the FCA financial penalty imposed on Meridian and on Harrington personally?",
        "required_facts": ["12.8 million", "3.2 million"],
        "forbidden_facts": ["20 million penalty", "50 million penalty"],
        "expected_source": "phoenix_v_meridian",
        "category": "regulatory",
    },
    {
        "id": "T06",
        "question": "What evidence showed that Catherine Pemberton knew about the misappropriation?",
        "required_facts": ["September 2019", "reconciliation", "email", "gap"],
        "forbidden_facts": ["2020", "2021", "telephone call", "board meeting"],
        "expected_source": "phoenix_v_meridian",
        "category": "liability",
    },
    {
        "id": "T07",
        "question": "Describe the round-tripping mechanism used to conceal the fraud at Meridian",
        "required_facts": ["14.2 million", "month-end", "quarterly", "Northern Trust"],
        "forbidden_facts": ["daily", "weekly", "HMRC"],
        "expected_source": "thornton",
        "category": "mechanism",
    },
    {
        "id": "T08",
        "question": "What was the total judgment amount awarded including interest?",
        "required_facts": ["9.27 million", "£9.27"],
        "forbidden_facts": ["10 million", "15 million", "47.3 million judgment"],
        "expected_source": "phoenix_v_meridian",
        "category": "quantum",
    },
    {
        "id": "T09",
        "question": "Where was the Villa Les Oliviers located and how much was it purchased for?",
        "required_facts": ["Saint-Jean-Cap-Ferrat", "8.4 million"],
        "forbidden_facts": ["Monaco", "Cannes"],
        "expected_source": "phoenix_v_meridian",
        "category": "asset_tracing",
    },
    {
        "id": "T10",
        "question": "What legal tests did the court apply for constructive trust and dishonest assistance?",
        "required_facts": ["Foskett", "Royal Brunei"],
        "forbidden_facts": ["Donoghue v Stevenson", "Caparo", "Anns v Merton"],
        "expected_source": "phoenix_v_meridian",
        "category": "legal_principles",
    },
    {
        "id": "T11",
        "question": "What was Caledonia Ventures Ltd and who owned it?",
        "required_facts": ["British Virgin Islands", "Harrington Family Trust", "Mr Harrington"],
        "forbidden_facts": ["Pemberton", "Meridian owned", "Phoenix owned"],
        "expected_source": "phoenix_v_meridian",
        "category": "entities",
    },
    {
        "id": "T12",
        "question": "What was the estimated realisable value of traced assets according to the expert report?",
        "required_facts": ["26.4 million", "£26.4"],
        "forbidden_facts": ["30 million", "47 million", "20 million"],
        "expected_source": "thornton",
        "category": "quantum",
    },
]


def ingest_test_documents() -> bool:
    """Ingest synthetic test documents into the RAG pipeline."""
    from src.retrieval import get_embedding_service, VectorStore
    from src.retrieval.fts5_index import FTS5Index
    from src.schema import Chunk

    settings = get_settings()
    test_dir = Path("data/test_docs")
    test_files = list(test_dir.glob("TEST_*.txt"))
    
    if not test_files:
        print("ERROR: No test documents found in data/test_docs/")
        return False

    print(f"Found {len(test_files)} test documents")
    
    embed_svc = get_embedding_service(settings=settings)
    vs = VectorStore(settings=settings)
    fts = FTS5Index(settings=settings)
    
    for fpath in test_files:
        print(f"  Ingesting {fpath.name}...")
        
        # Parse as plain text
        text = fpath.read_text(encoding="utf-8")
        
        # Simple chunking: split by paragraph markers (double newline)
        paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip() and len(p.strip()) > 50]
        
        doc_id = f"test_{fpath.stem.lower()}"
        chunks = []
        for i, para in enumerate(paragraphs):
            chunk = Chunk(
                chunk_id=f"{doc_id}_chunk_{i:03d}",
                document_id=doc_id,
                file_name=fpath.name,
                text=para,
                document_type="case_law",
                paragraph_number=max(i, 1),
                char_start=0,
                char_end=len(para),
                metadata={
                    "file_name": fpath.name,
                    "document_id": doc_id,
                    "document_type": "case_law",
                    "paragraph_number": str(max(i, 1)),
                    "char_start": str(0),
                    "char_end": str(len(para)),
                },
            )
            chunks.append(chunk)
        
        print(f"    Created {len(chunks)} chunks")
        
        # Embed
        chunk_texts = [c.text for c in chunks]
        embeddings = embed_svc.embed_batch(chunk_texts)
        
        # Store
        vs.add_chunks(chunks, embeddings)
        fts.add_chunks(chunks)
        print(f"    Stored in vector DB and FTS5")
    
    print(f"Total chunks in vector store: {vs.collection.count()}")
    return True


def evaluate_answer(answer: str, test_case: dict) -> dict:
    """Evaluate a RAG answer against ground truth."""
    answer_lower = answer.lower()
    
    # Check required facts (any ONE variant match counts as found)
    found_facts = []
    missing_facts = []
    for fact in test_case["required_facts"]:
        fact_lower = fact.lower().strip()
        if fact_lower in answer_lower:
            found_facts.append(fact)
        else:
            missing_facts.append(fact)
    
    # Check forbidden facts (hallucination detection — use word boundary matching)
    hallucinations = []
    for forbidden in test_case["forbidden_facts"]:
        forbidden_lower = forbidden.lower().strip()
        # Use word boundary to avoid partial number matches
        pattern = r'\b' + re.escape(forbidden_lower) + r'\b'
        if re.search(pattern, answer_lower):
            hallucinations.append(forbidden)
    
    # Compute scores
    fact_count = len(test_case["required_facts"])
    # At least one required fact variant found counts as match for that fact group
    fact_recall = len(found_facts) / max(fact_count, 1)
    hallucination_free = len(hallucinations) == 0
    
    # Check for "not found" or refusal
    is_refusal = any(phrase in answer_lower for phrase in [
        "not found in provided documents",
        "cannot answer",
        "no information available",
        "i don't have",
        "the documents do not contain",
    ])
    
    # Overall score: fact recall weighted, penalise hallucinations
    if is_refusal:
        accuracy_score = 0.0
    elif hallucinations:
        accuracy_score = max(0, fact_recall - 0.3 * len(hallucinations))
    else:
        accuracy_score = fact_recall
    
    return {
        "test_id": test_case["id"],
        "category": test_case["category"],
        "question": test_case["question"],
        "found_facts": found_facts,
        "missing_facts": missing_facts,
        "hallucinations": hallucinations,
        "fact_recall": round(fact_recall, 2),
        "hallucination_free": hallucination_free,
        "is_refusal": is_refusal,
        "accuracy_score": round(accuracy_score, 2),
        "answer_preview": answer[:300],
    }


def run_accuracy_test(skip_ingest: bool = False) -> dict:
    """Run the full accuracy test suite."""
    settings = get_settings()
    llm_config = load_llm_config()
    
    print("=" * 70)
    print("SC Gen 6 — RAG Accuracy Test Suite")
    print(f"Model: {llm_config.model_name} | Provider: {llm_config.provider}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    print()
    
    # Step 1: Ingest test documents
    if not skip_ingest:
        print("[INGEST] Loading test documents...")
        if not ingest_test_documents():
            print("ERROR: Ingestion failed")
            return {}
        print()
    else:
        print("[INGEST] Skipped (using existing index)")
        print()
    
    # Step 2: Initialise retrieval + generation
    print("[INIT] Loading services...")
    from src.retrieval import get_embedding_service, HybridRetriever, VectorStore, FTS5IndexCompat
    from src.models.reranker import get_reranker_service
    from src.generation.llm_service import LLMService
    
    embed_svc = get_embedding_service(settings=settings)
    vs = VectorStore(settings=settings)
    fts = FTS5IndexCompat(settings=settings)
    reranker = get_reranker_service(settings=settings)
    retriever = HybridRetriever(
        embedding_service=embed_svc, vector_store=vs,
        keyword_index=fts, reranker_service=reranker, settings=settings,
    )
    llm_service = LLMService(settings=settings)
    print(f"[INIT] Vector store: {vs.collection.count()} chunks")
    print()
    
    # Step 3: Run tests
    results = []
    total_start = time.time()
    
    for i, tc in enumerate(TEST_CASES, 1):
        print(f"[{i:02d}/{len(TEST_CASES)}] {tc['id']}: {tc['question'][:65]}...")
        
        query_start = time.time()
        
        # Retrieve
        try:
            chunks = retriever.retrieve(tc["question"])
        except Exception as e:
            print(f"         RETRIEVAL ERROR: {e}")
            results.append({
                "test_id": tc["id"], "error": f"retrieval: {e}",
                "accuracy_score": 0, "fact_recall": 0,
                "hallucination_free": True, "is_refusal": True,
            })
            continue
        
        if not chunks:
            print(f"         No chunks retrieved — marking as refusal")
            results.append({
                "test_id": tc["id"], "error": "no chunks",
                "accuracy_score": 0, "fact_recall": 0,
                "hallucination_free": True, "is_refusal": True,
                "chunks_retrieved": 0,
            })
            continue
        
        # Generate
        try:
            answer = llm_service.generate_with_context(
                query=tc["question"], chunks=chunks, stream=False,
            )
        except Exception as e:
            print(f"         GENERATION ERROR: {e}")
            results.append({
                "test_id": tc["id"], "error": f"generation: {e}",
                "accuracy_score": 0, "fact_recall": 0,
                "hallucination_free": True, "is_refusal": True,
                "chunks_retrieved": len(chunks),
            })
            continue
        
        query_time = time.time() - query_start
        
        # Evaluate
        eval_result = evaluate_answer(answer, tc)
        eval_result["retrieval_time_s"] = round(query_time, 2)
        eval_result["chunks_retrieved"] = len(chunks)
        results.append(eval_result)
        
        # Print result
        score_str = f"{eval_result['accuracy_score']:.0%}"
        facts_str = f"{len(eval_result['found_facts'])}/{len(tc['required_facts'])}"
        hall_str = "CLEAN" if eval_result["hallucination_free"] else f"HALLUCINATED: {eval_result['hallucinations']}"
        print(f"         Score: {score_str} | Facts: {facts_str} | {hall_str} | {query_time:.1f}s")
        if eval_result["missing_facts"]:
            print(f"         Missing: {eval_result['missing_facts']}")
        print()
    
    total_time = time.time() - total_start
    
    # Step 4: Summary
    scores = [r["accuracy_score"] for r in results]
    fact_recalls = [r["fact_recall"] for r in results]
    hallucination_rates = [0 if r.get("hallucination_free", True) else 1 for r in results]
    refusals = sum(1 for r in results if r.get("is_refusal", False))
    
    summary = {
        "total_tests": len(TEST_CASES),
        "avg_accuracy": round(sum(scores) / max(len(scores), 1), 3),
        "avg_fact_recall": round(sum(fact_recalls) / max(len(fact_recalls), 1), 3),
        "hallucination_rate": round(sum(hallucination_rates) / max(len(results), 1), 3),
        "refusal_count": refusals,
        "perfect_scores": sum(1 for s in scores if s >= 1.0),
        "passing_scores": sum(1 for s in scores if s >= 0.5),
        "total_time_s": round(total_time, 1),
    }
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model": llm_config.model_name,
        "provider": llm_config.provider,
        "summary": summary,
        "results": results,
    }
    
    # Save
    out_dir = Path("benchmarks")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"accuracy_test_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print("=" * 70)
    print("ACCURACY TEST RESULTS")
    print("=" * 70)
    print(f"  Tests run:          {summary['total_tests']}")
    print(f"  Avg accuracy:       {summary['avg_accuracy']:.1%}")
    print(f"  Avg fact recall:    {summary['avg_fact_recall']:.1%}")
    print(f"  Hallucination rate: {summary['hallucination_rate']:.1%}")
    print(f"  Perfect scores:     {summary['perfect_scores']}/{summary['total_tests']}")
    print(f"  Passing (>=50%):    {summary['passing_scores']}/{summary['total_tests']}")
    print(f"  Refusals:           {summary['refusal_count']}")
    print(f"  Total time:         {summary['total_time_s']}s")
    print(f"  Saved to:           {out_path}")
    print("=" * 70)
    
    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r["accuracy_score"])
    
    print("\nPer-category breakdown:")
    for cat, cat_scores in sorted(categories.items()):
        avg = sum(cat_scores) / len(cat_scores)
        print(f"  {cat:25s}: {avg:.0%} ({len(cat_scores)} tests)")
    
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SC Gen 6 RAG Accuracy Test")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip document ingestion")
    args = parser.parse_args()
    run_accuracy_test(skip_ingest=args.skip_ingest)
