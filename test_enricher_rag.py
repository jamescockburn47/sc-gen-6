#!/usr/bin/env python3
"""Comprehensive RAG test suite with Kanon 2 Enricher and Enron emails.

Tests:
  1. Kanon 2 Enricher accuracy on existing case docs
  2. Enron email ingestion and enrichment
  3. RAG retrieval quality with enriched vs non-enriched chunks
  4. LLM generation accuracy (GLM-4.7-Flash via Ollama)

Usage:
    python test_enricher_rag.py              # Run all tests
    python test_enricher_rag.py --enricher   # Enricher tests only
    python test_enricher_rag.py --enron      # Enron ingestion + tests
    python test_enricher_rag.py --rag        # RAG accuracy tests
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import requests

# Ensure SCGen6 modules are importable
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

# Load .env
from dotenv import load_dotenv
load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "glm-4.7-flash"
ENRON_CSV = "/mnt/windows/Users/James/Downloads/archive/emails.csv"
ENRON_SAMPLE_SIZE = 500  # Number of emails to ingest
CHROMA_COLLECTION = "enricher_test"
RESULTS_DIR = Path("benchmarks")
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: Ollama generation
# ---------------------------------------------------------------------------

def ollama_generate(prompt: str, num_predict: int = 300, temperature: float = 0.3) -> dict:
    """Generate with Ollama, return response + metrics."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "num_ctx": 32768,
                    "num_predict": num_predict,
                    "temperature": temperature,
                },
            },
            timeout=120,
        )
        data = resp.json()
        if "error" in data:
            return {"error": data["error"], "response": ""}
        eval_ns = data.get("eval_duration", 0)
        eval_tokens = data.get("eval_count", 0)
        return {
            "response": data.get("response", ""),
            "tokens": eval_tokens,
            "time_s": round(eval_ns / 1e9, 2) if eval_ns else 0,
            "tps": round(eval_tokens / (eval_ns / 1e9), 1) if eval_ns > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e), "response": ""}


# ---------------------------------------------------------------------------
# Test 1: Kanon 2 Enricher accuracy
# ---------------------------------------------------------------------------

def test_enricher_accuracy() -> dict:
    """Test Kanon 2 Enricher on known legal texts with ground truth."""
    from src.graph.enricher import KanonEnricher

    print("\n" + "=" * 70)
    print("TEST 1: Kanon 2 Enricher Accuracy")
    print("=" * 70)

    enricher = KanonEnricher()
    if not enricher.is_available:
        print("  SKIP: No ISAACUS_API_KEY configured")
        return {"status": "skipped"}

    test_cases = [
        {
            "name": "UK commercial litigation parties",
            "text": (
                "In the High Court of Justice, Business and Property Courts of England and Wales, "
                "Commercial Court (QBD). HPII Holdings Ltd (In Liquidation) (Claimant) v "
                "Dr Cochrane and Mr Stevens (Defendants). Before Mr Justice Foxton. "
                "Neutral Citation Number: [2021] EWHC 1272 (Comm). "
                "Clifford Chance LLP acted for the Claimant. "
                "The hearing took place at the Rolls Building, Fetter Lane, London EC4A 1NL."
            ),
            "expected_persons": [
                ("HPII Holdings Ltd", "corporate"),
                ("Dr Cochrane", "natural"),
                ("Mr Stevens", "natural"),
            ],
            "expected_locations": ["Rolls Building"],
            "min_persons": 3,
        },
        {
            "name": "Enron-style email with entities",
            "text": (
                "From: jeffrey.skilling@enron.com\n"
                "To: kenneth.lay@enron.com\n"
                "Date: August 14, 2001\n"
                "Subject: Re: Q3 Earnings Call\n\n"
                "Ken, I've reviewed the Raptor III SPE transactions with Andy Fastow. "
                "The LJM2 partnership positions need to be unwound before the SEC filing "
                "deadline. Arthur Andersen has flagged concerns about the mark-to-market "
                "accounting on the California energy trading positions. "
                "We should discuss at the Houston office this Thursday.\n"
                "Jeff"
            ),
            "expected_persons": [
                ("Andy Fastow", "natural"),
            ],
            "expected_locations": [],
            "min_persons": 2,
        },
        {
            "name": "Contract clause with parties and amounts",
            "text": (
                "Pursuant to clause 7.3 of the Share Purchase Agreement dated 15 March 2023 "
                "between Phoenix Capital Partners LLP (the 'Buyer') and Meridian Asset "
                "Management Ltd (the 'Seller'), the Buyer shall pay to the Seller the sum of "
                "£12,500,000 (twelve million five hundred thousand pounds sterling) in "
                "consideration for the transfer of the entire issued share capital of "
                "TargetCo Holdings Ltd. Completion shall take place at the offices of "
                "Freshfields Bruckhaus Deringer LLP, 100 Bishopsgate, London EC2P 2SR."
            ),
            "expected_persons": [
                ("Phoenix Capital Partners LLP", "corporate"),
                ("Meridian Asset Management Ltd", "corporate"),
                ("TargetCo Holdings Ltd", "corporate"),
            ],
            "expected_locations": ["100 Bishopsgate"],
            "min_persons": 3,
        },
    ]

    results = []
    total_score = 0
    total_max = 0

    for i, tc in enumerate(test_cases, 1):
        print(f"\n  [{i}/{len(test_cases)}] {tc['name']}...")
        ilgs_doc = enricher.enrich_text(tc["text"])

        if ilgs_doc is None:
            print(f"    FAIL: Enrichment returned None")
            results.append({"name": tc["name"], "score": 0, "max": 1, "detail": "API failure"})
            continue

        # Score: persons found
        score = 0
        max_score = 0

        # Check minimum person count
        max_score += 1
        found_persons = len(ilgs_doc.persons)
        if found_persons >= tc["min_persons"]:
            score += 1
            print(f"    ✓ Found {found_persons} persons (min {tc['min_persons']})")
        else:
            print(f"    ✗ Found {found_persons} persons (expected ≥{tc['min_persons']})")

        # Check specific expected persons
        for exp_name, exp_type in tc["expected_persons"]:
            max_score += 1
            matched = False
            for p in ilgs_doc.persons:
                if exp_name.lower() in p.name_text.lower():
                    if p.person_type == exp_type:
                        score += 1
                        matched = True
                        print(f"    ✓ Found '{p.name_text}' as {p.person_type} (role: {p.role})")
                    else:
                        score += 0.5  # Partial credit for name match
                        matched = True
                        print(f"    ~ Found '{p.name_text}' but type={p.person_type} (expected {exp_type})")
                    break
            if not matched:
                print(f"    ✗ Missing: '{exp_name}' ({exp_type})")

        # Check locations
        for exp_loc in tc["expected_locations"]:
            max_score += 1
            matched = any(exp_loc.lower() in loc.name_text.lower() for loc in ilgs_doc.locations)
            if matched:
                score += 1
                print(f"    ✓ Found location containing '{exp_loc}'")
            else:
                print(f"    ✗ Missing location: '{exp_loc}'")

        # Summary entities
        print(f"    Entities: {found_persons} persons, {len(ilgs_doc.locations)} locations, "
              f"{len(ilgs_doc.crossreferences)} cross-refs")
        print(f"    Score: {score}/{max_score}")

        total_score += score
        total_max += max_score
        results.append({
            "name": tc["name"],
            "score": score,
            "max": max_score,
            "persons": [{"name": p.name_text, "type": p.person_type, "role": p.role}
                       for p in ilgs_doc.persons],
            "locations": [{"name": loc.name_text, "type": loc.location_type}
                         for loc in ilgs_doc.locations],
        })

    pct = (total_score / total_max * 100) if total_max > 0 else 0
    print(f"\n  ENRICHER ACCURACY: {total_score}/{total_max} ({pct:.0f}%)")
    print(f"  API usage: {enricher.usage_stats}")

    return {
        "status": "complete",
        "total_score": total_score,
        "total_max": total_max,
        "accuracy_pct": round(pct, 1),
        "test_cases": results,
        "usage": enricher.usage_stats,
    }


# ---------------------------------------------------------------------------
# Test 2: Enron email ingestion + enrichment
# ---------------------------------------------------------------------------

def parse_enron_email(raw: str) -> dict[str, str]:
    """Parse a raw Enron email into structured fields."""
    headers: dict[str, str] = {}
    body_lines: list[str] = []
    in_body = False

    for line in raw.split("\n"):
        if in_body:
            body_lines.append(line)
        elif line.strip() == "":
            in_body = True
        else:
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip().lower()
                if key in ("from", "to", "cc", "bcc", "date", "subject", "message-id"):
                    headers[key] = val.strip()

    headers["body"] = "\n".join(body_lines).strip()
    return headers


def load_enron_sample(csv_path: str, n: int = 500) -> list[dict]:
    """Load a diverse sample of Enron emails from the CSV.

    Distributes evenly across key Enron figures for maximum test diversity.
    """
    print(f"\n  Loading Enron emails from {csv_path}...")

    # Key Enron figures whose emails are most studied
    key_senders = [
        "skilling-j", "lay-k", "fastow-a", "delainey-d", "dasovich-j",
        "kitchen-l", "lavorato-j", "shackleton-s", "beck-s", "kaminski-v",
        "bass-e", "allen-p", "campbell-l", "farmer-d", "germany-c",
        "haedicke-m", "jones-t", "lokay-m", "sanders-r", "shapiro-r",
        "steffes-j", "taylor-m", "watson-k", "williams-w3",
    ]

    # Collect per-sender buckets
    per_sender: dict[str, list[dict]] = {s: [] for s in key_senders}
    per_sender_limit = max(n // len(key_senders), 10)
    seen_ids: set[str] = set()
    total_loaded = 0

    try:
        csv.field_size_limit(10 * 1024 * 1024)  # 10MB field limit for large emails
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if total_loaded >= n * 3:  # Read enough to fill buckets
                    break

                filepath = row.get("file", "")
                raw_message = row.get("message", "")
                if not raw_message or len(raw_message) < 100:
                    continue

                # Find which sender this belongs to
                matched_sender = None
                for sender_key in key_senders:
                    if sender_key in filepath.lower():
                        matched_sender = sender_key
                        break
                if not matched_sender:
                    continue

                # Skip if this sender's bucket is full
                if len(per_sender[matched_sender]) >= per_sender_limit:
                    continue

                # Skip duplicates
                msg_hash = hashlib.md5(raw_message[:500].encode()).hexdigest()
                if msg_hash in seen_ids:
                    continue
                seen_ids.add(msg_hash)

                parsed = parse_enron_email(raw_message)
                if len(parsed.get("body", "")) < 50:
                    continue

                parsed["enron_file"] = filepath
                parsed["sender_key"] = matched_sender
                per_sender[matched_sender].append(parsed)
                total_loaded += 1

    except Exception as e:
        print(f"  ERROR loading Enron CSV: {e}")
        return []

    # Merge all buckets, taking proportionally
    emails: list[dict] = []
    for sender_key in key_senders:
        bucket = per_sender[sender_key]
        if bucket:
            emails.extend(bucket[:per_sender_limit])

    emails = emails[:n]  # Cap at requested size

    # Report distribution
    dist: dict[str, int] = {}
    for e in emails:
        sk = e.get("sender_key", "unknown")
        dist[sk] = dist.get(sk, 0) + 1

    print(f"  Loaded {len(emails)} emails from {len([k for k,v in dist.items() if v > 0])} senders:")
    for sk, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {sk}: {count}")

    return emails


def test_enron_ingestion_and_enrichment() -> dict:
    """Ingest Enron emails, enrich with Kanon 2, and test retrieval."""
    from src.graph.enricher import KanonEnricher
    from src.retrieval.embedding_service import EmbeddingService
    from src.retrieval.fts5_index import FTS5Index
    from src.schema import Chunk

    import chromadb

    print("\n" + "=" * 70)
    print("TEST 2: Enron Email Ingestion + Enrichment")
    print("=" * 70)

    if not os.path.exists(ENRON_CSV):
        print(f"  SKIP: Enron CSV not found at {ENRON_CSV}")
        return {"status": "skipped", "reason": "Enron CSV not mounted"}

    # Load emails
    emails = load_enron_sample(ENRON_CSV, n=ENRON_SAMPLE_SIZE)
    if not emails:
        return {"status": "failed", "reason": "No emails loaded"}

    # Initialize services
    enricher = KanonEnricher()
    embedder = EmbeddingService()

    # Create isolated ChromaDB collection
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")
    try:
        chroma_client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare chunks from emails
    chunks: list[Chunk] = []
    enrichment_results: list[dict] = []

    print(f"\n  Chunking {len(emails)} emails...")
    for i, email in enumerate(emails):
        body = email.get("body", "")
        sender = email.get("from", "unknown")
        date_str = email.get("date", "")
        subject = email.get("subject", "")

        # Create chunk text with email context
        chunk_text = f"From: {sender}\nDate: {date_str}\nSubject: {subject}\n\n{body}"

        chunk_id = f"enron_{i:05d}"
        chunk = Chunk(
            chunk_id=chunk_id,
            document_id=f"enron_email_{i}",
            file_name=email.get("enron_file", f"email_{i}.eml"),
            text=chunk_text[:2000],  # Cap at 2000 chars per chunk
            page_number=1,
            paragraph_number=1,
            char_start=0,
            char_end=min(len(chunk_text), 2000),
            document_type="email",
            metadata={
                "sender": sender,
                "recipients": email.get("to", ""),
                "date": date_str,
                "subject": subject,
            },
        )
        chunks.append(chunk)

    # Enrich a sample with Kanon 2 (first 50 to conserve API credits)
    enrich_sample_size = min(50, len(chunks))
    print(f"\n  Enriching {enrich_sample_size} emails with Kanon 2 Enricher...")
    enrich_texts = [c.text for c in chunks[:enrich_sample_size]]

    t0 = time.perf_counter()
    enriched_docs = enricher.enrich_batch(enrich_texts)
    enrich_time = time.perf_counter() - t0

    total_persons = 0
    total_locations = 0
    total_orgs = 0
    for doc in enriched_docs:
        if doc:
            total_persons += sum(1 for p in doc.persons if p.person_type == "natural")
            total_orgs += sum(1 for p in doc.persons if p.person_type == "corporate")
            total_locations += len(doc.locations)

    print(f"  Enrichment: {enrich_time:.1f}s for {enrich_sample_size} emails")
    print(f"  Extracted: {total_persons} persons, {total_orgs} organizations, "
          f"{total_locations} locations")

    # Embed all chunks
    print(f"\n  Embedding {len(chunks)} chunks...")
    t0 = time.perf_counter()
    texts_to_embed = [c.text for c in chunks]
    embeddings = embedder.embed_batch(texts_to_embed)
    embed_time = time.perf_counter() - t0
    print(f"  Embedded in {embed_time:.1f}s ({len(chunks)/embed_time:.0f} chunks/s)")

    # Store in ChromaDB
    print(f"  Storing in ChromaDB collection '{CHROMA_COLLECTION}'...")
    BATCH = 500
    for batch_start in range(0, len(chunks), BATCH):
        batch_end = min(batch_start + BATCH, len(chunks))
        batch_chunks = chunks[batch_start:batch_end]
        batch_embeddings = embeddings[batch_start:batch_end]

        collection.add(
            ids=[c.chunk_id for c in batch_chunks],
            embeddings=[e.tolist() if hasattr(e, 'tolist') else list(e) for e in batch_embeddings],
            documents=[c.text for c in batch_chunks],
            metadatas=[{
                "file_name": c.file_name,
                "document_type": c.document_type,
                "sender": c.metadata.get("sender", ""),
                "subject": c.metadata.get("subject", ""),
            } for c in batch_chunks],
        )

    print(f"  Stored {collection.count()} chunks in ChromaDB")

    return {
        "status": "complete",
        "emails_loaded": len(emails),
        "chunks_created": len(chunks),
        "enrichment": {
            "sample_size": enrich_sample_size,
            "time_s": round(enrich_time, 1),
            "persons_found": total_persons,
            "orgs_found": total_orgs,
            "locations_found": total_locations,
        },
        "embedding": {
            "time_s": round(embed_time, 1),
            "chunks_per_s": round(len(chunks) / embed_time, 0),
        },
        "collection_count": collection.count(),
        "api_usage": enricher.usage_stats,
    }


# ---------------------------------------------------------------------------
# Test 3: RAG accuracy on Enron with verifiable ground truth
# ---------------------------------------------------------------------------

# Ground truth based on published Enron research & known email content
ENRON_GROUND_TRUTH = [
    # --- Queries likely answerable from email corpus ---
    {
        "query": "What energy trading topics were discussed in Enron emails?",
        "required_facts": ["energy", "gas"],
        "forbidden_facts": [],
        "category": "business_activity",
    },
    {
        "query": "What California energy issues were discussed at Enron?",
        "required_facts": ["California"],
        "forbidden_facts": [],
        "category": "business_activity",
    },
    {
        "query": "Who did Enron employees communicate with about natural gas trading?",
        "required_facts": ["gas"],
        "forbidden_facts": [],
        "category": "communication",
    },
    {
        "query": "What FERC regulatory issues were discussed in Enron emails?",
        "required_facts": ["FERC"],
        "forbidden_facts": [],
        "category": "regulatory",
    },
    {
        "query": "What internal meetings or strategy sessions were scheduled at Enron?",
        "required_facts": ["meeting"],
        "forbidden_facts": [],
        "category": "events",
    },
    # --- Harder queries (may require specific emails to be present) ---
    {
        "query": "Who was the CFO of Enron involved in financial fraud?",
        "required_facts": ["Fastow"],
        "forbidden_facts": ["Skilling was CFO", "Lay was CFO"],
        "category": "entity_identification",
    },
    {
        "query": "What was Jeffrey Skilling's role at Enron?",
        "required_facts": ["Skilling"],
        "forbidden_facts": [],
        "category": "entity_identification",
    },
    {
        "query": "Who was Enron's external auditor?",
        "required_facts": ["Arthur Andersen"],
        "forbidden_facts": ["Deloitte", "KPMG", "PwC", "Ernst"],
        "category": "entity_identification",
    },
]


def test_rag_accuracy() -> dict:
    """Test RAG retrieval + generation accuracy on Enron data."""
    from src.retrieval.embedding_service import EmbeddingService

    import chromadb

    print("\n" + "=" * 70)
    print("TEST 3: RAG Accuracy on Enron Data")
    print("=" * 70)

    # Check if Enron collection exists
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")
    try:
        collection = chroma_client.get_collection(CHROMA_COLLECTION)
        count = collection.count()
        if count == 0:
            print("  SKIP: Enron collection empty — run --enron first")
            return {"status": "skipped", "reason": "No Enron data"}
    except Exception:
        print("  SKIP: Enron collection not found — run --enron first")
        return {"status": "skipped", "reason": "Collection not found"}

    print(f"  Using '{CHROMA_COLLECTION}' collection ({count} chunks)")

    embedder = EmbeddingService()
    results: list[dict] = []

    for i, tc in enumerate(ENRON_GROUND_TRUTH, 1):
        print(f"\n  [{i}/{len(ENRON_GROUND_TRUTH)}] {tc['query'][:60]}...")

        # Retrieve
        query_embedding = embedder.embed_query(tc["query"])
        query_emb_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)

        search_results = collection.query(
            query_embeddings=[query_emb_list],
            n_results=5,
        )

        retrieved_docs = search_results.get("documents", [[]])[0]
        if not retrieved_docs:
            print(f"    ✗ No chunks retrieved")
            results.append({
                "query": tc["query"],
                "score": 0,
                "max_score": 1,
                "detail": "No retrieval",
            })
            continue

        # Build context
        context = "\n\n---\n\n".join(retrieved_docs[:5])

        # Generate
        prompt = (
            f"Based ONLY on the following email evidence, answer the question.\n"
            f"If the evidence doesn't contain the answer, say 'insufficient evidence'.\n\n"
            f"EVIDENCE:\n{context}\n\n"
            f"QUESTION: {tc['query']}\n\n"
            f"ANSWER:"
        )

        gen_result = ollama_generate(prompt, num_predict=200)
        answer = gen_result.get("response", "")

        if gen_result.get("error"):
            print(f"    ✗ Generation error: {gen_result['error']}")
            results.append({
                "query": tc["query"],
                "score": 0,
                "max_score": 1,
                "detail": f"Error: {gen_result['error']}",
            })
            continue

        # Score
        score = 0
        max_score = len(tc["required_facts"]) + len(tc["forbidden_facts"])
        if max_score == 0:
            max_score = 1

        answer_lower = answer.lower()
        found_facts = []
        missing_facts = []
        hallucinations = []

        for fact in tc["required_facts"]:
            if fact.lower() in answer_lower:
                score += 1
                found_facts.append(fact)
            else:
                missing_facts.append(fact)

        for forbidden in tc["forbidden_facts"]:
            if re.search(r'\b' + re.escape(forbidden.lower()) + r'\b', answer_lower):
                hallucinations.append(forbidden)
            else:
                score += 1

        pct = (score / max_score * 100) if max_score > 0 else 0
        status = "✓" if pct >= 60 else "~" if pct >= 30 else "✗"
        print(f"    {status} Score: {score}/{max_score} ({pct:.0f}%)")
        if found_facts:
            print(f"      Found: {', '.join(found_facts)}")
        if missing_facts:
            print(f"      Missing: {', '.join(missing_facts)}")
        if hallucinations:
            print(f"      Hallucinations: {', '.join(hallucinations)}")
        print(f"      Answer: {answer[:150]}...")

        results.append({
            "query": tc["query"],
            "category": tc["category"],
            "score": score,
            "max_score": max_score,
            "pct": round(pct, 1),
            "found_facts": found_facts,
            "missing_facts": missing_facts,
            "hallucinations": hallucinations,
            "answer_preview": answer[:300],
            "chunks_retrieved": len(retrieved_docs),
            "gen_tokens": gen_result.get("tokens", 0),
            "gen_time_s": gen_result.get("time_s", 0),
            "gen_tps": gen_result.get("tps", 0),
        })

    # Summary
    total_score = sum(r["score"] for r in results)
    total_max = sum(r["max_score"] for r in results)
    avg_pct = (total_score / total_max * 100) if total_max > 0 else 0

    print(f"\n  RAG ACCURACY: {total_score}/{total_max} ({avg_pct:.0f}%)")
    print(f"  Tests: {len(results)} | "
          f"Passed (≥60%): {sum(1 for r in results if r.get('pct', 0) >= 60)} | "
          f"Partial: {sum(1 for r in results if 30 <= r.get('pct', 0) < 60)} | "
          f"Failed: {sum(1 for r in results if r.get('pct', 0) < 30)}")

    return {
        "status": "complete",
        "total_score": total_score,
        "total_max": total_max,
        "accuracy_pct": round(avg_pct, 1),
        "test_results": results,
    }


# ---------------------------------------------------------------------------
# Test 4: Enricher vs Retrieval comparison
# ---------------------------------------------------------------------------

def test_enriched_retrieval_comparison() -> dict:
    """Compare retrieval quality on enriched vs non-enriched Enron chunks."""
    from src.graph.enricher import KanonEnricher

    import chromadb

    print("\n" + "=" * 70)
    print("TEST 4: Enriched Entity Extraction Summary")
    print("=" * 70)

    chroma_client = chromadb.PersistentClient(path="data/chroma_db")
    try:
        collection = chroma_client.get_collection(CHROMA_COLLECTION)
        if collection.count() == 0:
            print("  SKIP: No Enron data")
            return {"status": "skipped"}
    except Exception:
        print("  SKIP: Collection not found")
        return {"status": "skipped"}

    enricher = KanonEnricher()
    if not enricher.is_available:
        print("  SKIP: No API key")
        return {"status": "skipped"}

    # Get a sample of chunks for entity analysis
    sample = collection.get(limit=20, include=["documents"])
    docs = sample.get("documents", [])

    print(f"  Enriching {len(docs)} sample chunks for entity comparison...")
    t0 = time.perf_counter()
    enriched = enricher.enrich_batch(docs)
    elapsed = time.perf_counter() - t0

    # Aggregate entity stats
    all_persons: dict[str, int] = {}
    all_orgs: dict[str, int] = {}
    all_locations: dict[str, int] = {}

    for doc in enriched:
        if doc is None:
            continue
        for p in doc.persons:
            if p.person_type == "natural":
                all_persons[p.name_text] = all_persons.get(p.name_text, 0) + 1
            else:
                all_orgs[p.name_text] = all_orgs.get(p.name_text, 0) + 1
        for loc in doc.locations:
            all_locations[loc.name_text] = all_locations.get(loc.name_text, 0) + 1

    # Top entities
    top_persons = sorted(all_persons.items(), key=lambda x: -x[1])[:15]
    top_orgs = sorted(all_orgs.items(), key=lambda x: -x[1])[:10]
    top_locations = sorted(all_locations.items(), key=lambda x: -x[1])[:10]

    print(f"\n  Enrichment time: {elapsed:.1f}s for {len(docs)} chunks")
    print(f"\n  Top Persons:")
    for name, count in top_persons:
        print(f"    {name}: {count} mentions")
    print(f"\n  Top Organizations:")
    for name, count in top_orgs:
        print(f"    {name}: {count} mentions")
    print(f"\n  Top Locations:")
    for name, count in top_locations:
        print(f"    {name}: {count} mentions")

    return {
        "status": "complete",
        "chunks_enriched": len(docs),
        "enrichment_time_s": round(elapsed, 1),
        "unique_persons": len(all_persons),
        "unique_orgs": len(all_orgs),
        "unique_locations": len(all_locations),
        "top_persons": top_persons[:10],
        "top_orgs": top_orgs[:10],
        "top_locations": top_locations[:10],
        "api_usage": enricher.usage_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SCGen6 Enricher + RAG Test Suite")
    parser.add_argument("--enricher", action="store_true", help="Run enricher accuracy tests only")
    parser.add_argument("--enron", action="store_true", help="Run Enron ingestion + enrichment")
    parser.add_argument("--rag", action="store_true", help="Run RAG accuracy tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    args = parser.parse_args()

    run_all = args.all or not (args.enricher or args.enron or args.rag)

    print("=" * 70)
    print("SCGen6 — Kanon 2 Enricher + RAG Test Suite")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {OLLAMA_MODEL} | Backend: Ollama + Vulkan")
    print(f"Enricher: Kanon 2 Enricher (Isaacus Beta)")
    print("=" * 70)

    all_results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model": OLLAMA_MODEL,
    }

    # Test 1: Enricher accuracy
    if run_all or args.enricher:
        all_results["enricher_accuracy"] = test_enricher_accuracy()

    # Test 2: Enron ingestion
    if run_all or args.enron:
        all_results["enron_ingestion"] = test_enron_ingestion_and_enrichment()

    # Test 3: RAG accuracy
    if run_all or args.rag:
        all_results["rag_accuracy"] = test_rag_accuracy()

    # Test 4: Entity comparison
    if run_all:
        all_results["entity_analysis"] = test_enriched_retrieval_comparison()

    # Save results
    output_path = RESULTS_DIR / f"enricher_rag_test_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n{'='*70}")
    print(f"Results saved to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
