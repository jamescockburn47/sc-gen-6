"""Project XX — Technical Explainer page.

A comprehensive, richly formatted in-app page explaining every system
capability: RAG pipeline, anonymisation, privilege handling, data
sovereignty, and the rationale behind each design decision.

Designed for solicitors, compliance officers, and technical reviewers
who need to understand exactly what this system does and why.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QFont, QDesktopServices
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.ui.styles import COLORS, FONT_FAMILY, FONT_FAMILY_BODY


# ---------------------------------------------------------------------------
# Colour tokens used in the explainer
# ---------------------------------------------------------------------------

_VIOLET = COLORS["primary"]
_VIOLET_LIGHT = COLORS["primary_light"]
_GREEN = COLORS["success"]
_AMBER = COLORS["warning"]
_RED = COLORS["error"]
_BLUE = COLORS["info"]
_BG = COLORS["bg_medium"]
_BG_CARD = COLORS["bg_card"]
_BG_LIGHT = COLORS["bg_light"]
_TEXT = COLORS["text_primary"]
_TEXT_SEC = COLORS["text_secondary"]
_TEXT_MUTED = COLORS["text_muted"]
_BORDER = COLORS["border"]
_BORDER_LIGHT = COLORS["border_light"]


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

def _build_content() -> str:
    """Build the full HTML content for the technical explainer."""

    return f"""
    <style>
        body {{
            font-family: {FONT_FAMILY_BODY};
            color: {_TEXT};
            background: {_BG};
            line-height: 1.7;
            margin: 0;
            padding: 0;
        }}
        .container {{
            max-width: 920px;
            margin: 0 auto;
            padding: 32px 40px 80px 40px;
        }}
        h1 {{
            font-family: {FONT_FAMILY};
            font-size: 32px;
            font-weight: 700;
            color: {_VIOLET_LIGHT};
            margin: 0 0 6px 0;
            letter-spacing: -0.5px;
        }}
        .subtitle {{
            font-size: 14px;
            color: {_TEXT_MUTED};
            margin-bottom: 32px;
        }}
        h2 {{
            font-family: {FONT_FAMILY};
            font-size: 22px;
            font-weight: 600;
            color: {_TEXT};
            margin: 40px 0 16px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid {_BORDER_LIGHT};
        }}
        h3 {{
            font-family: {FONT_FAMILY};
            font-size: 17px;
            font-weight: 600;
            color: {_VIOLET_LIGHT};
            margin: 24px 0 10px 0;
        }}
        p {{
            font-size: 14px;
            color: {_TEXT_SEC};
            margin: 0 0 12px 0;
        }}
        .card {{
            background: {_BG_CARD};
            border: 1px solid {_BORDER};
            border-radius: 8px;
            padding: 18px 20px;
            margin: 12px 0 16px 0;
        }}
        .card-title {{
            font-family: {FONT_FAMILY};
            font-size: 15px;
            font-weight: 600;
            color: {_TEXT};
            margin: 0 0 8px 0;
        }}
        .card p {{
            margin: 0 0 6px 0;
            font-size: 13px;
        }}
        .badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.3px;
            margin-right: 4px;
        }}
        .badge-cutting-edge {{
            background: rgba(139, 124, 246, 0.2);
            color: {_VIOLET_LIGHT};
            border: 1px solid rgba(139, 124, 246, 0.3);
        }}
        .badge-security {{
            background: rgba(74, 222, 128, 0.15);
            color: {_GREEN};
            border: 1px solid rgba(74, 222, 128, 0.25);
        }}
        .badge-compliance {{
            background: rgba(251, 191, 36, 0.15);
            color: {_AMBER};
            border: 1px solid rgba(251, 191, 36, 0.25);
        }}
        .badge-core {{
            background: rgba(96, 165, 250, 0.15);
            color: {_BLUE};
            border: 1px solid rgba(96, 165, 250, 0.25);
        }}
        .rationale {{
            background: rgba(139, 124, 246, 0.06);
            border-left: 3px solid {_VIOLET};
            padding: 10px 16px;
            margin: 8px 0 16px 0;
            border-radius: 0 6px 6px 0;
        }}
        .rationale p {{
            font-size: 13px;
            color: {_TEXT_MUTED};
            margin: 0;
            font-style: italic;
        }}
        .feature-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin: 12px 0;
        }}
        .feature-item {{
            background: {_BG_LIGHT};
            border: 1px solid {_BORDER};
            border-radius: 6px;
            padding: 12px 16px;
            flex: 1 1 280px;
            min-width: 280px;
        }}
        .feature-item .label {{
            font-size: 13px;
            font-weight: 600;
            color: {_TEXT};
            margin-bottom: 4px;
        }}
        .feature-item .desc {{
            font-size: 12px;
            color: {_TEXT_MUTED};
        }}
        code {{
            background: {_BG_LIGHT};
            border: 1px solid {_BORDER};
            padding: 1px 6px;
            border-radius: 3px;
            font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
            font-size: 12px;
            color: {_VIOLET_LIGHT};
        }}
        .example-block {{
            background: {_BG_LIGHT};
            border: 1px solid {_BORDER};
            border-radius: 6px;
            padding: 14px 18px;
            margin: 10px 0 16px 0;
            font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace;
            font-size: 12px;
            line-height: 1.8;
            color: {_TEXT_SEC};
            white-space: pre-wrap;
        }}
        .token-example {{
            color: {_VIOLET_LIGHT};
            font-weight: 600;
        }}
        .preserved {{
            color: {_GREEN};
        }}
        .removed {{
            color: {_RED};
            text-decoration: line-through;
            opacity: 0.6;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0 16px 0;
            font-size: 13px;
        }}
        th {{
            text-align: left;
            padding: 10px 12px;
            background: {_BG_LIGHT};
            color: {_TEXT};
            font-weight: 600;
            border-bottom: 2px solid {_BORDER_LIGHT};
        }}
        td {{
            padding: 8px 12px;
            border-bottom: 1px solid {_BORDER};
            color: {_TEXT_SEC};
        }}
        .divider {{
            border: none;
            border-top: 1px solid {_BORDER};
            margin: 32px 0;
        }}
        a {{
            color: {_VIOLET_LIGHT};
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        ul {{
            padding-left: 20px;
            margin: 8px 0 12px 0;
        }}
        li {{
            font-size: 13px;
            color: {_TEXT_SEC};
            margin-bottom: 6px;
            line-height: 1.6;
        }}
    </style>

    <div class="container">

    <h1>Project XX</h1>
    <p class="subtitle">
        Litigation Support RAG &mdash; Technical Architecture, Security Design &amp; Compliance Rationale
    </p>

    <p>Project XX is a fully local, privacy-first retrieval-augmented generation system
    built for UK civil litigation involving the most sensitive categories of personal
    data. Every component of the RAG pipeline runs on the local machine with zero
    network calls. Cloud LLM features are optional and pass through a mandatory
    anonymisation gateway that renders all exported data non-personal before transmission.</p>

    <hr class="divider">

    <!-- ================================================================== -->
    <!-- SECTION 0: WHY THIS SYSTEM EXISTS                                  -->
    <!-- ================================================================== -->

    <h2>0 &nbsp; Why This System Exists</h2>

    <div class="card">
        <div class="card-title">The Problem</div>
        <p>This system was built for a <strong>church institutional abuse case</strong>
        &mdash; a multi-party claim involving allegations of historical sexual abuse
        of children by clergy members, systemic failures in safeguarding, and
        cover-ups by church authorities over decades. The case materials include:</p>
        <ul>
            <li>Witness statements from survivors describing abuse (special category data
            under UK GDPR Article 9 &mdash; data concerning sex life, health, criminal
            allegations)</li>
            <li>Safeguarding reports identifying victims by name, school, parish, and age
            at the time of abuse</li>
            <li>Solicitor-client privileged advice on liability, quantum, and settlement strategy</li>
            <li>Expert psychiatric and medical reports on the psychological impact of abuse</li>
            <li>Internal church correspondence acknowledging and concealing complaints</li>
            <li>Without-prejudice settlement negotiations referencing specific victims and
            compensation figures</li>
        </ul>
        <p>The volume and complexity of these materials exceeds what can be manually
        reviewed and cross-referenced. A RAG system is essential for effective case
        preparation.</p>
    </div>

    <div class="card">
        <div class="card-title">The Constraint</div>
        <p>The state-of-the-art cloud LLMs &mdash; <strong>Anthropic Claude (Teams)</strong>,
        OpenAI GPT-4o, Google Gemini &mdash; vastly outperform local models for complex
        legal analysis tasks: multi-document synthesis, doctrinal reasoning, quantum
        assessment, and drafting. However:</p>
        <ul>
            <li><strong>US-based servers</strong> &mdash; Anthropic, OpenAI, and Google process
            data on US infrastructure, creating a cross-border personal data transfer
            under UK GDPR Chapter V</li>
            <li><strong>No Zero Data Retention (ZDR) agreement</strong> &mdash; Claude Teams,
            for example, does not currently offer a contractual ZDR commitment. Anthropic's
            standard terms state they may retain inputs for safety monitoring, abuse
            prevention, and model improvement unless opted out. Even with opt-out, there
            is no contractual guarantee equivalent to a Data Processing Agreement with
            ZDR provisions</li>
            <li><strong>Special category data</strong> &mdash; the case materials contain data
            about sexual offences, health conditions, and child protection matters. Under
            UK GDPR Article 9, processing this data requires explicit consent or a legal
            basis, and transferring it to a US provider without adequate safeguards would
            be a serious regulatory breach</li>
            <li><strong>Legal professional privilege</strong> &mdash; solicitor-client advice,
            counsel opinions, and without-prejudice settlement communications could lose
            their privileged status if disclosed to a third-party cloud service</li>
        </ul>
    </div>

    <div class="card">
        <div class="card-title">The Solution</div>
        <p>Project XX resolves this by <strong>keeping all identifiable data local</strong>
        while still enabling cloud LLM analysis of the legal content. The approach:</p>
        <ol style="padding-left: 20px; font-size: 13px; color: {_TEXT_SEC};">
            <li><strong>Process locally</strong> &mdash; all ingestion, retrieval, and base
            generation runs on the local machine using llama.cpp with Vulkan/ROCm
            acceleration. The local RAG pipeline handles day-to-day queries without
            any external connectivity.</li>
            <li><strong>Anonymise before export</strong> &mdash; when cloud analysis is needed
            (complex doctrinal questions, multi-document synthesis, quality review), the
            data passes through a multi-layer anonymisation pipeline that replaces all
            identifying information with contextually enriched tokens. The anonymised
            output is no longer &ldquo;personal data&rdquo; under UK GDPR Article 4(1),
            so cross-border transfer restrictions do not apply.</li>
            <li><strong>Preserve analytical value</strong> &mdash; unlike conventional redaction,
            the tokens carry non-identifying metadata (party roles, amount magnitudes,
            years, vulnerability categories) that allow the cloud LLM to perform useful
            legal analysis. Published case citations are preserved intact.</li>
            <li><strong>Protect privilege</strong> &mdash; privilege markers are anonymised
            (not blocked), so the legal reasoning in advice letters and counsel opinions
            can be analysed without constituting a waiver.</li>
            <li><strong>Reverse locally</strong> &mdash; cloud LLM responses containing tokens
            are de-anonymised on the local machine using an encrypted registry. The
            mapping between tokens and real identities never leaves the system.</li>
        </ol>
        <p>The result: we get the analytical power of Claude/GPT-4o on our case materials
        without any identifiable client data — or any data that could reasonably be
        re-identified — ever reaching US servers.</p>
    </div>

    <div class="rationale">
        <p>This is not a theoretical compliance exercise. If a survivor's identity were
        disclosed through a cloud API &mdash; even transiently, even in logs &mdash; the
        harm would be profound and irreversible. The system is designed on the assumption
        that every piece of data leaving the local machine will be stored, logged, and
        potentially accessed by the provider's employees, and that the anonymised output
        must withstand that worst case. The ICO's &ldquo;motivated intruder&rdquo; test is
        the minimum standard; we design for a &ldquo;determined adversary with full API
        logs&rdquo; standard.</p>
    </div>

    <hr class="divider">

    <!-- ================================================================== -->
    <!-- SECTION 1: RAG PIPELINE                                            -->
    <!-- ================================================================== -->

    <h2>1 &nbsp; Retrieval-Augmented Generation Pipeline</h2>

    <h3>1.1 &nbsp; Document Ingestion</h3>

    <p>Documents are parsed, classified, chunked, embedded, and indexed in a
    single pipeline with checkpoint/resume support for large batches.</p>

    <div class="card">
        <div class="card-title">Parsers</div>
        <table>
            <tr><th>Format</th><th>Engine</th><th>Notes</th></tr>
            <tr><td>PDF</td><td>PyMuPDF + Docling fallback</td>
                <td><span class="badge badge-cutting-edge">CUTTING EDGE</span>
                Docling provides AI-powered layout analysis, table structure recognition,
                and reading order detection for complex legal documents</td></tr>
            <tr><td>DOCX / DOC</td><td>python-docx / antiword</td><td>Paragraph-level metadata, style detection</td></tr>
            <tr><td>Excel / CSV</td><td>OpenPyXL / csv</td><td>Cell-level metadata, formula detection</td></tr>
            <tr><td>Email</td><td>RFC 2822 parser</td><td>Header extraction, MIME parsing, thread detection</td></tr>
            <tr><td>Scanned docs</td><td>Tesseract OCR</td><td>Image preprocessing pipeline</td></tr>
            <tr><td>Plain text / HTML</td><td>Native</td><td>Markdown, email text, HTML stripping</td></tr>
        </table>

        <p>The base parser includes a <strong>legal document classifier</strong> that
        identifies 19 document types (witness statements, pleadings, correspondence,
        expert reports, etc.) using weighted scoring tuned for UK litigation.</p>
    </div>

    <div class="card">
        <div class="card-title">Chunking Strategies</div>

        <p><span class="badge badge-cutting-edge">CUTTING EDGE</span>
        <strong>Agentic Chunker</strong> &mdash; Uses the local LLM to detect document
        structure (headings, clauses, paragraphs) and determine optimal split points.
        Multi-pass strategy with semantic fallback. The LLM understands that a witness
        statement paragraph should not be split mid-sentence, that a contract clause
        should stay together, and that a chronology entry is atomic.</p>

        <p><span class="badge badge-cutting-edge">CUTTING EDGE</span>
        <strong>Semantic Chunker</strong> &mdash; GPU-accelerated LegalBERT ONNX model
        measures sentence-to-sentence semantic similarity and splits where the meaning
        shifts. Structural awareness preserves headings with their content. Contextual
        enrichment prepends document-level context to each chunk.</p>

        <p><span class="badge badge-core">CORE</span>
        <strong>Adaptive Chunker</strong> &mdash; Robust paragraph/sentence-aware
        fixed-size chunking with high overlap. The reliable fallback when LLM-based
        or semantic chunking is unavailable.</p>
    </div>

    <div class="rationale">
        <p>Why three chunking strategies? Legal documents vary enormously. A
        100-page bundle of correspondence needs different chunking to a 3-page
        particulars of claim. The agentic chunker handles structure; the semantic
        chunker handles meaning boundaries; the adaptive chunker guarantees
        completeness. The system selects automatically based on document type
        and available GPU resources.</p>
    </div>

    <h3>1.2 &nbsp; Embedding &amp; Indexing</h3>

    <div class="card">
        <div class="card-title">Embedding Services</div>
        <table>
            <tr><th>Service</th><th>Model</th><th>Acceleration</th></tr>
            <tr><td><span class="badge badge-cutting-edge">CUTTING EDGE</span> ONNX Embeddings</td>
                <td>BGE / mxbai</td><td>ONNX Runtime + DirectML GPU</td></tr>
            <tr><td><span class="badge badge-cutting-edge">CUTTING EDGE</span> LlamaCpp Embeddings</td>
                <td>Nemotron-embed-8b</td><td>llama.cpp server (Vulkan/ROCm)</td></tr>
            <tr><td><span class="badge badge-cutting-edge">CUTTING EDGE</span> Stella Embeddings</td>
                <td>Stella v5 / GTE-large</td><td>FP16, configurable 512-8192 dims</td></tr>
            <tr><td>ROCm Embeddings</td><td>sentence-transformers</td><td>ROCm PyTorch FP16</td></tr>
            <tr><td>CPU Fallback</td><td>sentence-transformers</td><td>CPU</td></tr>
        </table>

        <p>Embeddings are stored in <strong>ChromaDB</strong> (persistent vector store with
        cosine similarity) alongside a <strong>SQLite FTS5</strong> full-text index with
        BM25 ranking for keyword search.</p>
    </div>

    <h3>1.3 &nbsp; Retrieval</h3>

    <div class="card">
        <div class="card-title">Hybrid Retrieval Pipeline</div>

        <p><span class="badge badge-cutting-edge">CUTTING EDGE</span>
        Parallel semantic + keyword search, fused with <strong>Reciprocal Rank Fusion
        (RRF)</strong>, then refined through four stages:</p>

        <ol style="padding-left: 20px; font-size: 13px; color: {_TEXT_SEC};">
            <li><strong>Cross-encoder reranking</strong> &mdash; ONNX GPU-accelerated
            reranker scores each candidate against the query</li>
            <li><strong>MMR diversity filtering</strong> &mdash; Maximal Marginal Relevance
            ensures result diversity (avoids returning five chunks from the same paragraph)</li>
            <li><strong>LLM relevance grading</strong> &mdash; The local LLM scores each
            candidate for actual relevance, filtering out high-similarity but off-topic chunks</li>
            <li><strong>Summary enhancement</strong> &mdash; Adds hierarchical context from
            document/section summaries (RAPTOR-style parent document retrieval)</li>
        </ol>
    </div>

    <div class="card">
        <div class="card-title">Summary-Enhanced Retrieval</div>
        <p><span class="badge badge-cutting-edge">CUTTING EDGE</span>
        A RAPTOR-style hierarchical retrieval system. Documents are summarised at
        three levels (document, section, chunk). Retrieval searches summaries first
        to identify relevant documents, then drills into chunks. Each chunk is
        returned with its parent summary as context, giving the LLM both the
        specific evidence and the broader narrative.</p>
    </div>

    <div class="card">
        <div class="card-title">Query Preprocessing</div>
        <p>Conversational phrases are stripped ("Can you tell me about..."),
        legal entities are detected, and key terms are extracted for
        enhanced retrieval. This bridges the gap between how people naturally
        ask questions and how the retrieval system searches.</p>
    </div>

    <div class="rationale">
        <p>Why this many retrieval stages? Litigation queries are precise. A
        solicitor asking "what did the defendant say about the March payment"
        needs exactly those chunks, not vaguely similar ones. Each stage
        filters differently: RRF handles recall, reranking handles precision,
        MMR handles diversity, LLM grading handles relevance, and summary
        enhancement handles context. The result is consistently high-quality
        retrieval even on large document sets.</p>
    </div>

    <h3>1.4 &nbsp; Generation</h3>

    <div class="card">
        <div class="card-title">LLM Backend</div>
        <p>The system runs entirely on <strong>llama.cpp</strong> with Vulkan/ROCm
        GPU acceleration. Models are loaded via a <strong>llama-swap proxy</strong>
        that enables dynamic model switching without restart &mdash; swap between
        a fast 8B model for queries and a 70B model for complex analysis.</p>

        <p>A <strong>circuit breaker</strong> pattern monitors LLM health: if the
        server fails repeatedly, the breaker trips and auto-recovers, preventing
        cascading failures during long batch operations.</p>
    </div>

    <div class="card">
        <div class="card-title">Generation Capabilities</div>
        <table>
            <tr><th>Feature</th><th>Description</th></tr>
            <tr><td><span class="badge badge-cutting-edge">CUTTING EDGE</span> Parallel Chunk Batching</td>
                <td>Processes retrieved chunks in parallel batches, synthesises
                partial answers, then produces a final cited response</td></tr>
            <tr><td>Multi-Type Summaries</td>
                <td>Overview, key points, entity extraction, and timeline &mdash;
                generated per document in the background</td></tr>
            <tr><td>Case Overview Synthesis</td>
                <td>Builds a structured case narrative from individual document summaries</td></tr>
            <tr><td>Intelligent Document Renaming</td>
                <td>LLM analyses content and proposes descriptive filenames</td></tr>
            <tr><td>Source Citations</td>
                <td>Every generated answer cites specific source documents and chunks</td></tr>
            <tr><td>Streaming Output</td>
                <td>Responses stream token-by-token for responsive UX</td></tr>
        </table>
    </div>

    <h3>1.5 &nbsp; Knowledge Graph</h3>

    <div class="card">
        <div class="card-title">Entity &amp; Relationship Extraction</div>
        <p>Documents are processed to extract entities (persons, organisations,
        locations, dates, monetary amounts) and their relationships. These form
        a knowledge graph that powers the Case Graph visualisation, enabling
        solicitors to see connections between parties, events, and documents
        at a glance.</p>

        <p><strong>Timeline extraction</strong> identifies dated events from both
        document summaries and individual chunks, building a chronological
        narrative of the case.</p>

        <p><span class="badge badge-cutting-edge">CUTTING EDGE</span>
        Optional <strong>Isaacus Kanon 2 Enricher</strong> integration for
        ILGS-schema entity extraction (persons, locations, segments,
        cross-references). Routed through the anonymisation gateway when enabled.</p>
    </div>

    <h3>1.6 &nbsp; Quality Assurance</h3>

    <div class="card">
        <div class="card-title">Guardrails &amp; Assessment</div>
        <ul>
            <li><strong>Input validation</strong> &mdash; jailbreak detection, prompt injection defence</li>
            <li><strong>Retrieval confidence</strong> &mdash; flags low-confidence results for manual review</li>
            <li><strong>Output hallucination detection</strong> &mdash; checks generated answers against source material</li>
            <li><strong>Cloud evaluation</strong> &mdash; optional quality assessment via cloud LLMs
            (anonymised through the gateway)</li>
            <li><strong>Suggestion pipeline</strong> &mdash; parses improvement suggestions and can auto-apply them</li>
            <li><strong>Performance analytics</strong> &mdash; LLM-generated insights on query patterns and system health</li>
        </ul>
    </div>

    <hr class="divider">

    <!-- ================================================================== -->
    <!-- SECTION 2: ANONYMISATION                                           -->
    <!-- ================================================================== -->

    <h2>2 &nbsp; Anonymisation &amp; Pseudonymisation</h2>

    <p>The anonymisation system enables secure export of case materials to cloud
    LLMs for advanced analysis. It is designed for litigation involving the most
    sensitive categories of data &mdash; including sexual abuse, child protection,
    and domestic violence &mdash; where identification of victims could cause
    serious harm.</p>

    <h3>2.1 &nbsp; Multi-Layer PII Detection</h3>

    <div class="card">
        <div class="card-title">Four Detection Layers</div>
        <table>
            <tr><th>Layer</th><th>Engine</th><th>Strength</th></tr>
            <tr><td>1. Microsoft Presidio</td><td>NER + custom pattern recognisers</td>
                <td>Strong on structured identifiers (NI numbers, phone numbers)</td></tr>
            <tr><td>2. spaCy NER</td><td>Transformer-based (with sm/md fallback)</td>
                <td>Strong on named entities (persons, organisations, locations)</td></tr>
            <tr><td>3. Rule-based Patterns</td><td>UK legal, abuse-specific, financial regex</td>
                <td>Catches domain-specific identifiers that statistical models miss</td></tr>
            <tr><td>4. Local LLM</td><td>Contextual analysis (optional)</td>
                <td>Catches indirect identifiers and relationship chains in abuse cases</td></tr>
        </table>

        <p>When multiple layers agree on the same entity, confidence is <strong>boosted
        by +10%</strong>. Entities are merged, de-duplicated, and ranked by a
        <strong>category specificity score</strong> (victim identifiers score 10,
        generic organisations score 2).</p>
    </div>

    <div class="card">
        <div class="card-title">Domain-Specific Pattern Coverage</div>
        <p><strong>UK Legal:</strong> postcodes, NI numbers, phone numbers (mobile/landline/London),
        neutral citations, claim numbers, SRA numbers, bar numbers, vehicle registrations,
        email addresses, dates (all UK formats), age patterns, monetary amounts (GBP),
        IP addresses.</p>

        <p><strong>Abuse-Specific:</strong> relationship descriptors ("the victim's mother"),
        reverse relationships, care institutions (children's homes, secure units, CAMHS),
        school names, social services (CAFCASS, LADO, MASH, Section 47), minor age
        indicators, ABE interview markers, offence descriptors.</p>

        <p><strong>Financial:</strong> sort codes, account numbers (context-gated), IBAN,
        card numbers, SWIFT/BIC codes, Companies House numbers, VAT numbers.</p>
    </div>

    <div class="rationale">
        <p>Why four layers? No single detection method catches everything.
        Presidio is strong on structured patterns but misses contextual
        identifiers. spaCy catches named entities but misses UK-specific
        formats. Rule patterns are precise but rigid. The local LLM
        understands context ("the defendant's daughter" is an indirect
        identifier in an abuse case) but is expensive. The layers complement
        each other, and multi-layer agreement boosts confidence.</p>
    </div>

    <h3>2.2 &nbsp; Contextual Tokenisation</h3>

    <p>This is where Project XX differs from conventional redaction systems.
    Standard redaction produces opaque tokens that destroy analytical value.
    Project XX produces <strong>contextually enriched tokens</strong> that preserve
    the information a cloud LLM needs for legal analysis while removing all
    identifying data.</p>

    <div class="card">
        <div class="card-title">Token Examples</div>
        <div class="example-block">Original:  <span class="removed">David Brown</span> transferred <span class="removed">GBP 450,000</span> on <span class="removed">20th January 2023</span>
Anonymised: <span class="token-example">[PERSON_001:director]</span> transferred <span class="token-example">[AMOUNT_001:six_figures]</span> on <span class="token-example">[DATE_001:2023]</span></div>

        <table>
            <tr><th>Entity Type</th><th>Standard Redaction</th><th>Project XX Token</th><th>Why</th></tr>
            <tr><td>Person</td><td><code>[PERSON_001]</code></td>
                <td><code>[PERSON_001:director]</code></td>
                <td>Cloud LLM knows the party's role for duty/liability analysis</td></tr>
            <tr><td>Amount</td><td><code>[AMOUNT_001]</code></td>
                <td><code>[AMOUNT_001:six_figures]</code></td>
                <td>Order of magnitude preserved for quantum assessment</td></tr>
            <tr><td>Date</td><td><code>[DATE_001]</code></td>
                <td><code>[DATE_001:2023]</code></td>
                <td>Year preserved for limitation period analysis</td></tr>
            <tr><td>Age</td><td><code>[AGE_001]</code></td>
                <td><code>[AGE_001:primary_school_age &mdash; child_vulnerability]</code></td>
                <td>Legal vulnerability category preserved for sentencing/protection analysis</td></tr>
        </table>

        <p><strong>Role detection</strong> identifies 25+ analytical roles from context:
        director, claimant, defendant, witness, solicitor, counsel, judge, expert,
        trustee, beneficiary, shareholder, employer, employee, tenant, landlord,
        victim, perpetrator, insurer, creditor, debtor, insolvency practitioner,
        guardian, child, patient, client.</p>
    </div>

    <div class="card">
        <div class="card-title">Age Vulnerability Bands</div>
        <p>Specific ages are replaced with legally relevant bands that preserve
        the analytical category without identifying the individual:</p>
        <table>
            <tr><th>Age Range</th><th>Band</th><th>Vulnerability Category</th></tr>
            <tr><td>0&ndash;1</td><td>Infant (under 2)</td><td style="color:{_RED}">Young child vulnerability</td></tr>
            <tr><td>2&ndash;4</td><td>Pre-school (2&ndash;4)</td><td style="color:{_RED}">Young child vulnerability</td></tr>
            <tr><td>5&ndash;10</td><td>Primary school age (5&ndash;10)</td><td style="color:{_AMBER}">Child vulnerability</td></tr>
            <tr><td>11&ndash;13</td><td>Early secondary (11&ndash;13)</td><td style="color:{_AMBER}">Child vulnerability</td></tr>
            <tr><td>14&ndash;15</td><td>Mid-teens (14&ndash;15)</td><td style="color:{_AMBER}">Adolescent vulnerability</td></tr>
            <tr><td>16&ndash;17</td><td>Older minor (16&ndash;17)</td><td style="color:{_AMBER}">Adolescent vulnerability</td></tr>
            <tr><td>18&ndash;25</td><td>Young adult (18&ndash;25)</td><td>None</td></tr>
            <tr><td>26&ndash;64</td><td>Adult (26&ndash;64)</td><td>None</td></tr>
            <tr><td>65&ndash;74</td><td>Older adult (65&ndash;74)</td><td style="color:{_AMBER}">Elderly vulnerability</td></tr>
            <tr><td>75&ndash;84</td><td>Elderly (75&ndash;84)</td><td style="color:{_AMBER}">Elderly vulnerability</td></tr>
            <tr><td>85+</td><td>Very elderly (85+)</td><td style="color:{_RED}">Elderly vulnerability</td></tr>
        </table>
    </div>

    <div class="rationale">
        <p>The insight: conventional redaction treats anonymisation and analysis
        as opposites. If you remove enough to be safe, the output is useless.
        Project XX resolves this by recognising that what makes data identifying
        is different from what makes it analytically useful. A cloud LLM
        doesn't need to know the director's name to analyse whether their
        conduct breached s.172 Companies Act 2006 &mdash; it just needs to
        know they were a director. The identifying fact (the name) is removed;
        the analytical fact (the role) is preserved.</p>
    </div>

    <h3>2.3 &nbsp; Citation Guard</h3>

    <div class="card">
        <div class="card-title">Published Case Law Protection</div>
        <p><span class="badge badge-cutting-edge">CUTTING EDGE</span>
        Ported from <strong>CaseKit</strong>'s citation verification system.
        Before PII detection runs, the citation guard scans for published case
        citations and marks them as <strong>protected spans</strong>. Any PII
        entity that overlaps with a reported case is excluded from anonymisation.</p>

        <div class="example-block"><span class="preserved">Re Smith &amp; Fawcett Ltd [1942] Ch 304</span> &mdash; PRESERVED (public law)
<span class="preserved">Armitage v Nurse [1998] Ch 241</span> &mdash; PRESERVED (public law)
<span class="preserved">Sainsbury's Supermarkets Ltd v Mastercard Inc [2020] UKSC 24</span> &mdash; PRESERVED (public law)
<span class="removed">David Brown</span> &rarr; <span class="token-example">[PERSON_001:director]</span> &mdash; ANONYMISED (party in the matter)</div>

        <p><strong>Coverage:</strong> All UK neutral citations (UKSC, UKHL, UKPC, EWCA Civ/Crim,
        EWHC, EWCOP, EWFC, UKUT, UKFTT, UKEAT, Scottish, Northern Irish courts),
        traditional law reports (AC, QB, KB, WLR, All ER, Ch, Fam, ICR, IRLR, FLR,
        BCLC, BCC, Lloyd's Rep, CMLR, and more), and associated case names extracted
        via backward word-walk from &ldquo;v&rdquo; (including R v, In re, Re patterns).</p>
    </div>

    <div class="rationale">
        <p>Without this, anonymisation destroys the legal reasoning. If
        &ldquo;Re Smith &amp; Fawcett Ltd [1942] Ch 304&rdquo; becomes
        &ldquo;[ORGANISATION_002] [[DATE_003]] Ch 304&rdquo;, the cloud LLM
        cannot look up the authority, cannot understand the precedent, and
        the entire export becomes legally useless. Published judgments are
        public domain &mdash; they are the law itself. The citation guard
        ensures the system removes identifying facts about the parties to
        the current matter while preserving the body of law being applied.</p>
    </div>

    <hr class="divider">

    <!-- ================================================================== -->
    <!-- SECTION 3: PRIVILEGE                                               -->
    <!-- ================================================================== -->

    <h2>3 &nbsp; Legal Professional Privilege Protection</h2>

    <p>Sending privileged material to a cloud LLM could constitute a waiver of
    privilege under English law. Project XX handles this by <strong>anonymising
    privilege, not blocking it</strong>.</p>

    <div class="card">
        <div class="card-title">The Principle</div>
        <p>Privilege attaches to the <em>communication</em> between solicitor and
        client about <em>specific facts</em>. It does not attach to abstract legal
        reasoning. Once all identifying information is tokenised &mdash; parties,
        dates, amounts, case references &mdash; the remaining text is an abstract
        legal analysis that could apply to any number of cases. The communication
        can no longer be attributed to a specific client or matter, so disclosure
        to a cloud service does not constitute waiver.</p>
    </div>

    <div class="card">
        <div class="card-title">Privilege Handling Modes</div>
        <table>
            <tr><th>Mode</th><th>Behaviour</th><th>Use Case</th></tr>
            <tr><td><code>anonymise</code> (default)</td>
                <td>Privilege markers (&ldquo;PRIVILEGED AND CONFIDENTIAL&rdquo;,
                &ldquo;without prejudice&rdquo;, etc.) are replaced with category
                tokens. The standard PII pipeline then anonymises all entities.
                Legal reasoning survives.</td>
                <td>General cloud analysis of legal advice</td></tr>
            <tr><td><code>principles_only</code></td>
                <td>Within privileged sections, only sentences containing legal
                reasoning indicators (statutory references, case citations, legal
                tests, doctrinal terms) are preserved. All factual narrative is
                stripped before anonymisation.</td>
                <td>When you only need the cloud LLM to analyse the law being
                applied, not the facts it's being applied to</td></tr>
        </table>
    </div>

    <div class="card">
        <div class="card-title">Detected Privilege Categories</div>
        <ul>
            <li><strong>Solicitor-client privilege</strong> &mdash; &ldquo;privileged and confidential&rdquo;,
            &ldquo;legal advice privilege&rdquo;, &ldquo;subject to legal professional privilege&rdquo;</li>
            <li><strong>Litigation privilege</strong> &mdash; &ldquo;prepared in contemplation of litigation&rdquo;,
            &ldquo;dominant purpose of litigation&rdquo;, draft pleadings, expert reports</li>
            <li><strong>Without prejudice</strong> &mdash; &ldquo;without prejudice&rdquo;,
            Part 36 offers, Calderbank letters, settlement negotiations</li>
            <li><strong>Counsel instructions</strong> &mdash; instructions to/from counsel,
            brief to counsel, conference notes</li>
        </ul>
    </div>

    <div class="rationale">
        <p>The alternative &mdash; blocking privileged content entirely &mdash; would
        make the system useless for its core purpose. Most of the valuable legal
        analysis in a case file <em>is</em> in the privileged material: the advice
        letters, the counsel opinions, the settlement analyses. The insight is
        that once entities are removed, what remains is abstract law. A cloud LLM
        receiving &ldquo;[PERSON_001:director] transferred [AMOUNT_001:six_figures]
        in breach of s.172 Companies Act 2006&rdquo; cannot attribute that to any
        specific matter. The privilege has been preserved by removing the facts
        that make the communication identifiable, while keeping the legal analysis
        that makes it useful.</p>
    </div>

    <hr class="divider">

    <!-- ================================================================== -->
    <!-- SECTION 4: DATA SOVEREIGNTY                                        -->
    <!-- ================================================================== -->

    <h2>4 &nbsp; Data Sovereignty &amp; Cross-Border Compliance</h2>

    <div class="card">
        <div class="card-title">The Claude Teams Problem</div>
        <p>Anthropic Claude (Teams tier) is the preferred cloud LLM for complex legal
        analysis. However, Claude Teams presents specific compliance challenges:</p>
        <ul>
            <li><strong>US-based processing</strong> &mdash; all API requests are processed
            on Anthropic's US infrastructure (GCP us-east-4, AWS us-west-2)</li>
            <li><strong>No Zero Data Retention</strong> &mdash; unlike some enterprise API
            tiers, Claude Teams does not offer a contractual ZDR commitment. Anthropic's
            Usage Policy permits retention of inputs for safety evaluation, trust &amp;
            safety review, and potential abuse investigation</li>
            <li><strong>No DPA with adequacy</strong> &mdash; Anthropic's standard DPA relies
            on Standard Contractual Clauses (SCCs) for cross-border transfer, which the
            ICO considers a weaker safeguard than adequacy decisions. The UK-US Data Bridge
            provides partial adequacy for some US organisations, but the specific terms
            may not cover all processing activities</li>
        </ul>
        <p>The same constraints apply to OpenAI (GPT-4o) and Google (Gemini) &mdash; all
        US-based, none offering ZDR on standard tiers.</p>
    </div>

    <div class="card">
        <div class="card-title">Our Compliance Strategy: Anonymise Below the Threshold</div>
        <p>Rather than relying on contractual safeguards (SCCs, DPAs) or adequacy
        decisions &mdash; all of which are uncertain and require ongoing monitoring &mdash;
        Project XX takes the most robust approach available:</p>
        <p><strong>Ensure the exported data is no longer &ldquo;personal data&rdquo; under
        UK GDPR Article 4(1).</strong></p>
        <p>If the data sent to Claude cannot be used to identify any natural person,
        then it is not personal data. If it is not personal data, then Chapter V
        (cross-border transfers) does not apply, ZDR is irrelevant, and the lack of
        a DPA is moot. The data can be sent to any server, anywhere, because there
        is nothing identifiable in it.</p>
        <p>This is option (c) of the DUAA 2025 three-step test for international
        transfers: <strong>effective anonymisation renders the data non-personal</strong>.</p>
    </div>

    <div class="card">
        <div class="card-title">Sovereignty Checks</div>
        <p>The gateway enforces:</p>
        <ul>
            <li>Provider classification (US-based vs EU/UK-based) with automatic detection</li>
            <li>High-risk entity confidence verification &mdash; if any critical/high-risk
            entities were detected with confidence below 0.8, human review is flagged</li>
            <li>Entity density analysis &mdash; too few detections in a long document may
            indicate the detection layers missed something</li>
            <li>Compliance notes attached to every export payload for audit trail</li>
        </ul>
    </div>

    <div class="card">
        <div class="card-title">ICO 2025 Motivated Intruder Test</div>
        <p>The anonymisation standard is calibrated to the ICO's 2025 anonymisation
        guidance. The test: could a &ldquo;motivated intruder&rdquo; &mdash; someone with
        access to publicly available information and reasonable resources &mdash;
        re-identify any individual from the anonymised output?</p>
        <p>For church abuse cases, this test is particularly stringent because:</p>
        <ul>
            <li>Parish names + dates + ages can narrow identification to a small cohort</li>
            <li>Relationship descriptors (&ldquo;the vicar&rsquo;s daughter&rdquo;) can be
            uniquely identifying in small communities</li>
            <li>Published IICSA reports and media coverage provide the &ldquo;motivated
            intruder&rdquo; with substantial background knowledge</li>
        </ul>
        <p>The multi-layer detection, double-pass validation, abuse-specific pattern
        matching, and mandatory human review for special category data are all
        designed to exceed this test &mdash; even against an adversary with full
        knowledge of the case from public sources.</p>
    </div>

    <div class="rationale">
        <p>We design for the worst case: that every byte sent to the API will be
        stored indefinitely, read by Anthropic employees, subpoenaed by US
        authorities, and cross-referenced against published abuse inquiry reports.
        The anonymised output must withstand all of that without any individual
        being identifiable. If it can survive that scenario, it can survive anything
        Claude Teams' actual data practices might involve.</p>
    </div>

    <hr class="divider">

    <!-- ================================================================== -->
    <!-- SECTION 5: SECURITY ARCHITECTURE                                   -->
    <!-- ================================================================== -->

    <h2>5 &nbsp; Security Architecture</h2>

    <div class="card">
        <div class="card-title">Cloud Export Gateway</div>
        <p><span class="badge badge-security">SECURITY BOUNDARY</span>
        All data leaving the local machine passes through a single gateway.
        There is no other path to external APIs. The gateway enforces:</p>
        <ul>
            <li><strong>Mandatory anonymisation</strong> &mdash; no unanonymised data can be exported</li>
            <li><strong>Double-pass validation</strong> &mdash; PII detection is re-run on the
            anonymised output to catch any entities the first pass missed</li>
            <li><strong>Human review gate</strong> &mdash; documents containing special category
            data (abuse, medical, etc.) require human approval before export</li>
            <li><strong>Privilege anonymisation</strong> &mdash; privilege markers are tokenised</li>
            <li><strong>Data sovereignty checks</strong> &mdash; cross-border compliance verification</li>
            <li><strong>Kanon toggle</strong> &mdash; the Isaacus enricher API (external, not verified
            secure) can be disabled entirely</li>
            <li><strong>Audit logging</strong> &mdash; every export is logged with full provenance</li>
        </ul>
    </div>

    <div class="card">
        <div class="card-title">Encryption &amp; Key Management</div>
        <table>
            <tr><th>Component</th><th>Implementation</th></tr>
            <tr><td>Encryption algorithm</td><td>Fernet (AES-128-CBC + HMAC-SHA256)</td></tr>
            <tr><td>Key derivation</td><td>PBKDF2-HMAC-SHA256, 600,000 iterations (OWASP 2024)</td></tr>
            <tr><td>Key storage</td><td><span class="badge badge-security">MEMORY ONLY</span>
                Derived from passphrase at startup, never written to disk, wiped on close</td></tr>
            <tr><td>Token lookup</td><td>Deterministic HMAC-SHA256 hash (Fernet output is non-deterministic)</td></tr>
            <tr><td>Matter isolation</td><td>Tokens scoped per matter &mdash; no cross-case leakage</td></tr>
        </table>
    </div>

    <div class="card">
        <div class="card-title">Risk-Based Review Framework</div>
        <table>
            <tr><th>Risk Level</th><th style="width:100px">Colour</th><th>Categories</th><th>Review Requirement</th></tr>
            <tr><td><strong>CRITICAL</strong></td>
                <td style="color:{_RED}; font-weight:600;">Red</td>
                <td>Victim identifiers, perpetrator identifiers</td>
                <td>Mandatory human review</td></tr>
            <tr><td><strong>HIGH</strong></td>
                <td style="color:#fb923c; font-weight:600;">Orange</td>
                <td>Addresses, witnesses, DOB, schools, care institutions, relationships, medical</td>
                <td>Flagged for review</td></tr>
            <tr><td><strong>MEDIUM</strong></td>
                <td style="color:{_AMBER}; font-weight:600;">Amber</td>
                <td>Legal professionals, case refs, dates, amounts, postcodes</td>
                <td>Automated</td></tr>
            <tr><td><strong>LOW</strong></td>
                <td style="color:{_BLUE}; font-weight:600;">Blue</td>
                <td>Organisations, URLs, IP addresses</td>
                <td>Automated</td></tr>
        </table>
    </div>

    <div class="card">
        <div class="card-title">Human Review Interface</div>
        <p>A standalone review panel provides:</p>
        <ul>
            <li>Side-by-side comparison: original (local only) vs anonymised (cloud-safe)</li>
            <li>Colour-coded entity highlighting by risk level</li>
            <li>Entity table with category, confidence, detection layer, and risk</li>
            <li>Warning banners for critical risk, validation failures, and low confidence</li>
            <li>Manual operations: add anonymisation, remove false positives, selective de-anonymisation</li>
            <li>Priority queue: documents ranked by critical entity count and validation status</li>
        </ul>
    </div>

    <hr class="divider">

    <!-- ================================================================== -->
    <!-- SECTION 6: REVERSIBILITY                                           -->
    <!-- ================================================================== -->

    <h2>6 &nbsp; Reversible Pseudonymisation</h2>

    <div class="card">
        <div class="card-title">De-anonymisation (Local Only)</div>
        <p>The system is fully reversible. When a cloud LLM returns a response
        containing tokens like <code>[PERSON_001:director]</code>, the de-anonymiser
        resolves these back to the original values using the encrypted registry.
        This happens <strong>exclusively on the local machine</strong> &mdash; the
        mapping between tokens and original values never leaves the system.</p>

        <p>The de-anonymiser handles enriched tokens (strips the <code>:role</code>
        suffix before lookup), preview mode (shows mappings without applying), and
        full audit logging of every de-anonymisation operation.</p>
    </div>

    <div class="rationale">
        <p>Reversibility is essential for the system's purpose. The cloud LLM's
        analysis needs to be understood in context. When it says
        &ldquo;[PERSON_001:director] breached their duty under s.172&rdquo;, the
        solicitor needs to see &ldquo;David Brown breached their duty under
        s.172&rdquo; to action the advice. The token registry is the bridge
        between the anonymous cloud world and the real local world.</p>
    </div>

    <hr class="divider">

    <!-- ================================================================== -->
    <!-- SECTION 7: ARCHITECTURE SUMMARY                                    -->
    <!-- ================================================================== -->

    <h2>7 &nbsp; Architecture Summary</h2>

    <div class="card">
        <div class="card-title">Data Flow</div>
        <div class="example-block">LOCAL MACHINE (all case data stays here)
&boxdr;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxdl;
&boxv; Documents &rarr; Parsers &rarr; Chunkers &rarr; Embeddings &rarr; Vector DB  &boxv;
&boxv;                                                              &boxv;
&boxv; Query &rarr; Preprocessor &rarr; Hybrid Retrieval &rarr; Reranker     &boxv;
&boxv;                         &darr;                                    &boxv;
&boxv;            LLM Generation &rarr; Cited Response                  &boxv;
&boxv;                         &darr;                                    &boxv;
&boxv;              [OPTIONAL CLOUD EXPORT]                           &boxv;
&boxv;                         &darr;                                    &boxv;
&boxv;       PII Detection (4 layers) &rarr; Citation Guard            &boxv;
&boxv;                         &darr;                                    &boxv;
&boxv;       Contextual Tokenisation &rarr; Privilege Handling         &boxv;
&boxv;                         &darr;                                    &boxv;
&boxv;       Double-Pass Validation &rarr; Human Review Gate           &boxv;
&boxv;                         &darr;                                    &boxv;
&boxur;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxh;&boxul;
                         &darr;
         Cloud Export Gateway (security boundary)
                         &darr;
              Anonymised payload only
                         &darr;
              Cloud LLM (OpenAI / Anthropic)
                         &darr;
              Anonymised response
                         &darr;
         De-anonymisation (local only)
                         &darr;
              Readable response with original names</div>
    </div>

    <div class="card">
        <div class="card-title">Design Principles</div>
        <table>
            <tr><th>Principle</th><th>Implementation</th></tr>
            <tr><td>Privacy by default</td><td>No network calls in the RAG pipeline. Cloud features are opt-in.</td></tr>
            <tr><td>Defence in depth</td><td>Four detection layers, double-pass validation, human review, citation guard.</td></tr>
            <tr><td>Analytically useful anonymisation</td><td>Contextual tokens preserve roles, magnitudes, years, vulnerability categories.</td></tr>
            <tr><td>Privilege without blocking</td><td>Anonymise the identifying facts; preserve the legal reasoning.</td></tr>
            <tr><td>Reversibility</td><td>Encrypted registry enables local de-anonymisation of cloud responses.</td></tr>
            <tr><td>Audit trail</td><td>Every detection, anonymisation, export, and de-anonymisation is logged.</td></tr>
            <tr><td>Graceful degradation</td><td>If Presidio segfaults (ROCm), spaCy + rules still provide full coverage.</td></tr>
        </table>
    </div>

    <p style="margin-top: 32px; font-size: 12px; color: {_TEXT_MUTED}; text-align: center;">
        Project XX &mdash; Built for UK litigation. All processing local.
        Cloud export anonymised. Privilege preserved. Citations protected.
    </p>

    </div>
    """


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class TechnicalExplainerPage(QWidget):
    """Full-page technical explanation of Project XX's architecture and security design."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Scroll area wrapping the HTML content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                background: {_BG};
                border: none;
            }}
            QScrollBar:vertical {{
                background: {_BG};
                width: 8px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['border_light']};
                border-radius: 4px;
                min-height: 40px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0;
            }}
        """)

        # Content label using rich HTML
        self._content_label = QLabel()
        self._content_label.setTextFormat(Qt.RichText)
        self._content_label.setWordWrap(True)
        self._content_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._content_label.setOpenExternalLinks(True)
        self._content_label.setStyleSheet(f"""
            QLabel {{
                background: {_BG};
                padding: 0;
                margin: 0;
            }}
        """)
        self._content_label.setText(_build_content())

        scroll.setWidget(self._content_label)
        layout.addWidget(scroll)
