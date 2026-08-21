# Case Graph & Multi-Matter Architecture

## Overview

A **Case Graph** is a structured knowledge graph extracted from documents that:
1. Identifies **entities** (people, companies, dates, amounts, locations)
2. Maps **relationships** (who communicated with whom, what events occurred)
3. Builds a **timeline** of events
4. Links back to **source chunks** for citation

The graph enhances search and generation without inflating context by providing:
- **Query expansion**: "John Smith" → also finds "Mr Smith", "JS", "the Claimant"
- **Entity filtering**: Find all chunks mentioning a specific person
- **Timeline view**: Navigate events chronologically
- **Relationship paths**: Discover connections between entities

---

## Architecture

### Data Model

```
Matter/
├── config.json              # Matter metadata (name, parties, date range)
├── documents/               # Source documents
├── chroma_db/               # Vector embeddings
├── bm25_index/              # Keyword index
├── case_graph/
│   ├── entities.json        # Entity definitions with aliases
│   ├── relationships.json   # Entity-to-entity relationships
│   ├── timeline.json        # Events with dates
│   ├── chunk_links.json     # Entity → chunk_id mappings
│   └── user_edits.json      # Manual corrections (overlay)
└── exports/                 # Generated reports
```

### Entity Types

| Type | Examples | Auto-Extracted From |
|------|----------|---------------------|
| `person` | John Smith, Ms Jones | Names, witness statements, email headers |
| `organization` | Acme Ltd, FCA | Company names, regulatory bodies |
| `date` | 15 January 2024 | Date patterns in text |
| `amount` | £1,500,000 | Currency patterns |
| `location` | London, 10 Downing Street | Address patterns |
| `case_ref` | [2024] EWHC 123 | Case citation patterns |
| `document` | Exhibit A, Appendix 1 | Document references |
| `event` | "the Meeting", "the Transaction" | Key event phrases |

### Entity Schema

```python
@dataclass
class Entity:
    id: str                      # UUID
    type: str                    # person, organization, date, etc.
    canonical_name: str          # Primary display name
    aliases: list[str]           # Alternative names/spellings
    metadata: dict               # Type-specific data (e.g., role for person)
    source_chunks: list[str]     # Chunk IDs where entity appears
    confidence: float            # Extraction confidence (0-1)
    user_verified: bool          # Has user confirmed this entity?
    created_at: datetime
    updated_at: datetime
```

### Relationship Schema

```python
@dataclass
class Relationship:
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str       # sent_to, employed_by, attended, etc.
    properties: dict             # Date, context, etc.
    source_chunks: list[str]     # Evidence chunks
    confidence: float
    user_verified: bool
```

### Timeline Event Schema

```python
@dataclass
class TimelineEvent:
    id: str
    date: date | None            # Exact date if known
    date_range: tuple[date, date] | None  # Range if approximate
    date_text: str               # Original text ("mid-January 2024")
    description: str             # Event description
    entities_involved: list[str] # Entity IDs
    source_chunks: list[str]     # Evidence chunks
    event_type: str              # meeting, communication, transaction, filing
    user_verified: bool
```

---

## Extraction Pipeline

### Phase 1: Entity Extraction (During Ingestion)

```
Document → Parse → Chunks → For each chunk:
    1. Named Entity Recognition (NER)
       - Use spaCy or local LLM for extraction
       - Extract: PERSON, ORG, DATE, MONEY, GPE, etc.
    
    2. Pattern Matching
       - Case citations: [YYYY] EWHC NNNN
       - UK dates: DD/MM/YYYY, DD Month YYYY
       - Amounts: £X,XXX.XX
       - Email addresses, phone numbers
    
    3. Document-Type Specific
       - Emails: Extract From/To/CC as person entities
       - Witness statements: Extract witness name, employer
       - Contracts: Extract parties, effective date
    
    4. Entity Resolution
       - Match against existing entities (fuzzy matching)
       - Create new or link to existing
       - Track confidence scores
```

### Phase 2: Relationship Extraction

```
For chunks with multiple entities:
    1. Co-occurrence Analysis
       - Entities in same sentence = potential relationship
       - Weight by proximity
    
    2. Pattern-Based Extraction
       - "X sent Y to Z" → sent_to relationship
       - "X v Y" → party_to relationship
       - "X employed by Y" → employed_by relationship
    
    3. Email-Specific
       - From → To = sent_to
       - CC = copied_on
       - Reply chains = in_conversation_with
    
    4. Temporal Relationships
       - Events with same date = co-temporal
       - Sequential events = follows
```

### Phase 3: Timeline Construction

```
For each date entity and related content:
    1. Extract event description from surrounding context
    2. Link to entities mentioned nearby
    3. Classify event type
    4. Resolve date ambiguity (UK vs US format)
    5. Build chronological sequence
```

---

## User Editing Interface

### Entity Management Panel

```
┌─────────────────────────────────────────────────────┐
│ 📊 Case Graph - Entities                      [+Add]│
├─────────────────────────────────────────────────────┤
│ 🔍 Search: [________________] Type: [All ▼]        │
├─────────────────────────────────────────────────────┤
│ 👤 John Smith                              ✓ Verified│
│    Aliases: Mr Smith, JS, the Claimant              │
│    Role: Claimant                                   │
│    Appears in: 47 chunks                   [Edit] [🔗]│
├─────────────────────────────────────────────────────┤
│ 👤 Jane Doe                                ⚠ Unverified│
│    Aliases: Ms Doe, JD                              │
│    Role: Defendant Director                         │
│    Appears in: 23 chunks          [Merge] [Edit] [🔗]│
├─────────────────────────────────────────────────────┤
│ 🏢 Acme Corporation Ltd                    ✓ Verified│
│    Aliases: Acme, Acme Corp, the Company            │
│    Role: First Defendant                            │
│    Appears in: 156 chunks                  [Edit] [🔗]│
└─────────────────────────────────────────────────────┘
```

### Edit Entity Dialog

```
┌─────────────────────────────────────────────────────┐
│ Edit Entity: John Smith                             │
├─────────────────────────────────────────────────────┤
│ Canonical Name: [John Smith____________]            │
│                                                     │
│ Type: [Person ▼]                                    │
│                                                     │
│ Aliases (one per line):                             │
│ ┌─────────────────────────────────────────────────┐ │
│ │ Mr Smith                                        │ │
│ │ JS                                              │ │
│ │ the Claimant                                    │ │
│ │ John                                            │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ Role/Description: [Claimant, former CEO___]         │
│                                                     │
│ ☑ Mark as verified                                  │
│                                                     │
│              [Cancel]  [Save]                       │
└─────────────────────────────────────────────────────┘
```

### Merge Entities Dialog

```
┌─────────────────────────────────────────────────────┐
│ Merge Duplicate Entities                            │
├─────────────────────────────────────────────────────┤
│ These entities may be the same person:              │
│                                                     │
│ ○ John Smith (47 chunks)                            │
│   Aliases: Mr Smith, JS                             │
│                                                     │
│ ○ J. Smith (12 chunks)                              │
│   Aliases: John S                                   │
│                                                     │
│ ○ Mr John Smith (8 chunks)                          │
│   Aliases: (none)                                   │
│                                                     │
│ Keep as canonical: [John Smith ▼]                   │
│                                                     │
│ Combined aliases will be:                           │
│ Mr Smith, JS, J. Smith, John S, Mr John Smith       │
│                                                     │
│              [Cancel]  [Merge All]                  │
└─────────────────────────────────────────────────────┘
```

### Timeline View

```
┌─────────────────────────────────────────────────────┐
│ 📅 Timeline                    [2020 ▼] to [2024 ▼]│
├─────────────────────────────────────────────────────┤
│                                                     │
│ 2024                                                │
│ ├─ Jan 15  Meeting between JS and JD at Acme HQ    │
│ │          [View Sources] [Edit]                    │
│ │                                                   │
│ ├─ Jan 20  Email: JS to JD re: "Contract Terms"    │
│ │          [View Sources] [Edit]                    │
│ │                                                   │
│ ├─ Feb 01  Contract signed                          │
│ │          [View Sources] [Edit]                    │
│ │                                                   │
│ 2023                                                │
│ ├─ Dec 10  Initial proposal sent                    │
│ ...                                                 │
└─────────────────────────────────────────────────────┘
```

---

## Search Enhancement

### Query Expansion

When user searches for "John Smith":

```python
def expand_query(query: str, graph: CaseGraph) -> str:
    """Expand query with entity aliases."""
    expanded_terms = [query]
    
    # Find matching entities
    for entity in graph.find_entities(query):
        expanded_terms.extend(entity.aliases)
    
    # Return OR-joined query for BM25
    # For semantic search, embed each variant and average
    return " OR ".join(expanded_terms)
```

### Entity-Filtered Search

```python
def search_by_entity(entity_id: str, graph: CaseGraph) -> list[str]:
    """Get all chunk IDs mentioning an entity."""
    entity = graph.get_entity(entity_id)
    return entity.source_chunks
```

### Graph Context for Generation

Instead of stuffing all graph data into context, provide **targeted context**:

```python
def get_graph_context(query: str, chunks: list[Chunk], graph: CaseGraph) -> str:
    """Generate concise graph context for LLM."""
    
    # Find entities mentioned in retrieved chunks
    chunk_ids = [c.chunk_id for c in chunks]
    relevant_entities = graph.get_entities_in_chunks(chunk_ids)
    
    # Build compact context
    context_lines = ["## Key Entities Mentioned:"]
    for entity in relevant_entities[:10]:  # Limit to top 10
        aliases = ", ".join(entity.aliases[:3])
        context_lines.append(f"- {entity.canonical_name} ({entity.type}): also known as {aliases}")
    
    # Add relevant timeline events
    dates_in_query = extract_dates(query)
    if dates_in_query:
        events = graph.get_events_near_dates(dates_in_query)
        if events:
            context_lines.append("\n## Timeline Context:")
            for event in events[:5]:
                context_lines.append(f"- {event.date_text}: {event.description}")
    
    return "\n".join(context_lines)
```

This adds ~200-500 tokens of structured context, not thousands.

---

## Date Intelligence

### Smart Date Filtering

```python
class DateFilter:
    """Intelligent date filtering for search."""
    
    def __init__(self, graph: CaseGraph):
        self.graph = graph
    
    def filter_chunks(
        self,
        chunks: list[Chunk],
        date_range: tuple[date, date],
        mode: str = "document_date"  # or "mentioned_date" or "both"
    ) -> list[Chunk]:
        """Filter chunks by date with intelligence."""
        
        filtered = []
        for chunk in chunks:
            # Check document date (from metadata)
            doc_date = self._parse_date(chunk.metadata.get("document_date"))
            
            # Check dates mentioned in chunk text
            mentioned_dates = self._extract_dates_from_text(chunk.text)
            
            # Check timeline events linked to this chunk
            events = self.graph.get_events_for_chunk(chunk.chunk_id)
            event_dates = [e.date for e in events if e.date]
            
            # Apply filter based on mode
            if mode == "document_date":
                if doc_date and self._in_range(doc_date, date_range):
                    filtered.append(chunk)
            elif mode == "mentioned_date":
                if any(self._in_range(d, date_range) for d in mentioned_dates):
                    filtered.append(chunk)
            else:  # both
                all_dates = [doc_date] + mentioned_dates + event_dates
                if any(d and self._in_range(d, date_range) for d in all_dates):
                    filtered.append(chunk)
        
        return filtered
```

### Date Display in Results

Show date context in search results:

```
Result 1: Witness Statement of John Smith
📅 Document Date: 15 January 2024
📅 Dates Mentioned: 10 Dec 2023, 5 Jan 2024
Score: 0.92
```

---

## Incremental Updates

### On New Document Ingestion

```python
async def update_graph_for_document(doc: ParsedDocument, graph: CaseGraph):
    """Update case graph when new document is ingested."""
    
    # 1. Extract entities from new chunks
    new_entities = extract_entities(doc.chunks)
    
    # 2. Resolve against existing entities
    for entity in new_entities:
        existing = graph.find_similar_entity(entity)
        if existing and existing.similarity > 0.85:
            # Link to existing entity
            graph.link_chunks_to_entity(existing.id, entity.source_chunks)
            # Add any new aliases
            graph.add_aliases(existing.id, entity.aliases)
        else:
            # Create new entity
            graph.add_entity(entity)
    
    # 3. Extract new relationships
    new_relationships = extract_relationships(doc.chunks, graph)
    for rel in new_relationships:
        graph.add_relationship(rel)
    
    # 4. Update timeline
    new_events = extract_timeline_events(doc.chunks, graph)
    for event in new_events:
        graph.add_event(event)
    
    # 5. Save changes
    graph.save()
```

### Handling Deletions

When document is removed:
1. Remove chunk links from entities
2. If entity has no remaining chunks, mark as "orphaned" (don't delete - user may have edited)
3. Remove relationships with no evidence
4. Remove timeline events with no evidence

---

## Multi-Matter Support

### Matter Configuration

```python
@dataclass
class MatterConfig:
    id: str                      # UUID
    name: str                    # "Smith v Acme Ltd"
    reference: str               # "CL-2024-001234"
    client: str                  # "John Smith"
    parties: dict[str, str]      # {"claimant": "John Smith", "defendant": "Acme Ltd"}
    date_range: tuple[date, date] | None  # Relevant period
    created_at: datetime
    last_accessed: datetime
    
    # Paths (relative to matters directory)
    documents_path: str
    chroma_path: str
    bm25_path: str
    graph_path: str
```

### Matter Selector UI

```
┌─────────────────────────────────────────────────────┐
│ 📁 Matters                              [+ New Matter]│
├─────────────────────────────────────────────────────┤
│ ● Smith v Acme Ltd                        [Active]  │
│   CL-2024-001234 | 156 documents | Updated: 2h ago  │
├─────────────────────────────────────────────────────┤
│ ○ Jones Competition Appeal                          │
│   CAT-2023-0089 | 423 documents | Updated: 3d ago   │
├─────────────────────────────────────────────────────┤
│ ○ FCA Investigation - Project Alpha                 │
│   REF-2024-FCA-001 | 89 documents | Updated: 1w ago │
├─────────────────────────────────────────────────────┤
│                                                     │
│ [Archive] [Export] [Delete]                         │
└─────────────────────────────────────────────────────┘
```

### Storage Layout

```
data/
└── global/
    ├── settings.json
    └── model_cache/
```

### Matter Switching

```python
class MatterManager:
    """Manages multiple matter workspaces."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.current_matter: MatterConfig | None = None
        self._loaded_services = {}
    
    def list_matters(self) -> list[MatterConfig]:
        """List all available matters."""
        matters = []
        for matter_dir in (self.base_path / "matters").iterdir():
            config_path = matter_dir / "matter.json"
            if config_path.exists():
                matters.append(MatterConfig.from_json(config_path))
        return sorted(matters, key=lambda m: m.last_accessed, reverse=True)
    
    def switch_matter(self, matter_id: str) -> None:
        """Switch to a different matter."""
        # Unload current matter services
        self._unload_current()
        
        # Load new matter
        matter_path = self.base_path / "matters" / matter_id
        self.current_matter = MatterConfig.from_json(matter_path / "matter.json")
        
        # Update last accessed
        self.current_matter.last_accessed = datetime.now()
        self.current_matter.save()
        
        # Lazy-load services on demand
        self._loaded_services = {}
    
    def get_vector_store(self) -> VectorStore:
        """Get vector store for current matter."""
        if "vector_store" not in self._loaded_services:
            self._loaded_services["vector_store"] = VectorStore(
                persist_path=self.current_matter.chroma_path
            )
        return self._loaded_services["vector_store"]
    
    def get_case_graph(self) -> CaseGraph:
        """Get case graph for current matter."""
        if "case_graph" not in self._loaded_services:
            self._loaded_services["case_graph"] = CaseGraph(
                path=self.current_matter.graph_path
            )
        return self._loaded_services["case_graph"]
```

---

## Implementation Plan

### Phase 1: Core Graph Infrastructure (Week 1-2)
- [ ] Define entity, relationship, event schemas
- [ ] Implement CaseGraph class with CRUD operations
- [ ] JSON persistence layer
- [ ] Basic entity extraction (regex patterns)

### Phase 2: Extraction Pipeline (Week 2-3)
- [ ] Integrate spaCy NER (or local LLM extraction)
- [ ] Document-type specific extractors
- [ ] Entity resolution (fuzzy matching)
- [ ] Relationship extraction from co-occurrence

### Phase 3: User Interface (Week 3-4)
- [ ] Entity management panel
- [ ] Edit/merge dialogs
- [ ] Timeline view
- [ ] Integration with query panel

### Phase 4: Search Integration (Week 4-5)
- [ ] Query expansion with aliases
- [ ] Entity-based filtering
- [ ] Graph context for generation
- [ ] Smart date filtering modes

### Phase 5: Multi-Matter Support (Week 5-6)
- [ ] MatterManager implementation
- [ ] Matter selector UI
- [ ] Matter creation wizard
- [ ] Import/export functionality

### Phase 6: Polish & Testing (Week 6-7)
- [ ] Incremental update testing
- [ ] Performance optimization
- [ ] User testing & feedback
- [ ] Documentation

---

## Technical Decisions

### Entity Extraction Approach

**Option A: Pattern-based + spaCy NER (Recommended)**
- Fast, runs locally
- Good for structured patterns (dates, amounts, case refs)
- spaCy for person/org names
- ~95% of cases covered

**Option B: Local LLM Extraction**
- More accurate for complex relationships
- Slower, higher resource usage
- Use as secondary pass for high-value docs

### Graph Storage

**Option A: JSON files (Recommended for v1)**
- Simple, portable
- Easy to inspect/debug
- Git-friendly
- Good for <10k entities

**Option B: SQLite with FTS**
- Better for large graphs
- Full-text search on aliases
- Consider for v2

**Option C: Neo4j/NetworkX**
- Overkill for this use case
- Complex deployment

### Context Budget

When adding graph context to LLM:
- **Entity context**: Max 10 entities × ~50 tokens = 500 tokens
- **Timeline context**: Max 5 events × ~30 tokens = 150 tokens
- **Total graph overhead**: ~650 tokens (well under 1k)

This preserves context for actual document chunks.



