# SC Gen 6 - Case Graph & Multi-Matter Implementation Plan

## Overview

This document outlines the complete implementation plan for:
1. **Case Graph** - Entity extraction, relationships, timeline
2. **Multi-Matter Support** - Isolated workspaces per case
3. **Enhanced Search** - Optional graph-powered features
4. **User Interface** - Panels for graph management

**Key Principle**: All graph features are **user-optional** via toggles.

---

## Phase 1: Core Integration (Foundation)

### 1.1 Config Updates
**File**: `config/config.yaml`

Add graph configuration section:
```yaml
graph:
  enabled: true                    # Master toggle
  auto_extract: true               # Extract on ingest
  extraction_confidence: 0.7       # Min confidence for auto-add
  
search:
  use_query_expansion: false       # OFF by default - user enables
  use_graph_context: false         # OFF by default - user enables
  date_filter_mode: "document"     # "document" | "mentioned" | "both"
```

### 1.2 Settings Model Update
**File**: `src/config_loader.py`

```python
@dataclass
class GraphConfig:
    enabled: bool = True
    auto_extract: bool = True
    extraction_confidence: float = 0.7

@dataclass
class SearchConfig:
    use_query_expansion: bool = False  # User enables
    use_graph_context: bool = False    # User enables
    date_filter_mode: str = "document"
```

### 1.3 Ingestion Pipeline Hook
**File**: `src/ingestion/pipeline.py`

```python
def ingest_document(file_path, document_type=None):
    # ... existing parsing/chunking ...
    
    # Graph extraction (if enabled)
    settings = get_settings()
    if settings.graph.enabled and settings.graph.auto_extract:
        from src.graph import EntityExtractor, CaseGraph
        graph = get_current_graph()  # From matter manager or default
        extractor = EntityExtractor(graph)
        stats = extractor.extract_from_document(parsed_doc, chunks)
        graph.save()
        logger.info(f"Extracted {stats['entities_added']} entities")
    
    # ... existing vector/bm25 indexing ...
```

---

## Phase 2: Search Integration (Optional Features)

### 2.1 Query Expansion Toggle
**File**: `src/retrieval/hybrid_retriever.py`

```python
def retrieve(
    self,
    query: str,
    use_query_expansion: bool = False,  # NEW: user toggle
    use_graph_context: bool = False,    # NEW: user toggle
    ...
) -> list[dict]:
    
    # Optional query expansion
    search_query = query
    expanded_terms = []
    if use_query_expansion and self.case_graph:
        expanded_terms = self.case_graph.expand_query(query)
        if len(expanded_terms) > 1:
            # Use expanded query for BM25 (OR search)
            search_query = " OR ".join(f'"{term}"' for term in expanded_terms)
    
    # ... existing retrieval pipeline ...
    
    # Optional graph context (added to results, not to LLM directly)
    if use_graph_context and self.case_graph:
        chunk_ids = [r["chunk_id"] for r in results]
        graph_context = self.case_graph.get_graph_context(chunk_ids)
        # Attach to results for UI display / optional LLM use
        for result in results:
            result["graph_context"] = graph_context
    
    return results
```

### 2.2 UI Controls for Graph Features
**File**: `src/ui/query_panel.py`

Add toggles in Advanced Settings:

```python
# Graph-enhanced search section
graph_section = CollapsibleSection("🔗 Graph Features (Optional)")

self.query_expansion_checkbox = QCheckBox("Expand query with entity aliases")
self.query_expansion_checkbox.setToolTip(
    "Automatically include known aliases when searching.\n"
    "E.g., 'John Smith' also finds 'Mr Smith', 'JS', 'the Claimant'"
)
self.query_expansion_checkbox.setChecked(False)  # OFF by default

self.graph_context_checkbox = QCheckBox("Include entity context")
self.graph_context_checkbox.setToolTip(
    "Add brief entity/timeline context to help the LLM.\n"
    "Uses ~500 extra tokens."
)
self.graph_context_checkbox.setChecked(False)  # OFF by default

graph_section.add_widget(self.query_expansion_checkbox)
graph_section.add_widget(self.graph_context_checkbox)
```

### 2.3 Date Filter Mode Selector
**File**: `src/ui/query_panel.py`

```python
# Date filter mode
date_mode_layout = QHBoxLayout()
date_mode_label = QLabel("Filter by:")
self.date_mode_combo = QComboBox()
self.date_mode_combo.addItem("Document Date", "document")
self.date_mode_combo.addItem("Dates Mentioned", "mentioned")
self.date_mode_combo.addItem("Both", "both")
self.date_mode_combo.setToolTip(
    "Document Date: When the document was created\n"
    "Dates Mentioned: Dates appearing in the text\n"
    "Both: Match either"
)
```

---

## Phase 3: Entity Management UI

### 3.1 Entity Panel Widget
**File**: `src/ui/entity_panel.py` (NEW)

```
┌─────────────────────────────────────────────────────┐
│ 🔗 Case Graph                              [⚙️] [↻] │
├─────────────────────────────────────────────────────┤
│ 🔍 [________________] [Type: All ▼] [Verified ☐]   │
├─────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────┐ │
│ │ 👤 John Smith                        ✓ [Edit]  │ │
│ │    Mr Smith, JS, the Claimant                   │ │
│ │    Role: Claimant | 47 chunks                   │ │
│ ├─────────────────────────────────────────────────┤ │
│ │ 👤 Jane Doe                          ⚠ [Edit]  │ │
│ │    Ms Doe, JD                                   │ │
│ │    Role: Director | 23 chunks                   │ │
│ ├─────────────────────────────────────────────────┤ │
│ │ 🏢 Acme Corporation Ltd              ✓ [Edit]  │ │
│ │    Acme, the Company                            │ │
│ │    156 chunks                                   │ │
│ └─────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────┤
│ [+ Add Entity]  [Merge Selected]  [Delete]          │
└─────────────────────────────────────────────────────┘
```

**Features**:
- Search/filter entities
- Edit entity (name, aliases, type, role)
- Merge duplicates
- Delete entities
- View linked chunks
- Mark as verified

### 3.2 Entity Edit Dialog
**File**: `src/ui/dialogs/entity_edit_dialog.py` (NEW)

```python
class EntityEditDialog(QDialog):
    """Dialog for editing entity details."""
    
    def __init__(self, entity: Entity, parent=None):
        # Fields:
        # - Canonical name (text input)
        # - Type (dropdown: person, organization, etc.)
        # - Aliases (text area, one per line)
        # - Role/Description (text input)
        # - Verified checkbox
        # - View source chunks button
```

### 3.3 Entity Merge Dialog
**File**: `src/ui/dialogs/entity_merge_dialog.py` (NEW)

```python
class EntityMergeDialog(QDialog):
    """Dialog for merging duplicate entities."""
    
    def __init__(self, entities: list[Entity], parent=None):
        # Shows:
        # - List of entities to merge
        # - Dropdown to select canonical name
        # - Preview of combined aliases
        # - Confirm/Cancel
```

---

## Phase 4: Timeline View

### 4.1 Timeline Panel Widget
**File**: `src/ui/timeline_panel.py` (NEW)

```
┌─────────────────────────────────────────────────────┐
│ 📅 Timeline                    [2020 ▼] to [2024 ▼]│
├─────────────────────────────────────────────────────┤
│ Filter: [Entity: All ▼] [Type: All ▼]              │
├─────────────────────────────────────────────────────┤
│                                                     │
│ ──── 2024 ────────────────────────────────────────  │
│                                                     │
│ 📧 15 Jan    Email: JS to JD re: Contract Terms    │
│              [View] [Edit]                          │
│                                                     │
│ 📋 20 Jan    Meeting at Acme HQ                    │
│              Attendees: JS, JD, MW                  │
│              [View] [Edit]                          │
│                                                     │
│ 📝 01 Feb    Contract signed                        │
│              [View] [Edit]                          │
│                                                     │
│ ──── 2023 ────────────────────────────────────────  │
│                                                     │
│ 📧 10 Dec    Initial proposal sent                  │
│              [View] [Edit]                          │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Features**:
- Chronological event list
- Filter by date range, entity, event type
- Click to view source chunks
- Edit event details
- Add manual events

### 4.2 Event Edit Dialog
**File**: `src/ui/dialogs/event_edit_dialog.py` (NEW)

```python
class EventEditDialog(QDialog):
    """Dialog for editing timeline events."""
    
    def __init__(self, event: TimelineEvent, graph: CaseGraph, parent=None):
        # Fields:
        # - Date (date picker) or date range
        # - Date text (original text)
        # - Description (text area)
        # - Event type (dropdown)
        # - Entities involved (multi-select from graph)
        # - Verified checkbox
```

---

## Phase 5: Matter Management UI

### 5.1 Matter Selector Dialog
**File**: `src/ui/dialogs/matter_selector.py` (NEW)

```
┌─────────────────────────────────────────────────────┐
│ 📁 Select Matter                      [+ New Matter]│
├─────────────────────────────────────────────────────┤
│                                                     │
│ ● Smith v Acme Ltd                      [Active]   │
│   CL-2024-001234                                    │
│   156 docs | 1,247 chunks | 89 entities            │
│   Last accessed: 2 hours ago                        │
│                                                     │
│ ○ Jones Competition Appeal                          │
│   CAT-2023-0089                                     │
│   423 docs | 3,891 chunks | 234 entities           │
│   Last accessed: 3 days ago                         │
│                                                     │
│ ○ FCA Investigation - Project Alpha                 │
│   REF-2024-FCA-001                                  │
│   89 docs | 756 chunks | 45 entities               │
│   Last accessed: 1 week ago                         │
│                                                     │
├─────────────────────────────────────────────────────┤
│ [Import...] [Export...]        [Open] [Delete]     │
└─────────────────────────────────────────────────────┘
```

### 5.2 New Matter Wizard
**File**: `src/ui/dialogs/new_matter_wizard.py` (NEW)

```
Step 1: Basic Info
┌─────────────────────────────────────────────────────┐
│ Create New Matter                                   │
├─────────────────────────────────────────────────────┤
│ Matter Name: [Smith v Acme Ltd____________]         │
│                                                     │
│ Reference:   [CL-2024-001234______________]         │
│                                                     │
│ Client:      [John Smith__________________]         │
│                                                     │
│ ☐ Migrate existing data to this matter              │
│   (moves current documents/indexes)                 │
│                                                     │
│                      [Cancel] [Next →]              │
└─────────────────────────────────────────────────────┘

Step 2: Parties (Optional)
┌─────────────────────────────────────────────────────┐
│ Define Parties                                      │
├─────────────────────────────────────────────────────┤
│ Claimant:    [John Smith__________________]         │
│ Defendant 1: [Acme Corporation Ltd________]         │
│ Defendant 2: [Jane Doe____________________]         │
│                                           [+ Add]   │
│                                                     │
│ Date Range:                                         │
│ From: [01/01/2020]  To: [31/12/2024]               │
│                                                     │
│                      [← Back] [Create]              │
└─────────────────────────────────────────────────────┘
```

### 5.3 Matter Bar in Main Window
**File**: `src/ui/modern_main_window.py`

Add matter indicator to title bar or status bar:

```python
# In status bar or header
self.matter_label = QLabel("No matter selected")
self.matter_label.setStyleSheet("font-weight: bold;")
self.matter_button = QPushButton("Switch...")
self.matter_button.clicked.connect(self._show_matter_selector)
```

---

## Phase 6: Main Window Integration

### 6.1 Tab/Panel Layout Options

**Option A: Sidebar Tabs**
```
┌────────────────────────────────────────────────────────────┐
│ SC Gen 6 - Smith v Acme Ltd                    [_][□][X]  │
├────┬───────────────────────────────────────────────────────┤
│ 📄 │  [Document Panel]              │  [Query Panel]      │
│ 🔍 │                                │                     │
│ 🔗 │                                ├─────────────────────│
│ 📅 │                                │  [Results Panel]    │
│ ⚙️ │                                │                     │
└────┴───────────────────────────────────────────────────────┘
```
- 📄 Documents
- 🔍 Query
- 🔗 Entities
- 📅 Timeline
- ⚙️ Settings

**Option B: Collapsible Right Panel**
```
┌────────────────────────────────────────────────────────────┐
│ [Documents]        │ [Query]              │ [▼ Graph]     │
│                    │                      │               │
│                    │                      │ Entities (89) │
│                    │                      │ Timeline (45) │
│                    │──────────────────────│               │
│                    │ [Results]            │               │
└────────────────────────────────────────────────────────────┘
```

**Recommendation**: Option A (sidebar tabs) - cleaner, familiar pattern

### 6.2 Updated Main Window Structure
**File**: `src/ui/modern_main_window.py`

```python
class ModernMainWindow(QMainWindow):
    def __init__(self):
        # Matter manager
        self.matter_manager = MatterManager(Path("data"))
        
        # Central widget with sidebar
        self.sidebar = QTabWidget()
        self.sidebar.setTabPosition(QTabWidget.West)
        
        # Tabs
        self.sidebar.addTab(self.document_panel, "📄")
        self.sidebar.addTab(self.query_results_panel, "🔍")
        self.sidebar.addTab(self.entity_panel, "🔗")
        self.sidebar.addTab(self.timeline_panel, "📅")
        self.sidebar.addTab(self.settings_panel, "⚙️")
```

---

## Phase 7: Incremental Update Logic

### 7.1 On Document Add
**File**: `src/ingestion/pipeline.py`

```python
def on_document_added(doc: ParsedDocument, chunks: list[Chunk]):
    """Called after document is parsed and chunked."""
    settings = get_settings()
    
    if settings.graph.enabled and settings.graph.auto_extract:
        graph = get_current_graph()
        extractor = EntityExtractor(graph)
        
        # Extract entities (will auto-resolve against existing)
        stats = extractor.extract_from_document(doc, chunks)
        
        # Save graph
        graph.save()
        
        # Update matter stats
        if matter_manager.current_matter:
            matter_manager.update_matter_stats()
```

### 7.2 On Document Delete
**File**: `src/ingestion/pipeline.py`

```python
def on_document_deleted(document_id: str, chunk_ids: list[str]):
    """Called when document is removed."""
    settings = get_settings()
    
    if settings.graph.enabled:
        graph = get_current_graph()
        
        # Remove chunk links from entities
        for entity in graph.entities.values():
            entity.source_chunks = [
                cid for cid in entity.source_chunks 
                if cid not in chunk_ids
            ]
        
        # Mark orphaned entities (no chunks left)
        # Don't delete - user may have edited them
        orphaned = [
            e for e in graph.entities.values() 
            if not e.source_chunks and not e.user_verified
        ]
        for entity in orphaned:
            entity.metadata["orphaned"] = True
        
        graph.save()
```

### 7.3 Manual Re-extraction
**File**: `src/ui/entity_panel.py`

```python
def _on_reextract_clicked(self):
    """Re-run extraction on all documents."""
    reply = QMessageBox.question(
        self,
        "Re-extract Entities",
        "This will re-run entity extraction on all documents.\n"
        "Existing verified entities will be preserved.\n\n"
        "Continue?",
        QMessageBox.Yes | QMessageBox.No
    )
    if reply == QMessageBox.Yes:
        # Run in background thread
        self.reextract_worker = ReextractWorker(self.graph, self.vector_store)
        self.reextract_worker.start()
```

---

## Phase 8: Testing Plan

### 8.1 Unit Tests
**File**: `tests/test_case_graph.py`

```python
def test_entity_add_and_find():
    """Test adding and finding entities."""

def test_entity_merge():
    """Test merging duplicate entities."""

def test_fuzzy_matching():
    """Test entity resolution with similar names."""

def test_query_expansion():
    """Test expanding query with aliases."""

def test_graph_context_generation():
    """Test generating LLM context from graph."""

def test_timeline_filtering():
    """Test timeline date range filtering."""
```

### 8.2 Integration Tests
**File**: `tests/test_graph_integration.py`

```python
def test_extraction_during_ingest():
    """Test entity extraction during document ingestion."""

def test_search_with_expansion():
    """Test search with query expansion enabled."""

def test_search_without_expansion():
    """Test search with query expansion disabled (default)."""

def test_matter_switching():
    """Test switching between matters preserves isolation."""
```

---

## Implementation Order

### Sprint 1 (Week 1-2): Foundation
- [ ] Update config schema (`config.yaml`, `config_loader.py`)
- [ ] Hook extractor into ingestion pipeline
- [ ] Add graph toggle to settings
- [ ] Basic entity extraction on ingest

### Sprint 2 (Week 2-3): Search Integration
- [ ] Add `use_query_expansion` parameter to retriever
- [ ] Add `use_graph_context` parameter to retriever
- [ ] Add UI toggles (OFF by default)
- [ ] Enhance date filter with mode selector

### Sprint 3 (Week 3-4): Entity Management UI
- [ ] Create EntityPanelWidget
- [ ] Create EntityEditDialog
- [ ] Create EntityMergeDialog
- [ ] Integrate with main window

### Sprint 4 (Week 4-5): Timeline UI
- [ ] Create TimelinePanelWidget
- [ ] Create EventEditDialog
- [ ] Filter by date/entity/type
- [ ] Click-to-source navigation

### Sprint 5 (Week 5-6): Multi-Matter UI
- [ ] Create MatterSelectorDialog
- [ ] Create NewMatterWizard
- [ ] Add matter bar to main window
- [ ] Import/export functionality

### Sprint 6 (Week 6-7): Polish & Testing
- [ ] Unit tests for all new components
- [ ] Integration tests
- [ ] UI polish and tooltips
- [ ] Documentation update
- [ ] Performance testing with large graphs

---

## Configuration Defaults Summary

| Setting | Default | Rationale |
|---------|---------|-----------|
| `graph.enabled` | `true` | Allow extraction, but search features off |
| `graph.auto_extract` | `true` | Build graph passively |
| `search.use_query_expansion` | `false` | User explicitly enables |
| `search.use_graph_context` | `false` | User explicitly enables |
| `search.date_filter_mode` | `"document"` | Most intuitive default |

**Philosophy**: Graph populates in background, but doesn't affect search until user opts in.

---

## File Summary

### New Files to Create
```
src/ui/entity_panel.py
src/ui/timeline_panel.py
src/ui/dialogs/entity_edit_dialog.py
src/ui/dialogs/entity_merge_dialog.py
src/ui/dialogs/event_edit_dialog.py
src/ui/dialogs/matter_selector.py
src/ui/dialogs/new_matter_wizard.py
tests/test_case_graph.py
tests/test_graph_integration.py
tests/test_matter_manager.py
```

### Files to Modify
```
config/config.yaml              # Add graph/search sections
src/config_loader.py            # Add GraphConfig, SearchConfig
src/ingestion/pipeline.py       # Hook extractor
src/retrieval/hybrid_retriever.py  # Add optional graph params
src/ui/query_panel.py           # Add graph toggles
src/ui/compact_query_panel.py   # Add graph toggles
src/ui/modern_main_window.py    # Add sidebar tabs, matter bar
```


