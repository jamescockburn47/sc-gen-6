from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, 
    QLabel, QTextEdit, QPushButton, QSplitter, QFrame, QScrollArea,
    QProgressBar, QMessageBox
)
from src.assessment.assessment_db import AssessmentDB
from src.assessment.suggestion_applicator import SuggestionApplicator
from src.assessment.suggestion_parser import SuggestionParser
from src.config_loader import get_settings
import json

class SuggestionsViewer(QWidget):
    """
    Widget to view assessment history and apply suggestions.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.db = AssessmentDB()
        self.settings = get_settings()
        self.applicator = SuggestionApplicator(self.settings) # Pass settings manager
        self._setup_ui()
        self.refresh_data()
        
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        
        # Left: Assessment History List
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        left_layout.addWidget(QLabel("Recent Assessments"))
        self.assessment_list = QListWidget()
        self.assessment_list.itemClicked.connect(self._on_assessment_selected)
        left_layout.addWidget(self.assessment_list)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_data)
        left_layout.addWidget(refresh_btn)
        
        # Right: Details & Suggestions
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Score Header
        self.score_label = QLabel("Select an assessment to view details")
        self.score_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(self.score_label)
        
        # Scores Grid
        self.scores_widget = QLabel()
        right_layout.addWidget(self.scores_widget)
        
        # Raw Response / Critique
        right_layout.addWidget(QLabel("Critique / Analysis:"))
        self.critique_text = QTextEdit()
        self.critique_text.setReadOnly(True)
        self.critique_text.setMaximumHeight(150)
        right_layout.addWidget(self.critique_text)
        
        # Suggestions List
        right_layout.addWidget(QLabel("Suggestions:"))
        self.suggestions_area = QScrollArea()
        self.suggestions_area.setWidgetResizable(True)
        self.suggestions_container = QWidget()
        self.suggestions_layout = QVBoxLayout(self.suggestions_container)
        self.suggestions_area.setWidget(self.suggestions_container)
        right_layout.addWidget(self.suggestions_area)
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
    def refresh_data(self):
        """Reload assessments from DB."""
        self.assessment_list.clear()
        assessments = self.db.get_recent_assessments(limit=20)
        
        for asm in assessments:
            item = QListWidgetItem(f"{asm['timestamp']} - Score: {asm['overall_rating']}/10")
            item.setData(Qt.UserRole, asm)
            self.assessment_list.addItem(item)
            
    def _on_assessment_selected(self, item):
        """Display details for selected assessment."""
        data = item.data(Qt.UserRole)
        
        # Update Header
        self.score_label.setText(f"Assessment from {data['timestamp']} (Provider: {data['provider']})")
        
        # Update Scores
        scores = json.loads(data['scores'])
        score_text = " | ".join([f"{k}: {v}" for k, v in scores.items()])
        self.scores_widget.setText(score_text)
        
        # Update Critique (using raw response for now, or parse it if structured)
        # The raw response is JSON, so let's try to extract the critique if possible
        try:
            raw_json = json.loads(data['raw_response'])
            critique = raw_json.get("critique", data['raw_response'])
            if isinstance(critique, dict) or isinstance(critique, list):
                critique = json.dumps(critique, indent=2)
            self.critique_text.setPlainText(str(critique))
        except:
            self.critique_text.setPlainText(data['raw_response'])
            
        # Update Suggestions
        self._populate_suggestions(data['id'])
        
    def _populate_suggestions(self, assessment_id):
        """Fetch and display suggestions for the assessment."""
        # Clear existing
        for i in reversed(range(self.suggestions_layout.count())): 
            self.suggestions_layout.itemAt(i).widget().setParent(None)
            
        # Fetch suggestions from DB (we need a method for this in DB, or filter pending)
        # Currently get_pending_suggestions gets ALL pending. 
        # We might want to show suggestions specific to this assessment.
        # Let's add a method to DB or just filter here if we had them.
        # For now, let's use the parsed suggestions from the raw response to show what WAS suggested,
        # and check DB status if possible.
        
        # Actually, let's just show actionable items.
        # We'll use SuggestionParser on the raw response to reconstruct them for display
        # since we didn't store them fully linked in a way that's easy to query by ID in the current DB method (it stores them but get_recent doesn't return them joined).
        # Wait, save_result inserts them. We should probably add `get_suggestions_for_assessment` to DB.
        # For now, I'll just parse the raw response again for display simplicity.
        
        try:
            # Re-construct EvaluationResult-like object or just parse raw json
            item = self.assessment_list.currentItem()
            data = item.data(Qt.UserRole)
            raw_json = json.loads(data['raw_response'])
            
            # Mock an EvaluationResult for the parser
            from src.assessment.assessment_models import EvaluationResult
            from datetime import datetime
            mock_result = EvaluationResult(
                provider=data['provider'],
                timestamp=datetime.now(),
                scores={},
                suggestions=raw_json.get("suggestions", []),
                prompt_improvements=raw_json.get("prompt_improvements", []),
                config_changes=raw_json.get("config_changes", {}),
                overall_rating=0,
                raw_response=""
            )
            
            parsed = SuggestionParser.parse(mock_result)
            
            for sugg in parsed:
                self._add_suggestion_card(sugg)
                
        except Exception as e:
            print(f"Error parsing suggestions: {e}")

    def _add_suggestion_card(self, suggestion):
        """Add a card for a single suggestion."""
        card = QFrame()
        card.setStyleSheet("background-color: #2a2a30; border-radius: 6px; padding: 8px;")
        layout = QVBoxLayout(card)
        
        # Type Label
        type_lbl = QLabel(suggestion['type'].replace("_", " ").title())
        type_lbl.setStyleSheet("color: #8b7cf6; font-weight: bold; font-size: 10px;")
        layout.addWidget(type_lbl)
        
        # Description
        desc = QLabel(suggestion['description'])
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Action Button
        if suggestion.get('actionable'):
            btn = QPushButton("Apply Change")
            btn.setStyleSheet("background-color: #4ade80; color: #000; font-weight: bold;")
            btn.clicked.connect(lambda: self._apply_suggestion(suggestion, btn))
            layout.addWidget(btn)
            
        self.suggestions_layout.addWidget(card)
        
    def _apply_suggestion(self, suggestion, btn):
        """Apply the suggestion."""
        if self.applicator.apply(suggestion):
            btn.setText("Applied")
            btn.setEnabled(False)
            btn.setStyleSheet("background-color: #27272a; color: #4ade80;")
            QMessageBox.information(self, "Success", "Suggestion applied successfully.")
        else:
            QMessageBox.warning(self, "Error", "Failed to apply suggestion.")
