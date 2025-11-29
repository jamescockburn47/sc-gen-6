"""Dialog for editing document metadata (label, category, graph flag)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from src.documents.catalog import DocumentCatalog, DocumentRecord


class DocumentMetadataDialog(QDialog):
    """Modal dialog to edit a single document's metadata."""

    def __init__(self, catalog: DocumentCatalog, record: DocumentRecord, parent=None):
        super().__init__(parent)
        self.catalog = catalog
        self.record = record
        self._reprocess = True
        self.new_label = record.label
        self.new_category = record.category
        self.new_graph_flag = record.include_in_graph
        self.setWindowTitle("Edit Document Metadata")
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.label_edit = QLineEdit(self.record.label)
        form.addRow("Label:", self.label_edit)

        self.category_combo = QComboBox()
        for category in self.catalog.CONFIGURED_CATEGORIES:
            self.category_combo.addItem(category.replace("_", " ").title(), category)
        current_index = self.category_combo.findData(self.record.category)
        if current_index >= 0:
            self.category_combo.setCurrentIndex(current_index)
        form.addRow("Category:", self.category_combo)

        self.graph_checkbox = QCheckBox("Include in case graph")
        self.graph_checkbox.setChecked(self.record.include_in_graph)
        form.addRow("", self.graph_checkbox)

        layout.addLayout(form)

        self.reprocess_checkbox = QCheckBox("Reprocess document after saving")
        self.reprocess_checkbox.setChecked(True)
        layout.addWidget(self.reprocess_checkbox)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        self.new_label = self.label_edit.text().strip()
        self.new_category = self.category_combo.currentData()
        self.new_graph_flag = self.graph_checkbox.isChecked()
        self._reprocess = self.reprocess_checkbox.isChecked()
        super().accept()

    # ------------------------------------------------------------------#
    # Accessors after exec()
    # ------------------------------------------------------------------#
    def updated_values(self) -> dict:
        return {
            "label": self.new_label,
            "category": self.new_category,
            "include_in_graph": self.new_graph_flag,
        }

    def should_reprocess(self) -> bool:
        return self._reprocess

