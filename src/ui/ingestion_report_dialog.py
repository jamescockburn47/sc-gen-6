"""Ingestion Report Dialog - displays after ingestion completes."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QWidget,
    QHeaderView, QFrame
)
from PySide6.QtCore import Qt, Signal

from src.ingestion.ingestion_report import IngestionReport


class IngestionReportDialog(QDialog):
    """Dialog displaying ingestion results."""
    
    # Signal emitted when user clicks Retry Failed
    retry_requested = Signal(list)  # List of failed file paths
    
    def __init__(self, report: IngestionReport, parent=None):
        super().__init__(parent)
        self.report = report
        self.setWindowTitle("Ingestion Report")
        self.setMinimumSize(700, 500)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        
        # Summary header
        summary = self.report.get_summary()
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a1a2e, stop:1 #16213e);
                border-radius: 8px;
                padding: 16px;
            }
            QLabel { color: white; }
        """)
        header_layout = QHBoxLayout(header)
        
        # Stats boxes
        stats = [
            ("✅ Success", str(summary["success"]), "#4ade80"),
            ("❌ Failed", str(summary["failed"]), "#f87171"),
            ("⚠️ Skipped", str(summary["skipped"]), "#fbbf24"),
            ("📦 Chunks", str(summary["total_chunks"]), "#60a5fa"),
            ("⏱️ Time", f"{summary['duration_seconds']:.1f}s", "#a78bfa"),
        ]
        
        for label, value, color in stats:
            stat_box = QVBoxLayout()
            value_label = QLabel(value)
            value_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color};")
            value_label.setAlignment(Qt.AlignCenter)
            
            name_label = QLabel(label)
            name_label.setStyleSheet("font-size: 11px; opacity: 0.8;")
            name_label.setAlignment(Qt.AlignCenter)
            
            stat_box.addWidget(value_label)
            stat_box.addWidget(name_label)
            header_layout.addLayout(stat_box)
        
        layout.addWidget(header)
        
        # Tabs for different result types
        tabs = QTabWidget()
        
        # All results tab
        all_table = self._create_results_table(self.report.results)
        tabs.addTab(all_table, f"All ({len(self.report.results)})")
        
        # Failed tab
        failed = self.report.get_failed()
        if failed:
            failed_table = self._create_results_table(failed)
            tabs.addTab(failed_table, f"❌ Failed ({len(failed)})")
        
        # Successful tab
        success = self.report.get_successful()
        if success:
            success_table = self._create_results_table(success)
            tabs.addTab(success_table, f"✅ Success ({len(success)})")
        
        layout.addWidget(tabs)
        
        # Button row
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        # Retry Failed button (only if there are failures)
        failed = self.report.get_failed()
        if failed:
            retry_btn = QPushButton(f"⟳ Retry Failed ({len(failed)})")
            retry_btn.setStyleSheet("""
                QPushButton {
                    background: #f59e0b;
                    color: white;
                    border: none;
                    padding: 8px 24px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: #d97706;
                }
            """)
            retry_btn.clicked.connect(self._on_retry_clicked)
            button_layout.addWidget(retry_btn)
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background: #3b82f6;
                color: white;
                border: none;
                padding: 8px 24px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #2563eb;
            }
        """)
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def _on_retry_clicked(self):
        """Handle retry button click."""
        failed_paths = [r.file_path for r in self.report.get_failed()]
        self.retry_requested.emit(failed_paths)
        self.accept()
    
    def _create_results_table(self, results) -> QTableWidget:
        """Create a table widget for results."""
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Status", "File", "Type", "Chunks", "Error"])
        table.setRowCount(len(results))
        
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        
        table.setAlternatingRowColors(True)
        table.setStyleSheet("""
            QTableWidget {
                background: #1e1e2e;
                alternate-background-color: #262636;
                gridline-color: #363646;
            }
            QHeaderView::section {
                background: #2a2a3a;
                color: white;
                padding: 8px;
                border: none;
            }
        """)
        
        for row, result in enumerate(results):
            # Status icon
            status_icons = {"success": "✅", "failed": "❌", "skipped": "⚠️"}
            status_item = QTableWidgetItem(status_icons.get(result.status, "?"))
            status_item.setTextAlignment(Qt.AlignCenter)
            table.setItem(row, 0, status_item)
            
            # File name
            table.setItem(row, 1, QTableWidgetItem(result.file_name))
            
            # Document type
            doc_type = result.document_type or "-"
            table.setItem(row, 2, QTableWidgetItem(doc_type))
            
            # Chunk count
            chunks = str(result.chunk_count) if result.status == "success" else "-"
            table.setItem(row, 3, QTableWidgetItem(chunks))
            
            # Error
            error = result.error or ""
            table.setItem(row, 4, QTableWidgetItem(error[:100] + "..." if len(error) > 100 else error))
        
        return table


def show_ingestion_report(report: IngestionReport, parent=None):
    """Show the ingestion report dialog."""
    dialog = IngestionReportDialog(report, parent)
    dialog.exec()
