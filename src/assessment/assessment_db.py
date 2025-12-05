import sqlite3
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from .assessment_models import EvaluationResult

class AssessmentDB:
    """
    Manages storage and retrieval of assessment results and suggestions.
    """
    
    def __init__(self, db_path: str = "data/assessment_history.db"):
        self.db_path = Path(db_path)
        self._ensure_db_dir()
        self._init_db()
        
    def _ensure_db_dir(self):
        """Ensure the database directory exists."""
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Assessments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                provider TEXT,
                overall_rating REAL,
                scores TEXT,
                raw_response TEXT
            )
        ''')
        
        # Suggestions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                assessment_id INTEGER,
                type TEXT,
                description TEXT,
                config_change TEXT,
                applied BOOLEAN DEFAULT 0,
                applied_timestamp TEXT,
                FOREIGN KEY(assessment_id) REFERENCES assessments(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def save_result(self, result: EvaluationResult) -> int:
        """
        Save an evaluation result and its suggestions.
        Returns the assessment ID.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert assessment
        cursor.execute('''
            INSERT INTO assessments (timestamp, provider, overall_rating, scores, raw_response)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            result.timestamp.isoformat(),
            result.provider,
            result.overall_rating,
            json.dumps(result.scores),
            result.raw_response
        ))
        
        assessment_id = cursor.lastrowid
        
        # Insert suggestions
        for suggestion in result.suggestions:
            cursor.execute('''
                INSERT INTO suggestions (assessment_id, type, description, config_change)
                VALUES (?, ?, ?, ?)
            ''', (
                assessment_id,
                "general",
                suggestion,
                None
            ))
            
        # Insert specific config changes
        for key, value in result.config_changes.items():
            cursor.execute('''
                INSERT INTO suggestions (assessment_id, type, description, config_change)
                VALUES (?, ?, ?, ?)
            ''', (
                assessment_id,
                "config",
                f"Change {key} to {value}",
                json.dumps({key: value})
            ))
            
        conn.commit()
        conn.close()
        return assessment_id

    def get_recent_assessments(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent assessments."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM assessments ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

    def get_pending_suggestions(self) -> List[Dict[str, Any]]:
        """Get unapplied suggestions."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM suggestions WHERE applied = 0 ORDER BY id DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]

    def mark_suggestion_applied(self, suggestion_id: int):
        """Mark a suggestion as applied."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE suggestions 
            SET applied = 1, applied_timestamp = ? 
            WHERE id = ?
        ''', (datetime.now().isoformat(), suggestion_id))
        
        conn.commit()
        conn.close()
