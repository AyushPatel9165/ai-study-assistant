import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

class DatabaseManager:
    """
    Database Manager for AI Study Assistant.
    Persists document metadata, conversation history, and quiz scores in SQLite (study_assistant.db)
    and synchronizes conversation memory to data/chat_memory.json (matching the RAG architecture diagram).
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            data_dir = base_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = str(data_dir / "study_assistant.db")
            self.memory_json_path = str(data_dir / "chat_memory.json")
        else:
            self.db_path = db_path
            self.memory_json_path = str(Path(db_path).parent / "chat_memory.json")

        self._init_tables()
        self._sync_chat_memory_file()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self):
        """Initializes application SQLite tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Documents Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    chunks_count INTEGER NOT NULL,
                    uploaded_at TEXT NOT NULL
                )
            """)

            # Conversations Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    query TEXT NOT NULL,
                    category TEXT,
                    answer TEXT NOT NULL,
                    confidence REAL,
                    tool_used TEXT,
                    timestamp TEXT NOT NULL
                )
            """)

            # Quizzes Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quizzes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    total_questions INTEGER NOT NULL,
                    percentage REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            conn.commit()

    # --- Document Operations ---
    def add_document(self, filename: str, file_type: str, file_size_bytes: int, chunks_count: int) -> int:
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO documents (filename, file_type, file_size_bytes, chunks_count, uploaded_at)
                VALUES (?, ?, ?, ?, ?)
            """, (filename, file_type, file_size_bytes, chunks_count, now))
            conn.commit()
            return cursor.lastrowid

    def get_documents(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]

    def delete_document(self, filename: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM documents WHERE filename = ?", (filename,))
            conn.commit()
            return cursor.rowcount > 0

    def _sync_chat_memory_file(self):
        """Synchronizes SQLite conversations to data/chat_memory.json to match diagram."""
        try:
            convs = self.get_conversations(limit=100)
            with open(self.memory_json_path, "w", encoding="utf-8") as f:
                json.dump(convs, f, indent=2)
        except Exception:
            pass

    # --- Conversation Operations ---
    def add_conversation(
        self,
        session_id: str,
        role: str,
        query: str,
        answer: str,
        category: Optional[str] = None,
        confidence: Optional[float] = None,
        tool_used: Optional[str] = None
    ) -> int:
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (session_id, role, query, category, answer, confidence, tool_used, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (session_id, role, query, category, answer, confidence, tool_used, now))
            conn.commit()
            last_id = cursor.lastrowid

        # Keep data/chat_memory.json in sync with SQLite
        self._sync_chat_memory_file()
        return last_id

    def get_conversations(self, session_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if session_id:
                cursor.execute("""
                    SELECT * FROM conversations
                    WHERE session_id = ?
                    ORDER BY id ASC
                    LIMIT ?
                """, (session_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM conversations
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def clear_conversations(self, session_id: Optional[str] = None) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if session_id:
                cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
            else:
                cursor.execute("DELETE FROM conversations")
            conn.commit()

        self._sync_chat_memory_file()
        return True

    # --- Quiz Operations ---
    def save_quiz_result(self, topic: str, score: int, total_questions: int) -> int:
        now = datetime.utcnow().isoformat()
        pct = round((score / max(1, total_questions)) * 100, 1)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO quizzes (topic, score, total_questions, percentage, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (topic, score, total_questions, pct, now))
            conn.commit()
            return cursor.lastrowid

    def get_quiz_results(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM quizzes ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    # --- Dashboard Metrics ---
    def get_metrics(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM documents")
            total_docs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_questions = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*), AVG(percentage) FROM quizzes")
            q_row = cursor.fetchone()
            total_quizzes = q_row[0] or 0
            avg_score = round(q_row[1] or 0.0, 1)

            # Total chunks
            cursor.execute("SELECT SUM(chunks_count) FROM documents")
            total_chunks = cursor.fetchone()[0] or 0

            return {
                "total_documents": total_docs,
                "total_questions_asked": total_questions,
                "total_quizzes_generated": total_quizzes,
                "average_quiz_score": avg_score,
                "total_chunks_indexed": total_chunks
            }

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Provides full consolidated data for the frontend dashboard view."""
        metrics = self.get_metrics()
        recent_convs = self.get_conversations(limit=5)
        recent_quizzes = self.get_quiz_results(limit=4)
        documents = self.get_documents()

        # Calculate subject breakdown from conversations
        subject_counts = {"Python": 0, "Artificial Intelligence": 0, "Machine Learning": 0, "General": 0}
        for c in recent_convs:
            cat = c.get("category") or "General"
            if cat in subject_counts:
                subject_counts[cat] += 1
            else:
                subject_counts["General"] += 1

        return {
            "metrics": metrics,
            "recent_conversations": recent_convs,
            "recent_quizzes": recent_quizzes,
            "documents": documents,
            "subject_breakdown": subject_counts
        }
