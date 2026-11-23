import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import List, Dict, Any, Optional

try:
    import pymysql
    import pymysql.cursors
    PYMYSQL_AVAILABLE = True
except ImportError:
    PYMYSQL_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class DatabaseManager:
    """
    Database Manager for AI Study Assistant.
    Primary Engine: MySQL (XAMPP / MySQL Server 8.0 on port 3306).
    Fallback Engine: SQLite (local study_assistant.db).
    Also synchronizes conversation memory to data/chat_memory.json to match
    the RAG architecture diagram.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        prefer_mysql: bool = True,
        **kwargs
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent
        if DOTENV_AVAILABLE:
            load_dotenv(base_dir / ".env")

        data_dir = base_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or str(data_dir / "study_assistant.db")
        self.memory_json_path = str(Path(self.db_path).parent / "chat_memory.json")

        self.engine = "sqlite"

        # MySQL configuration from environment
        self.mysql_config = {
            "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
            "port": int(os.getenv("MYSQL_PORT", "3306")),
            "user": os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD", "Ayush@123"),
            "database": os.getenv("MYSQL_DB", "study_assistant")
        }

        # Connect to MySQL if prefer_mysql is True and not explicitly forced to a local SQLite test file
        if prefer_mysql and PYMYSQL_AVAILABLE and db_path is None:
            try:
                # Ensure study_assistant database exists
                admin_conn = pymysql.connect(
                    host=self.mysql_config["host"],
                    port=self.mysql_config["port"],
                    user=self.mysql_config["user"],
                    password=self.mysql_config["password"],
                    connect_timeout=3
                )
                with admin_conn.cursor() as cur:
                    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{self.mysql_config['database']}` CHARACTER SET utf8mb4;")
                admin_conn.close()

                # Test connect to database
                conn = pymysql.connect(
                    host=self.mysql_config["host"],
                    port=self.mysql_config["port"],
                    user=self.mysql_config["user"],
                    password=self.mysql_config["password"],
                    database=self.mysql_config["database"],
                    connect_timeout=3
                )
                conn.close()
                self.engine = "mysql"
                print(f"[DatabaseManager] Connected to MySQL: {self.mysql_config['user']}@{self.mysql_config['host']}:{self.mysql_config['port']}/{self.mysql_config['database']}")
            except Exception as e:
                print(f"[DatabaseManager] MySQL connection unavailable ({e}). Falling back to SQLite.")
                self.engine = "sqlite"

        self._init_tables()
        self._sync_chat_memory_file()

    @contextmanager
    def _get_connection(self):
        """Yields an active database connection and manages transactions."""
        if self.engine == "mysql":
            conn = pymysql.connect(
                host=self.mysql_config["host"],
                port=self.mysql_config["port"],
                user=self.mysql_config["user"],
                password=self.mysql_config["password"],
                database=self.mysql_config["database"],
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=False
            )
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_tables(self):
        """Initializes tables for MySQL or SQLite."""
        if self.engine == "mysql":
            self._init_mysql_tables()
        else:
            self._init_sqlite_tables()

    def _init_mysql_tables(self):
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        filename VARCHAR(255) UNIQUE NOT NULL,
                        file_type VARCHAR(50) NOT NULL,
                        file_size_bytes BIGINT NOT NULL,
                        chunks_count INT NOT NULL,
                        uploaded_at VARCHAR(100) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        role VARCHAR(50) NOT NULL,
                        query TEXT NOT NULL,
                        category VARCHAR(100),
                        answer TEXT NOT NULL,
                        confidence FLOAT,
                        tool_used VARCHAR(100),
                        timestamp VARCHAR(100) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS quizzes (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        topic VARCHAR(255) NOT NULL,
                        score INT NOT NULL,
                        total_questions INT NOT NULL,
                        percentage FLOAT NOT NULL,
                        created_at VARCHAR(100) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """)

    def _init_sqlite_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
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

    # --- Document Operations ---
    def add_document(self, filename: str, file_type: str, file_size_bytes: int, chunks_count: int) -> int:
        now = datetime.utcnow().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if self.engine == "mysql":
                cursor.execute("""
                    INSERT INTO documents (filename, file_type, file_size_bytes, chunks_count, uploaded_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        file_type = VALUES(file_type),
                        file_size_bytes = VALUES(file_size_bytes),
                        chunks_count = VALUES(chunks_count),
                        uploaded_at = VALUES(uploaded_at)
                """, (filename, file_type, file_size_bytes, chunks_count, now))
                return cursor.lastrowid
            else:
                cursor.execute("""
                    INSERT OR REPLACE INTO documents (filename, file_type, file_size_bytes, chunks_count, uploaded_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (filename, file_type, file_size_bytes, chunks_count, now))
                return cursor.lastrowid

    def get_documents(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]

    def delete_document(self, filename: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholder = "%s" if self.engine == "mysql" else "?"
            cursor.execute(f"DELETE FROM documents WHERE filename = {placeholder}", (filename,))
            return cursor.rowcount > 0

    def _sync_chat_memory_file(self):
        """Synchronizes conversations to data/chat_memory.json to match diagram."""
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
            if self.engine == "mysql":
                cursor.execute("""
                    INSERT INTO conversations (session_id, role, query, category, answer, confidence, tool_used, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (session_id, role, query, category, answer, confidence, tool_used, now))
                last_id = cursor.lastrowid
            else:
                cursor.execute("""
                    INSERT INTO conversations (session_id, role, query, category, answer, confidence, tool_used, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (session_id, role, query, category, answer, confidence, tool_used, now))
                last_id = cursor.lastrowid

        self._sync_chat_memory_file()
        return last_id

    def get_conversations(self, session_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            p = "%s" if self.engine == "mysql" else "?"
            if session_id:
                cursor.execute(f"""
                    SELECT * FROM conversations
                    WHERE session_id = {p}
                    ORDER BY id ASC
                    LIMIT {p}
                """, (session_id, limit))
            else:
                cursor.execute(f"""
                    SELECT * FROM conversations
                    ORDER BY id DESC
                    LIMIT {p}
                """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def clear_conversations(self, session_id: Optional[str] = None) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if session_id:
                p = "%s" if self.engine == "mysql" else "?"
                cursor.execute(f"DELETE FROM conversations WHERE session_id = {p}", (session_id,))
            else:
                cursor.execute("DELETE FROM conversations")

        self._sync_chat_memory_file()
        return True

    # --- Quiz Operations ---
    def save_quiz_result(self, topic: str, score: int, total_questions: int) -> int:
        now = datetime.utcnow().isoformat()
        pct = round((score / max(1, total_questions)) * 100, 1)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if self.engine == "mysql":
                cursor.execute("""
                    INSERT INTO quizzes (topic, score, total_questions, percentage, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (topic, score, total_questions, pct, now))
                return cursor.lastrowid
            else:
                cursor.execute("""
                    INSERT INTO quizzes (topic, score, total_questions, percentage, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (topic, score, total_questions, pct, now))
                return cursor.lastrowid

    def get_quiz_results(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            p = "%s" if self.engine == "mysql" else "?"
            cursor.execute(f"SELECT * FROM quizzes ORDER BY id DESC LIMIT {p}", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    # --- Metrics & Engine Info ---
    def get_engine_info(self) -> Dict[str, Any]:
        """Returns details on the current active database engine."""
        target = f"{self.mysql_config['user']}@{self.mysql_config['host']}:{self.mysql_config['port']}/{self.mysql_config['database']}" if self.engine == "mysql" else self.db_path
        return {
            "engine": self.engine,
            "is_mysql": self.engine == "mysql",
            "target": target,
            "status": "connected"
        }

    def get_metrics(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM documents")
            row = cursor.fetchone()
            total_docs = list(row.values())[0] if isinstance(row, dict) else row[0]

            cursor.execute("SELECT COUNT(*) FROM conversations")
            row = cursor.fetchone()
            total_questions = list(row.values())[0] if isinstance(row, dict) else row[0]

            cursor.execute("SELECT COUNT(*), AVG(percentage) FROM quizzes")
            q_row = cursor.fetchone()
            vals = list(q_row.values()) if isinstance(q_row, dict) else list(q_row)
            total_quizzes = vals[0] or 0
            avg_score = round(float(vals[1] or 0.0), 1)

            cursor.execute("SELECT SUM(chunks_count) FROM documents")
            row = cursor.fetchone()
            total_chunks = (list(row.values())[0] if isinstance(row, dict) else row[0]) or 0

            return {
                "total_documents": total_docs,
                "total_questions_asked": total_questions,
                "total_quizzes_generated": total_quizzes,
                "average_quiz_score": avg_score,
                "total_chunks_indexed": total_chunks,
                "database_engine": self.engine
            }

    # --- Adaptive Learning Recommendations ---
    def get_adaptive_recommendations(self) -> Dict[str, Any]:
        """
        Analyzes historical quiz performance from MySQL/SQLite.
        Detects weak areas and generates personalized revision recommendations.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT topic, COUNT(*) as attempts, AVG(percentage) as avg_score
                FROM quizzes
                GROUP BY topic
                ORDER BY avg_score ASC
            """)
            rows = cursor.fetchall()

        if not rows:
            return {
                "has_data": False,
                "weak_topic": None,
                "weak_score": None,
                "strong_topic": None,
                "message": "Take a practice quiz to generate personalized learning recommendations!",
                "topic_breakdown": []
            }

        topics_stats = []
        for r in rows:
            topic = r["topic"] if isinstance(r, dict) else r[0]
            attempts = r["attempts"] if isinstance(r, dict) else r[1]
            avg_score = round(float(r["avg_score"] if isinstance(r, dict) else r[2]), 1)
            status = "Strong" if avg_score >= 75 else ("Moderate" if avg_score >= 60 else "Needs Practice")
            topics_stats.append({
                "topic": topic,
                "attempts": attempts,
                "avg_score": avg_score,
                "status": status
            })

        weakest = topics_stats[0]
        strongest = topics_stats[-1]

        if weakest["avg_score"] < 70:
            rec_message = f"Your performance in '{weakest['topic']}' is {weakest['avg_score']}%. Reviewing key concept definitions and attempting 3 focused practice questions will strengthen this subject."
        else:
            rec_message = f"Great work! Your lowest topic score is {weakest['avg_score']}% in '{weakest['topic']}'. Keep up the consistent practice across all technical areas."

        return {
            "has_data": True,
            "weak_topic": weakest["topic"],
            "weak_score": weakest["avg_score"],
            "weak_status": weakest["status"],
            "strong_topic": strongest["topic"],
            "strong_score": strongest["avg_score"],
            "recommendation_message": rec_message,
            "topic_breakdown": topics_stats
        }

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Provides full consolidated data for the frontend dashboard view."""
        metrics = self.get_metrics()
        recent_convs = self.get_conversations(limit=5)
        recent_quizzes = self.get_quiz_results(limit=4)
        documents = self.get_documents()
        engine_info = self.get_engine_info()
        recommendations = self.get_adaptive_recommendations()

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
            "subject_breakdown": subject_counts,
            "engine_info": engine_info,
            "recommendations": recommendations
        }
