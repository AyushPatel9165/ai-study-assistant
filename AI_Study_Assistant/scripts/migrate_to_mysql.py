"""
SQLite to MySQL Data Migration Script
Copies all existing documents, conversations, and quiz records from SQLite (data/study_assistant.db)
to MySQL (study_assistant database).

Usage:
    python scripts/migrate_to_mysql.py
"""

import sys
import os
import argparse
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

try:
    import pymysql
    import pymysql.cursors
except ImportError:
    print("[ERROR] pymysql is not installed.")
    sys.exit(1)

from src.database.db import DatabaseManager


def migrate(sqlite_path: Path):
    print("=" * 60)
    print("AI STUDY ASSISTANT: SQLITE -> MYSQL MIGRATION")
    print("=" * 60)

    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    port = int(os.getenv("MYSQL_PORT", "3306"))
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "Ayush@123")
    dbname = os.getenv("MYSQL_DB", "study_assistant")

    print(f"Source SQLite: {sqlite_path}")
    print(f"Target MySQL:  {user}@{host}:{port}/{dbname}")

    if not sqlite_path.exists():
        print(f"[ERROR] Source SQLite file not found at {sqlite_path}")
        sys.exit(1)

    # 1. Ensure database exists
    admin_conn = pymysql.connect(host=host, port=port, user=user, password=password)
    with admin_conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{dbname}` CHARACTER SET utf8mb4;")
    admin_conn.close()

    # 2. Connect to SQLite
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cur = sqlite_conn.cursor()

    # 3. Connect to MySQL
    mysql_conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=dbname,
        cursorclass=pymysql.cursors.DictCursor
    )
    mysql_cur = mysql_conn.cursor()

    # Ensure tables exist
    mysql_db = DatabaseManager(db_type="mysql")
    print(f"[OK] MySQL tables initialized (engine: {mysql_db.engine}).")

    # Migrate Documents
    sqlite_cur.execute("SELECT * FROM documents")
    docs = sqlite_cur.fetchall()
    migrated_docs = 0
    for doc in docs:
        mysql_cur.execute("""
            INSERT INTO documents (filename, file_type, file_size_bytes, chunks_count, uploaded_at)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                file_type = VALUES(file_type),
                file_size_bytes = VALUES(file_size_bytes),
                chunks_count = VALUES(chunks_count),
                uploaded_at = VALUES(uploaded_at)
        """, (
            doc["filename"],
            doc["file_type"],
            doc["file_size_bytes"],
            doc["chunks_count"],
            doc["uploaded_at"]
        ))
        migrated_docs += 1
    mysql_conn.commit()
    print(f"[OK] Documents migrated: {migrated_docs} records")

    # Migrate Conversations
    sqlite_cur.execute("SELECT * FROM conversations")
    convs = sqlite_cur.fetchall()
    migrated_convs = 0
    for c in convs:
        mysql_cur.execute("""
            INSERT INTO conversations (session_id, role, query, category, answer, confidence, tool_used, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            c["session_id"],
            c["role"],
            c["query"],
            c["category"],
            c["answer"],
            c["confidence"],
            c["tool_used"],
            c["timestamp"]
        ))
        migrated_convs += 1
    mysql_conn.commit()
    print(f"[OK] Conversations migrated: {migrated_convs} records")

    # Migrate Quizzes
    sqlite_cur.execute("SELECT * FROM quizzes")
    quizzes = sqlite_cur.fetchall()
    migrated_quizzes = 0
    for q in quizzes:
        mysql_cur.execute("""
            INSERT INTO quizzes (topic, score, total_questions, percentage, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            q["topic"],
            q["score"],
            q["total_questions"],
            q["percentage"],
            q["created_at"]
        ))
        migrated_quizzes += 1
    mysql_conn.commit()
    print(f"[OK] Quizzes migrated: {migrated_quizzes} records")

    sqlite_conn.close()
    mysql_conn.close()

    print("=" * 60)
    print("MIGRATION TO MYSQL COMPLETED SUCCESSFULLY")
    print(f"Total: {migrated_docs} docs, {migrated_convs} conversations, {migrated_quizzes} quizzes.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate SQLite to MySQL")
    parser.add_argument("--sqlite-path", default=str(BASE_DIR / "data" / "study_assistant.db"), help="Path to SQLite db")
    args = parser.parse_args()
    migrate(Path(args.sqlite_path))
