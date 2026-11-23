"""
Live Database Viewer Script
Displays current records in the active database (MySQL, PostgreSQL, or SQLite).

Usage:
    python scripts/view_data.py
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.database.db import DatabaseManager

def main():
    db = DatabaseManager()
    info = db.get_engine_info()

    print("=" * 65)
    print("LIVE DATABASE STATUS AND RECORDS")
    print("=" * 65)
    print(f"Active Database Engine: {info['engine'].upper()}")
    print(f"Target Connection:      {info['target']}")
    print(f"Connection Status:      {info['status']}")
    print("=" * 65)

    # 1. Documents
    docs = db.get_documents()
    print(f"\n[1] DOCUMENTS ({len(docs)} registered):")
    if docs:
        for d in docs:
            print(f"  - ID {d['id']}: {d['filename']} ({d['file_type']}, {d['chunks_count']} chunks, size: {d['file_size_bytes']} bytes)")
    else:
        print("  (No documents yet)")

    # 2. Conversations
    convs = db.get_conversations(limit=10)
    print(f"\n[2] RECENT CONVERSATIONS ({len(convs)} shown):")
    if convs:
        for c in convs:
            q_short = (c['query'][:50] + '...') if len(c['query']) > 50 else c['query']
            a_short = (c['answer'][:60] + '...') if len(c['answer']) > 60 else c['answer']
            cat = c.get('category') or 'General'
            print(f"  - [{c['timestamp'][:19]}] [{cat}] Q: {q_short}")
            print(f"    Ans: {a_short}")
    else:
        print("  (No conversations recorded yet)")

    # 3. Quizzes
    quizzes = db.get_quiz_results(limit=5)
    print(f"\n[3] RECENT QUIZ RESULTS ({len(quizzes)} shown):")
    if quizzes:
        for q in quizzes:
            print(f"  - [{q['created_at'][:19]}] Topic: {q['topic']} | Score: {q['score']}/{q['total_questions']} ({q['percentage']}%)")
    else:
        print("  (No quiz records yet)")

    print("\n" + "=" * 65)
    metrics = db.get_metrics()
    print(f"Summary Metrics: {metrics['total_documents']} docs, {metrics['total_questions_asked']} questions, {metrics['total_quizzes_generated']} quizzes.")
    print("=" * 65)

if __name__ == "__main__":
    main()
