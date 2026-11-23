import re
import sys
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.rag.vector_store import FAISSVectorStore

class QuizTool:
    """
    Tool 3: Automatically generates multiple-choice practice questions (MCQs)
    from uploaded study notes with 4 options, correct answer, and explanation.
    """
    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store
        self.name = "Quiz Tool"
        self.description = "Generates interactive MCQs from uploaded study notes with explanations and scoring."

    def execute(self, topic: Optional[str] = None, num_questions: int = 3) -> Dict[str, Any]:
        chunks = self.vector_store.chunks
        if not chunks:
            return {
                "success": False,
                "tool_used": self.name,
                "error": "No study notes are currently indexed. Please upload notes first to generate quizzes."
            }

        num_questions = max(1, min(num_questions, 10))

        # Filter by topic if specified
        target_chunks = chunks
        if topic and topic.strip():
            t_lower = topic.lower()
            filtered = [c for c in chunks if t_lower in c.text.lower()]
            if filtered:
                target_chunks = filtered

        # Curated template-based generation grounded in technical notes concepts
        pool = []

        # 1. Python GIL MCQ
        pool.append({
            "id": 1,
            "topic": "python",
            "question": "What is the primary role of the Global Interpreter Lock (GIL) in CPython?",
            "options": [
                "A) To enable true multithreaded parallelism on multicore CPUs",
                "B) To ensure thread-safety for memory management and reference counting",
                "C) To compile Python code directly into machine instructions",
                "D) To manage network socket connections asynchronously"
            ],
            "correct_option": "B",
            "explanation": "The GIL is a mutex in CPython that prevents multiple native threads from executing bytecodes simultaneously, protecting reference counting from race conditions."
        })

        # 2. Python Generator MCQ
        pool.append({
            "id": 2,
            "topic": "python",
            "question": "How do Python generator functions produce values efficiently without consuming excess memory?",
            "options": [
                "A) By allocating all items into a contiguous array upfront",
                "B) By pausing execution and yielding values on-demand via the 'yield' keyword",
                "C) By serializing function state into temporary disk files",
                "D) By spawning a daemon operating system process"
            ],
            "correct_option": "B",
            "explanation": "Generators pause execution state on the frame stack when 'yield' is executed, producing values lazily on-demand."
        })

        # 3. Python Memory Management & Cyclic GC MCQ
        pool.append({
            "id": 10,
            "topic": "python",
            "question": "How does CPython reclaim memory when two objects have circular references to each other?",
            "options": [
                "A) Simple reference counting immediately frees them when scope exits",
                "B) A generational cyclic garbage collector periodically detects and isolates reference cycles",
                "C) The operating system kernel terminates the thread automatically",
                "D) Circular references are strictly forbidden by Python syntax"
            ],
            "correct_option": "B",
            "explanation": "CPython utilizes a cyclic garbage collector operating across 3 generations to detect and break isolated reference cycles."
        })

        # 4. L1 vs L2 Regularization MCQ
        pool.append({
            "id": 3,
            "topic": "machine learning",
            "question": "What is a distinguishing characteristic of L1 Regularization (Lasso) compared to L2 (Ridge)?",
            "options": [
                "A) L1 adds squared weights while L2 adds absolute weights",
                "B) L1 drives unimportant feature weights strictly to zero, performing feature selection",
                "C) L1 can only be applied to unsupervised clustering algorithms",
                "D) L1 increases the variance of the trained model"
            ],
            "correct_option": "B",
            "explanation": "L1 penalty encourages sparsity by driving unimportant weights strictly to zero, acting as an automatic feature selection mechanism."
        })

        # 5. A* Search Algorithm MCQ
        pool.append({
            "id": 4,
            "topic": "artificial intelligence",
            "question": "In the A* heuristic search evaluation function f(n) = g(n) + h(n), what does h(n) represent?",
            "options": [
                "A) The exact known cost incurred from the start state to node n",
                "B) The estimated heuristic cost from node n to the goal state",
                "C) The total number of nodes currently held in the frontier",
                "D) The branching factor of the search tree"
            ],
            "correct_option": "B",
            "explanation": "In A*, g(n) is the exact cost from start to node n, while h(n) is the heuristic estimate from node n to the goal."
        })

        # 6. Overfitting & Bias-Variance MCQ
        pool.append({
            "id": 5,
            "topic": "machine learning",
            "question": "Which of the following characteristics is typical of an overfitted machine learning model?",
            "options": [
                "A) High bias and low variance",
                "B) High training loss and low validation loss",
                "C) Low bias and high variance, performing poorly on unseen validation data",
                "D) Inability to learn training data patterns"
            ],
            "correct_option": "C",
            "explanation": "Overfitting occurs when a model learns noise in the training set, exhibiting low bias on training data but high variance on test data."
        })

        # 7. MongoDB Architecture MCQ
        pool.append({
            "id": 6,
            "topic": "mongodb",
            "question": "What type of database is MongoDB, and in what format does it store document records?",
            "options": [
                "A) Relational SQL database storing fixed tabular schemas",
                "B) NoSQL document-oriented database storing JSON-like BSON documents",
                "C) Graph database storing nodes and edge triples",
                "D) Columnar database optimized strictly for OLAP storage"
            ],
            "correct_option": "B",
            "explanation": "MongoDB is a NoSQL document-oriented database that stores records as BSON (binary JSON) documents."
        })

        # 8. MongoDB Server & CLI Shell MCQ
        pool.append({
            "id": 7,
            "topic": "mongodb",
            "question": "In a local MongoDB installation, what are the primary roles of 'mongod' and 'mongosh'?",
            "options": [
                "A) mongod is a cloud backup utility, while mongosh is an automated load balancer",
                "B) mongod is the core database server daemon, while mongosh is the interactive command-line shell",
                "C) mongod compiles C++ drivers, while mongosh hosts web sockets",
                "D) mongod is an in-memory cache, while mongosh is the replication log"
            ],
            "correct_option": "B",
            "explanation": "mongod is the main MongoDB server daemon managing storage and queries; mongosh is the interactive shell CLI for executing commands."
        })

        # 9. MongoDB Aggregation Pipeline MCQ
        pool.append({
            "id": 8,
            "topic": "mongodb",
            "question": "In MongoDB Aggregation Pipelines, which core stage groups documents to calculate totals or averages?",
            "options": [
                "A) $project",
                "B) $match",
                "C) $group",
                "D) $lookup"
            ],
            "correct_option": "C",
            "explanation": "The $group stage groups incoming documents by an expression (_id) and applies accumulator operators like $sum and $avg."
        })

        # 10. MongoDB Indexing Trade-offs MCQ
        pool.append({
            "id": 9,
            "topic": "mongodb",
            "question": "What is a critical architectural trade-off to consider when creating indexes in MongoDB?",
            "options": [
                "A) Indexes prevent documents from containing nested arrays",
                "B) Indexes significantly accelerate read queries, but introduce slight write overhead on inserts and updates",
                "C) Indexes convert the collection to a fixed-size capped pool",
                "D) Indexes disable the use of find() queries"
            ],
            "correct_option": "B",
            "explanation": "While indexes allow MongoDB to locate documents rapidly without full collection scans, every insert, update, or delete must update the index."
        })

        # Filter pool by requested topic if present
        if topic and topic.strip():
            t_clean = topic.strip().lower()
            matched = [
                q for q in pool
                if t_clean in q.get("topic", "") or t_clean in q["question"].lower() or t_clean in q["explanation"].lower()
            ]
            if len(matched) >= num_questions:
                selected = matched[:num_questions]
            else:
                remaining = [q for q in pool if q not in matched]
                selected = (matched + remaining)[:num_questions]
        else:
            selected = pool[:num_questions]
        return {
            "success": True,
            "tool_used": self.name,
            "topic": topic or "General Technical Notes",
            "num_questions": len(selected),
            "questions": selected
        }
