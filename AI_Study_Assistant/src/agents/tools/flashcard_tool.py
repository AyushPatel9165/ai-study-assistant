import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.rag.vector_store import FAISSVectorStore

class FlashcardTool:
    """
    Generates interactive study flashcards from indexed document chunks in FAISS.
    Each flashcard contains a concept term/question on the front and a definition/takeaway on the back.
    """
    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store
        self.name = "Flashcard Tool"
        self.description = "Creates interactive revision flashcards from uploaded technical notes."

    def generate_flashcards(self, topic: Optional[str] = None, limit: int = 8) -> List[Dict[str, Any]]:
        """
        Extracts key definitions, architectural terms, and rules from vector store.
        """
        flashcards: List[Dict[str, Any]] = []
        chunks = self.vector_store.chunks

        if not chunks:
            # Fallback academic flashcard deck if vector store is empty
            return self._get_fallback_flashcards(topic, limit)

        # Filter chunks by topic if specified
        target_chunks = chunks
        if topic and topic.strip():
            t_lower = topic.strip().lower()
            target_chunks = [c for c in chunks if t_lower in c.document_name.lower() or t_lower in c.text.lower()]
            if not target_chunks:
                target_chunks = chunks

        card_id = 1
        seen_fronts = set()

        for c in target_chunks:
            text = c.text
            # Look for definition lines like "Term – Definition" or "Term: Definition" or "What is Term?"
            lines = re.split(r"\n+", text)
            for line in lines:
                line_clean = line.strip().lstrip("-•* ").strip()
                if not line_clean or len(line_clean) < 25:
                    continue

                front = ""
                back = ""

                # Pattern 1: Term - Description
                if "–" in line_clean or " - " in line_clean:
                    delim = "–" if "–" in line_clean else " - "
                    parts = line_clean.split(delim, 1)
                    if len(parts[0].split()) <= 6 and len(parts[1].split()) >= 4:
                        front = parts[0].strip()
                        back = parts[1].strip()

                # Pattern 2: Term: Description
                elif ":" in line_clean:
                    parts = line_clean.split(":", 1)
                    if len(parts[0].split()) <= 5 and len(parts[1].split()) >= 4:
                        front = parts[0].strip()
                        back = parts[1].strip()

                # Pattern 3: Term is a/an ...
                elif any(f" {ind} " in line_clean.lower() for ind in ["is a", "is an", "refers to", "ensures", "acts as"]):
                    match = re.search(r"^([A-Z][A-Za-z0-9\s\(\)]+?)\s+(is a|is an|refers to|ensures|acts as)\s+(.*)", line_clean)
                    if match and len(match.group(1).split()) <= 5:
                        front = match.group(1).strip()
                        back = f"{match.group(2).capitalize()} {match.group(3).strip()}"

                if front and back:
                    front_norm = front.lower()
                    if front_norm not in seen_fronts and len(front) <= 60:
                        seen_fronts.add(front_norm)

                        # Determine subject category for flashcard
                        combined_text = f"{front} {back} {c.document_name}".lower()
                        if any(k in combined_text for k in ["python", "gil", "generator", "yield", "comprehension", "threading", "interpreter"]):
                            card_topic = "Python"
                        elif any(k in combined_text for k in ["neural", "machine learning", "gradient", "loss", "backpropagation", "regularization", "overfitting", "deep learning"]):
                            card_topic = "Machine Learning"
                        elif any(k in combined_text for k in ["heuristic", "search", "a*", "agent", "planning", "inference", "admissible"]):
                            card_topic = "AI"
                        elif "mongo" in combined_text:
                            card_topic = "MongoDB"
                        else:
                            card_topic = c.document_name.replace(".pdf", "").replace(".txt", "").replace("_", " ").title()

                        flashcards.append({
                            "id": card_id,
                            "topic": card_topic,
                            "front": front,
                            "back": back,
                            "document": c.document_name,
                            "page": c.page_number
                        })
                        card_id += 1

                if len(flashcards) >= limit:
                    break
            if len(flashcards) >= limit:
                break

        if len(flashcards) < 3:
            flashcards.extend(self._get_fallback_flashcards(topic, limit - len(flashcards)))

        return flashcards[:limit]

    def _get_fallback_flashcards(self, topic: Optional[str] = None, limit: int = 8) -> List[Dict[str, Any]]:
        default_cards = [
            {
                "id": 101,
                "topic": "Python Internals",
                "front": "Global Interpreter Lock (GIL)",
                "back": "A mutex in CPython that protects access to Python objects, preventing multiple threads from executing bytecode simultaneously on multi-core CPUs.",
                "document": "Python Core",
                "page": 1
            },
            {
                "id": 102,
                "topic": "Python Internals",
                "front": "Python Generators & Yield",
                "back": "Functions that produce values lazily on-the-fly via the 'yield' keyword, preserving execution state and saving memory over standard lists.",
                "document": "Python Core",
                "page": 2
            },
            {
                "id": 103,
                "topic": "Machine Learning",
                "front": "L1 Regularization (Lasso)",
                "back": "Adds absolute weight penalty to the loss function (lambda * |w|), driving irrelevant feature coefficients strictly to zero to induce sparsity.",
                "document": "ML Notes",
                "page": 1
            },
            {
                "id": 104,
                "topic": "Machine Learning",
                "front": "L2 Regularization (Ridge)",
                "back": "Adds squared weight penalty (lambda * w^2), shrinking weights smoothly toward zero without eliminating them, preventing overfitting.",
                "document": "ML Notes",
                "page": 1
            },
            {
                "id": 105,
                "topic": "Artificial Intelligence",
                "front": "A* Search Algorithm",
                "back": "An informed heuristic graph traversal algorithm using evaluation function f(n) = g(n) + h(n), guaranteed to find optimal paths when h(n) is admissible.",
                "document": "AI Notes",
                "page": 1
            },
            {
                "id": 106,
                "topic": "Databases (MySQL)",
                "front": "ACID Properties",
                "back": "Atomicity, Consistency, Isolation, and Durability - the fundamental guarantees provided by relational database transaction engines (like InnoDB).",
                "document": "MySQL Handbook",
                "page": 1
            },
            {
                "id": 107,
                "topic": "Databases (MongoDB)",
                "front": "BSON Format",
                "back": "Binary JSON representation used by MongoDB to store document collections with support for richer data types like Date, Regex, and Binary Data.",
                "document": "MongoDB Handbook",
                "page": 1
            },
            {
                "id": 108,
                "topic": "Deep Learning",
                "front": "Backpropagation",
                "back": "The algorithm that computes the gradient of the loss function with respect to neural network weights using the chain rule of calculus to update weights via gradient descent.",
                "document": "Deep Learning Notes",
                "page": 1
            }
        ]
        return default_cards[:limit]
