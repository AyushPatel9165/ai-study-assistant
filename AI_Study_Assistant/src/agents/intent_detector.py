import re
from typing import Dict, Any, Optional

class IntentDetector:
    """
    Analyzes student input prompts to determine user intention:
    - ANSWER: Student is asking a conceptual or technical question.
    - SUMMARY: Student wants a summary, overview, or bulleted key points of notes.
    - QUIZ: Student wants practice MCQs, test questions, or a quiz.
    """

    # Keyword and pattern mappings for high precision
    QUIZ_KEYWORDS = [
        "quiz", "mcq", "mcqs", "multiple choice", "test me", "practice questions",
        "question paper", "test questions", "create test", "generate quiz"
    ]

    SUMMARY_KEYWORDS = [
        "summarize", "summary", "overview", "brief", "key points", "takeaways",
        "tldr", "synopsis", "notes summary", "short notes", "recap"
    ]

    ANSWER_KEYWORDS = [
        "what is", "how does", "explain", "why is", "define", "difference between",
        "tell me about", "describe", "can you explain", "what are", "how to"
    ]

    def __init__(self):
        pass

    def detect_intent(self, prompt: str) -> Dict[str, Any]:
        """
        Determines the student's intent and extracts parameters like topic, num_questions, etc.
        """
        if not prompt or not prompt.strip():
            return {
                "intent": "ANSWER",
                "confidence": 0.0,
                "reasoning": "Empty input defaulted to ANSWER.",
                "parameters": {}
            }

        cleaned = prompt.strip().lower()

        # Check for Quiz Intent
        quiz_score = sum(2 for kw in self.QUIZ_KEYWORDS if kw in cleaned)
        if re.search(r"\b\d+\s*(mcq|mcqs|questions|quiz)\b", cleaned):
            quiz_score += 3

        # Check for Summary Intent
        summary_score = sum(2 for kw in self.SUMMARY_KEYWORDS if kw in cleaned)

        # Check for Answer Intent
        answer_score = sum(1 for kw in self.ANSWER_KEYWORDS if kw in cleaned)
        if cleaned.endswith("?"):
            answer_score += 1

        # Decision Logic with confidence estimation
        if quiz_score > 0 and quiz_score >= summary_score:
            intent = "QUIZ"
            confidence = min(0.99, 0.70 + (quiz_score * 0.08))
            reasoning = f"Matched quiz-related keywords (score={quiz_score})."
            params = self._extract_quiz_params(prompt)
        elif summary_score > 0 and summary_score > quiz_score:
            intent = "SUMMARY"
            confidence = min(0.99, 0.70 + (summary_score * 0.08))
            reasoning = f"Matched summary-related keywords (score={summary_score})."
            params = self._extract_summary_params(prompt)
        else:
            intent = "ANSWER"
            confidence = 0.85 if answer_score > 0 else 0.70
            reasoning = f"Defaulted to conceptual Q&A based on input structure (score={answer_score})."
            params = {"query": prompt.strip()}

        return {
            "intent": intent,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
            "parameters": params
        }

    def _extract_quiz_params(self, text: str) -> Dict[str, Any]:
        """Extract number of questions and target topic if present."""
        num_questions = 3  # default
        num_match = re.search(r"\b(\d+)\s*(mcq|mcqs|questions|items)?\b", text, re.IGNORECASE)
        if num_match:
            try:
                val = int(num_match.group(1))
                if 1 <= val <= 10:
                    num_questions = val
            except ValueError:
                pass

        # Extract topic heuristic with support for all indexed subjects
        topic = None
        tech_map = [
            ("mongodb", "MongoDB"),
            ("mongo", "MongoDB"),
            ("python", "Python"),
            ("machine learning", "Machine Learning"),
            ("ml", "Machine Learning"),
            ("artificial intelligence", "Artificial Intelligence"),
            ("ai", "Artificial Intelligence"),
            ("regularization", "Machine Learning"),
            ("gil", "Python"),
            ("generator", "Python")
        ]
        for tech, canonical in tech_map:
            if re.search(rf"\b{tech}\b", text.lower()):
                topic = canonical
                break

        return {
            "num_questions": num_questions,
            "topic": topic
        }

    def _extract_summary_params(self, text: str) -> Dict[str, Any]:
        """Extract document name or topic for summary."""
        doc_name = None
        t_low = text.lower()
        if "mongo" in t_low:
            doc_name = "MongoDB Handbook.pdf"
        elif any(t in t_low for t in ["sample", "python", "gil", "regularization", "ml notes"]):
            doc_name = "sample_study_notes.txt"

        # Clean leading action phrases before extracting file name
        cleaned = re.sub(r"^(summarize|summary of|give summary of|overview of|synopsis of)\s+", "", text.strip(), flags=re.IGNORECASE)
        doc_match = re.search(r"([A-Za-z0-9_\- ]+?\.(?:pdf|txt))\b", cleaned, re.IGNORECASE)
        if doc_match:
            doc_name = doc_match.group(1).strip()

        return {
            "document_name": doc_name,
            "max_points": 5
        }
