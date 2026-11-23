import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.rag.pipeline import RAGPipeline

class AnswerTool:
    """
    Tool 1: Answers student questions by retrieving grounded passages from uploaded study notes.
    """
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.name = "Answer Tool"
        self.description = "Answers natural language technical questions grounded strictly in uploaded study notes."

    def execute(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        if not query or not query.strip():
            return {
                "success": False,
                "error": "Query string cannot be empty.",
                "tool_used": self.name
            }

        try:
            rag_result = self.rag_pipeline.query(query, top_k=top_k)
            return {
                "success": True,
                "tool_used": self.name,
                "query": query,
                "answer": rag_result["answer"],
                "found_in_notes": rag_result["found_in_notes"],
                "confidence": rag_result["confidence"],
                "citations": rag_result["citations"],
                "retrieved_count": rag_result["retrieved_count"]
            }
        except Exception as e:
            return {
                "success": False,
                "tool_used": self.name,
                "error": f"Answer Tool execution failed: {str(e)}"
            }
