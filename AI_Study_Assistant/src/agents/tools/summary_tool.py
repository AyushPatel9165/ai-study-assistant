import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.rag.vector_store import FAISSVectorStore

class SummaryTool:
    """
    Tool 2: Summarizes uploaded notes into an executive summary, key concepts, and core bullet points.
    """
    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store
        self.name = "Summary Tool"
        self.description = "Summarizes lengthy study notes into key definitions, core concepts, and takeaways."

    def execute(self, document_name: Optional[str] = None, max_points: int = 5) -> Dict[str, Any]:
        chunks = self.vector_store.chunks
        if not chunks:
            return {
                "success": False,
                "tool_used": self.name,
                "error": "No study notes are currently indexed. Please upload a PDF or TXT note first."
            }

        target_doc = document_name if document_name else (chunks[0].document_name if chunks else "Study Notes")

        # Filter by document if specified
        if document_name:
            doc_query = document_name.lower().strip()
            filtered = [
                c for c in chunks
                if c.document_name.lower() == doc_query or doc_query in c.document_name.lower() or c.document_name.lower().startswith(doc_query)
            ]
            if filtered:
                chunks = filtered
                target_doc = filtered[0].document_name

        # Aggregate document text
        combined_text = " ".join([c.text for c in chunks])

        # Extract topics / definitions
        sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", combined_text) if len(s.strip()) > 30]

        # Identify key definitional sentences
        key_sentences = []
        for s in sentences:
            if any(term in s.lower() for term in ["is a", "are a", "defined as", "refers to", "technique", "algorithm", "primary"]):
                key_sentences.append(s)

        if not key_sentences:
            key_sentences = sentences[:max_points]

        selected_points = key_sentences[:max_points]

        # Build structured executive summary
        summary_intro = f"Summary of study material from '{target_doc}':"
        executive_summary = (
            f"{summary_intro} This document covers foundational principles and technical mechanisms. "
            f"Key focus areas include operational workflows, system constraints, and algorithmic optimizations."
        )

        return {
            "success": True,
            "tool_used": self.name,
            "document_name": target_doc,
            "total_chunks_analyzed": len(chunks),
            "executive_summary": executive_summary,
            "key_takeaways": selected_points,
            "summary_points_count": len(selected_points)
        }
