import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.rag.document_loader import DocumentLoader
from src.rag.chunker import TextChunker, DocumentChunk
from src.rag.vector_store import FAISSVectorStore
from src.rag.generator import RAGGenerator
from src.nlp.transformer_pipeline import TransformerQuestionUnderstander

class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation (RAG) pipeline for student study notes.
    Orchestrates loading, chunking, embedding, FAISS indexing, semantic retrieval,
    and grounded generation.
    """
    def __init__(
        self,
        index_path: str = "models/faiss_index.bin",
        metadata_path: str = "models/faiss_chunks.json",
        chunk_size: int = 400,
        chunk_overlap: int = 60,
        confidence_threshold: float = 0.35
    ):
        project_root = Path(__file__).resolve().parent.parent.parent
        self.index_path = str(project_root / index_path) if not os.path.isabs(index_path) else index_path
        self.metadata_path = str(project_root / metadata_path) if not os.path.isabs(metadata_path) else metadata_path
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vector_store = FAISSVectorStore(dimension=384)
        self.generator = RAGGenerator(confidence_threshold=confidence_threshold)
        self.embedder = TransformerQuestionUnderstander()
        self._load_existing_index()

        # If store is empty, auto-load sample study notes if available
        if self.vector_store.total_vectors == 0:
            sample_file = project_root / "data" / "raw" / "sample_study_notes.txt"
            if sample_file.exists():
                try:
                    self.index_document(str(sample_file))
                except Exception:
                    pass

    def _load_existing_index(self):
        """Attempts to load persisted FAISS index if present."""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            try:
                self.vector_store.load(self.index_path, self.metadata_path)
            except Exception:
                pass

    def index_document(self, file_path: str) -> Dict[str, Any]:
        """
        Parses, chunks, embeds, and indexes a PDF or TXT study note.
        """
        pages = DocumentLoader.load_document(file_path)
        all_chunks: List[DocumentChunk] = []

        for page in pages:
            page_chunks = self.chunker.chunk_text(
                text=page.text,
                document_name=page.document_name,
                page_number=page.page_number
            )
            all_chunks.extend(page_chunks)

        if not all_chunks:
            raise ValueError(f"No text chunks could be produced from {file_path}")

        # Batch encode chunk texts using SentenceTransformer
        chunk_texts = [c.text for c in all_chunks]
        embeddings = self.embedder.encode_batch(chunk_texts)

        # Store in FAISS
        self.vector_store.add_chunks(all_chunks, embeddings)

        # Persist index and metadata
        self.vector_store.save(self.index_path, self.metadata_path)

        return {
            "document_name": Path(file_path).name,
            "pages_processed": len(pages),
            "chunks_created": len(all_chunks),
            "total_indexed_chunks": self.vector_store.total_vectors
        }

    @staticmethod
    def expand_academic_terms(text: str) -> str:
        """Expands common abbreviations so dense semantic search matches technical notes."""
        import re
        t = re.sub(r"\bml\b", "machine learning", text, flags=re.IGNORECASE)
        t = re.sub(r"\bai\b", "artificial intelligence", t, flags=re.IGNORECASE)
        t = re.sub(r"\bpy\b", "python", t, flags=re.IGNORECASE)
        t = re.sub(r"\bgil\b", "Global Interpreter Lock", t, flags=re.IGNORECASE)
        t = re.sub(r"\bgc\b", "garbage collection memory", t, flags=re.IGNORECASE)
        t = re.sub(r"\bdl\b", "deep learning", t, flags=re.IGNORECASE)
        t = re.sub(r"\bnn\b", "neural network", t, flags=re.IGNORECASE)
        return t

    @staticmethod
    def resolve_contextual_query(question: str, previous_query: Optional[str] = None) -> str:
        """
        Resolves pronouns and follow-up references using previous conversation query.
        E.g. 'How can I bypass it?' + previous 'GIL in Python' -> 'How can I bypass GIL in Python?'
        """
        import re
        if not previous_query or not previous_query.strip():
            return question

        q_lower = question.lower()
        pronouns = [r"\bit\b", r"\bthis\b", r"\bthese\b", r"\bthat\b", r"\bthe above\b", r"\bthem\b"]
        is_followup = any(re.search(p, q_lower) for p in pronouns) or len(question.split()) <= 3

        if is_followup:
            prev_terms = re.findall(r"\b[A-Za-z0-9_]{3,}\b", previous_query)
            stop_words = {"what", "how", "why", "does", "explain", "the", "and", "for", "with", "from", "can", "you", "tell", "about"}
            meaningful_prev = [t for t in prev_terms if t.lower() not in stop_words][:3]
            if meaningful_prev:
                return f"{question} ({' '.join(meaningful_prev)})"

        return question

    def query(self, question: str, previous_query: Optional[str] = None, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieves relevant study note segments and generates a grounded answer with citations.
        Supports multi-turn conversational follow-ups.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        resolved_q = self.resolve_contextual_query(question, previous_query)
        expanded_q = self.expand_academic_terms(resolved_q)
        query_emb = self.embedder.encode(expanded_q)
        retrieved_chunks = self.vector_store.similarity_search(
            query_embedding=query_emb,
            top_k=top_k,
            score_threshold=self.generator.confidence_threshold
        )

        # Fallback for short broad topic queries (<= 4 words, e.g. 'ml notes', 'gradient descent')
        if not retrieved_chunks and len(question.strip().split()) <= 4:
            retrieved_chunks = self.vector_store.similarity_search(
                query_embedding=query_emb,
                top_k=top_k,
                score_threshold=0.25
            )

        response = self.generator.generate(query=resolved_q, retrieved_chunks=retrieved_chunks)
        response["question"] = question
        response["resolved_query"] = resolved_q
        response["retrieved_count"] = len(retrieved_chunks)
        return response

    def reset(self):
        """Resets the vector index."""
        self.vector_store.reset()
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)

    def remove_document(self, document_name: str) -> bool:
        """Removes chunks for a document from FAISS index and persists changes."""
        removed = self.vector_store.remove_document(
            document_name=document_name,
            embedding_generator=self.embedder.encode_batch
        )
        if removed:
            self.vector_store.save(self.index_path, self.metadata_path)
        return removed

if __name__ == "__main__":
    pipeline = RAGPipeline()
    print("RAG Pipeline initialized. Total chunks in index:", pipeline.vector_store.total_vectors)
