import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss

from src.rag.chunker import DocumentChunk

class FAISSVectorStore:
    """
    Vector database using FAISS IndexFlatIP (Inner Product on L2-normalized embeddings)
    for exact cosine similarity search over study notes.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[DocumentChunk] = []

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """
        Adds text chunks and their normalized embedding vectors into the FAISS index.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings.")

        if len(chunks) == 0:
            return

        # Ensure float32 format and 2D shape
        emb_arr = np.array(embeddings, dtype=np.float32)
        if emb_arr.ndim == 1:
            emb_arr = np.expand_dims(emb_arr, axis=0)

        # Normalize to unit length for cosine similarity via Inner Product
        faiss.normalize_L2(emb_arr)

        self.index.add(emb_arr)
        self.chunks.extend(chunks)

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        score_threshold: float = 0.30
    ) -> List[Dict[str, Any]]:
        """
        Finds the top_k most relevant chunks matching the query embedding.
        Filters out low-confidence matches below score_threshold.
        """
        if self.index.ntotal == 0:
            return []

        q_arr = np.array(query_embedding, dtype=np.float32)
        if q_arr.ndim == 1:
            q_arr = np.expand_dims(q_arr, axis=0)
        faiss.normalize_L2(q_arr)

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_arr, actual_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= score_threshold:
                chunk = self.chunks[idx]
                results.append({
                    "score": round(float(score), 4),
                    "chunk_id": chunk.chunk_id,
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number,
                    "text": chunk.text
                })
        return results

    def save(self, index_path: str, metadata_path: str):
        """Persists the FAISS index and chunk metadata to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        meta_data = [c.to_dict() for c in self.chunks]
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)

    def remove_document(self, document_name: str, embedding_generator=None) -> bool:
        """Removes all chunks belonging to a document and rebuilds the FAISS index."""
        doc_lower = document_name.lower()
        retained = [c for c in self.chunks if c.document_name.lower() != doc_lower]
        if len(retained) == len(self.chunks):
            return False

        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks = []

        if retained:
            if embedding_generator:
                texts = [c.text for c in retained]
                embs = embedding_generator(texts)
                self.add_chunks(retained, embs)
            else:
                self.chunks = retained
        return True

    def load(self, index_path: str, metadata_path: str):
        """Loads FAISS index and chunk metadata from disk."""
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Index or metadata file does not exist.")

        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_meta = json.load(f)
        self.chunks = [
            DocumentChunk(
                chunk_id=m["chunk_id"],
                text=m["text"],
                document_name=m["document_name"],
                page_number=m["page_number"],
                chunk_index=m["chunk_index"]
            )
            for m in raw_meta
        ]

    def reset(self):
        """Clears all vectors and chunks."""
        self.index.reset()
        self.chunks.clear()
