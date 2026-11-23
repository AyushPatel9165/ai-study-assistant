import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.document_loader import DocumentLoader, LoadedPage
from src.rag.chunker import TextChunker, DocumentChunk
from src.rag.vector_store import FAISSVectorStore
from src.rag.generator import RAGGenerator
from src.rag.pipeline import RAGPipeline

@pytest.fixture
def sample_notes_path(tmp_path):
    note_file = tmp_path / "test_notes.txt"
    content = (
        "Gradient descent is an iterative optimization algorithm used to minimize loss functions. "
        "The learning rate determines the step size at each iteration. "
        "Overfitting happens when a model memorizes noise instead of learning general patterns. "
        "L2 regularization penalizes large weights by adding squared weight penalties."
    )
    note_file.write_text(content, encoding="utf-8")
    return str(note_file)

def test_document_loader_txt(sample_notes_path):
    pages = DocumentLoader.load_document(sample_notes_path)
    assert len(pages) == 1
    assert "Gradient descent" in pages[0].text
    assert pages[0].file_type == "txt"
    assert pages[0].page_number == 1

def test_document_loader_unsupported_extension(tmp_path):
    invalid_file = tmp_path / "notes.unsupported"
    invalid_file.write_text("Hello world", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        DocumentLoader.load_document(str(invalid_file))
    assert "Unsupported file type" in str(exc.value)

def test_chunker_sliding_window():
    chunker = TextChunker(chunk_size=120, chunk_overlap=30)
    text = (
        "Python generators yield items lazily one at a time. "
        "They preserve execution state on the frame stack. "
        "This makes them ideal for processing large streams of data efficiently."
    )
    chunks = chunker.chunk_text(text, document_name="python.txt", page_number=1)
    assert len(chunks) >= 2
    for c in chunks:
        assert isinstance(c, DocumentChunk)
        assert len(c.text) > 0
        assert c.document_name == "python.txt"
        assert c.page_number == 1

def test_vector_store_faiss():
    store = FAISSVectorStore(dimension=4)
    chunks = [
        DocumentChunk("c1", "Text about algorithms", "ai.txt", 1, 0),
        DocumentChunk("c2", "Text about decorators", "py.txt", 1, 1),
    ]
    # 2 normalized vectors
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ], dtype=np.float32)

    store.add_chunks(chunks, embeddings)
    assert store.total_vectors == 2

    # Query matching first vector
    q_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    results = store.similarity_search(q_vec, top_k=2, score_threshold=0.5)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["score"] > 0.99

def test_rag_generator_groundedness():
    generator = RAGGenerator(confidence_threshold=0.35)

    # 1. When chunks are provided
    chunks = [{
        "chunk_id": "test_c1",
        "document_name": "ml.txt",
        "page_number": 1,
        "score": 0.85,
        "text": "Regularization prevents overfitting by penalizing high weights."
    }]
    res = generator.generate("How does regularization work?", chunks)
    assert res["found_in_notes"] is True
    assert "Based on your uploaded study notes" in res["answer"]
    assert len(res["citations"]) == 1
    assert res["citations"][0]["document_name"] == "ml.txt"

    # 2. When no chunks meet threshold (anti-hallucination guardrail)
    res_empty = generator.generate("What is quantum computing?", [])
    assert res_empty["found_in_notes"] is False
    assert "could not find relevant information" in res_empty["answer"]
    assert len(res_empty["citations"]) == 0

def test_rag_pipeline_end_to_end(sample_notes_path, tmp_path):
    idx_file = str(tmp_path / "temp_faiss.bin")
    meta_file = str(tmp_path / "temp_meta.json")

    pipeline = RAGPipeline(
        index_path=idx_file,
        metadata_path=meta_file,
        chunk_size=150,
        chunk_overlap=30
    )

    # Index document
    stats = pipeline.index_document(sample_notes_path)
    assert stats["pages_processed"] == 1
    assert stats["chunks_created"] > 0

    # Query relevant concept
    ans = pipeline.query("What does L2 regularization do?")
    assert ans["found_in_notes"] is True
    assert len(ans["citations"]) > 0
    assert ans["citations"][0]["document_name"] == "test_notes.txt"

    # Query completely unrelated concept (must reject)
    ans_irrelevant = pipeline.query("Who is the president of France in 1900?")
    assert ans_irrelevant["found_in_notes"] is False
    assert "could not find relevant information" in ans_irrelevant["answer"]
