import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.document_loader import DocumentLoader
from src.rag.chunker import TextChunker
from src.rag.vector_store import FAISSVectorStore
from src.nlp.transformer_pipeline import TransformerQuestionUnderstander
from src.database.db import DatabaseManager

def ingest_file(
    file_path: str,
    index_path: str = "models/faiss_index.bin",
    metadata_path: str = "models/faiss_chunks.json",
    chunk_size: int = 400,
    chunk_overlap: int = 60
) -> dict:
    """
    Ingests, chunks, embeds, and indexes a PDF or TXT study note into FAISS & SQLite.
    Matches the 'PDF Ingestion & Chunking (ingest.py)' workflow in the architecture diagram.
    """
    target = Path(file_path)
    if not target.is_file():
        raise FileNotFoundError(f"Study note file not found: {file_path}")

    filename = target.name
    file_size_bytes = target.stat().st_size
    file_type = target.suffix.replace(".", "").upper()

    print(f"\n[INGEST] Processing file: {filename} ({file_type}, {file_size_bytes} bytes)")

    # 1. Extract Text & Page Metadata
    pages = DocumentLoader.load_document(str(target))
    print(f"  -> Extracted {len(pages)} pages/sections from {filename}")

    # 2. Chunk Text
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for page in pages:
        p_chunks = chunker.chunk_text(
            text=page.text,
            document_name=page.document_name,
            page_number=page.page_number
        )
        all_chunks.extend(p_chunks)

    if not all_chunks:
        raise ValueError(f"No chunks could be extracted from {filename}")

    print(f"  -> Generated {len(all_chunks)} overlapping chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")

    # 3. Dense Embeddings via SentenceTransformer (384-dimensional)
    print("  -> Computing dense vector embeddings with Transformer model...")
    embedder = TransformerQuestionUnderstander()
    chunk_texts = [c.text for c in all_chunks]
    embeddings = embedder.encode_batch(chunk_texts)
    print(f"  -> Generated {len(embeddings)} vectors of dimension {embeddings.shape[1]}")

    # 4. Store in FAISS Vector Store
    vector_store = FAISSVectorStore(dimension=384)
    full_idx_path = PROJECT_ROOT / index_path
    full_meta_path = PROJECT_ROOT / metadata_path

    if full_idx_path.exists() and full_meta_path.exists():
        try:
            vector_store.load(str(full_idx_path), str(full_meta_path))
            print(f"  -> Loaded existing FAISS index with {vector_store.total_vectors} vectors")
        except Exception:
            pass

    # Add new chunks
    vector_store.add_chunks(all_chunks, embeddings)
    vector_store.save(str(full_idx_path), str(full_meta_path))
    print(f"  -> Saved updated FAISS index to {index_path} (Total vectors: {vector_store.total_vectors})")

    # 5. Add Document Metadata to SQLite
    db = DatabaseManager()
    db.add_document(
        filename=filename,
        file_type=file_type,
        file_size_bytes=file_size_bytes,
        chunks_count=len(all_chunks)
    )
    print(f"  -> Saved document metadata to SQLite database (study_assistant.db)")

    return {
        "filename": filename,
        "file_type": file_type,
        "pages_processed": len(pages),
        "chunks_created": len(all_chunks),
        "total_indexed_vectors": vector_store.total_vectors
    }

def main():
    parser = argparse.ArgumentParser(description="Ingest, chunk, embed, and index study notes for RAG.")
    parser.add_argument("file", nargs="?", default=None, help="Path to PDF or TXT file to ingest")
    parser.add_argument("--chunk-size", type=int, default=400, help="Chunk size in characters (default: 400)")
    parser.add_argument("--overlap", type=int, default=60, help="Chunk overlap in characters (default: 60)")
    parser.add_argument("--clean", action="store_true", help="Reset existing index before ingesting")
    args = parser.parse_args()

    full_idx_path = PROJECT_ROOT / "models" / "faiss_index.bin"
    full_meta_path = PROJECT_ROOT / "models" / "faiss_chunks.json"

    if args.clean:
        print("[INGEST] --clean flag passed: Resetting existing FAISS vector index...")
        if full_idx_path.exists():
            full_idx_path.unlink()
        if full_meta_path.exists():
            full_meta_path.unlink()

    if args.file:
        files_to_ingest = [Path(args.file)]
    else:
        # Default scan: data/raw and uploads
        files_to_ingest = []
        raw_dir = PROJECT_ROOT / "data" / "raw"
        uploads_dir = PROJECT_ROOT / "uploads"
        for d in [raw_dir, uploads_dir]:
            if d.exists():
                for f in d.iterdir():
                    if f.suffix.lower() in [".pdf", ".txt"] and f.is_file():
                        files_to_ingest.append(f)

    if not files_to_ingest:
        print("[INGEST] No study note documents found to ingest. Provide a file path or place files in data/raw/ or uploads/")
        return

    print(f"[INGEST] Found {len(files_to_ingest)} document(s) to process.")
    for f in files_to_ingest:
        try:
            res = ingest_file(str(f), chunk_size=args.chunk_size, chunk_overlap=args.overlap)
            print(f"[SUCCESS] Ingested '{res['filename']}': {res['chunks_created']} chunks added. Index total: {res['total_indexed_vectors']} vectors.")
        except Exception as e:
            print(f"[ERROR] Failed to ingest {f.name}: {e}")

if __name__ == "__main__":
    main()
