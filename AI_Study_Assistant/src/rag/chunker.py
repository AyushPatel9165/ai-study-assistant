import re
from typing import List, Dict, Any

class DocumentChunk:
    def __init__(
        self,
        chunk_id: str,
        text: str,
        document_name: str,
        page_number: int,
        chunk_index: int
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.document_name = document_name
        self.page_number = page_number
        self.chunk_index = chunk_index

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index
        }

class TextChunker:
    """
    Sliding window chunker that splits document text into manageable segments
    with overlap, while preserving sentence boundaries where possible.
    """
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 60):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, document_name: str, page_number: int) -> List[DocumentChunk]:
        clean_text = re.sub(r"\s+", " ", text).strip()
        if not clean_text:
            return []

        chunks: List[DocumentChunk] = []
        start = 0
        text_len = len(clean_text)
        chunk_idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # If not at the end of the text, try to find a natural sentence or word boundary
            if end < text_len:
                # Look backwards for sentence end (period, newline, question mark)
                split_at = -1
                for punct in [". ", "? ", "! ", "\n"]:
                    pos = clean_text.rfind(punct, start + int(self.chunk_size * 0.5), end)
                    if pos != -1 and pos > split_at:
                        split_at = pos + len(punct)

                if split_at != -1:
                    end = split_at
                else:
                    # Look backwards for space to avoid cutting words in half
                    space_pos = clean_text.rfind(" ", start + int(self.chunk_size * 0.5), end)
                    if space_pos != -1:
                        end = space_pos

            chunk_content = clean_text[start:end].strip()
            if chunk_content:
                chunk_id = f"{document_name}_p{page_number}_c{chunk_idx}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_content,
                    document_name=document_name,
                    page_number=page_number,
                    chunk_index=chunk_idx
                ))
                chunk_idx += 1

            if end >= text_len:
                break
            start = max(start + 1, end - self.chunk_overlap)

        return chunks
