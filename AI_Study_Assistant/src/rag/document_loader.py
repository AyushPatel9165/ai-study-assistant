import os
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

class LoadedPage:
    def __init__(self, text: str, page_number: int, document_name: str, file_type: str):
        self.text = text
        self.page_number = page_number
        self.document_name = document_name
        self.file_type = file_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "page_number": self.page_number,
            "document_name": self.document_name,
            "file_type": self.file_type
        }

class DocumentLoader:
    """
    Extracts text from uploaded PDF and TXT study notes with page-level metadata.
    """
    SUPPORTED_EXTENSIONS = [".pdf", ".txt"]

    @classmethod
    def load_document(cls, file_path: str) -> List[LoadedPage]:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Document not found at: {file_path}")

        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{ext}'. Supported types: {cls.SUPPORTED_EXTENSIONS}")

        if ext == ".txt":
            return cls._load_txt(path)
        elif ext == ".pdf":
            return cls._load_pdf(path)
        else:
            raise ValueError(f"Unknown extension: {ext}")

    @classmethod
    def _load_txt(cls, path: Path) -> List[LoadedPage]:
        encodings = ["utf-8", "latin-1", "ascii"]
        text = ""
        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    text = f.read()
                break
            except (UnicodeDecodeError, Exception):
                continue

        if not text.strip():
            raise ValueError(f"File {path.name} is empty or unreadable.")

        return [LoadedPage(
            text=text.strip(),
            page_number=1,
            document_name=path.name,
            file_type="txt"
        )]

    @classmethod
    def _load_pdf(cls, path: Path) -> List[LoadedPage]:
        reader = PdfReader(str(path))
        if len(reader.pages) == 0:
            raise ValueError(f"PDF {path.name} contains no pages.")

        pages: List[LoadedPage] = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(LoadedPage(
                    text=page_text.strip(),
                    page_number=i + 1,
                    document_name=path.name,
                    file_type="pdf"
                ))

        if not pages:
            raise ValueError(f"No readable text could be extracted from PDF: {path.name}")
        return pages
