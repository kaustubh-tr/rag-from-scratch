import os
from typing import List
from pypdf import PdfReader
from src.rag.core.interfaces import Document


def load_text(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    file_stats = os.stat(path)
    metadata = {
        "file_name": os.path.basename(path),
        "file_type": "text/plain",
        "file_size": file_stats.st_size,  # in bytes
    }
    return [Document(content=content, metadata=metadata)]


def load_pdf(path: str) -> List[Document]:
    reader = PdfReader(path)
    documents = []
    file_stats = os.stat(path)
    base_metadata = {
        "file_name": os.path.basename(path),
        "file_type": "application/pdf",
        "file_size": file_stats.st_size,  # in bytes
        "author": (
            reader.metadata.get("/Author", "Unknown") if reader.metadata else "Unknown"
        ),
    }

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            page_metadata = base_metadata.copy()
            page_metadata["page_number"] = i + 1
            documents.append(Document(content=text, metadata=page_metadata))

    return documents


def load_document(path: str) -> List[Document]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return load_text(path)
    elif ext == ".pdf":
        return load_pdf(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
