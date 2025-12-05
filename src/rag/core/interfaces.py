import tiktoken
from abc import ABC, abstractmethod
from typing import List, Any, Dict
from dataclasses import dataclass, field


@dataclass
class Document:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChunkingStrategy(ABC):
    def __init__(self, chunk_size: int, chunk_overlap: int, encoding_name: str):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)

    @abstractmethod
    def chunk(self, document: Document) -> List[Document]:
        """Split document into chunks, preserving metadata."""
        pass


class EmbeddingModel(ABC):
    def __init__(self, model_name: str, dimensions: int = None):
        self.model_name = model_name
        self.dimensions = dimensions

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass


class VectorStore(ABC):
    @abstractmethod
    async def add(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        source: str,
        model_name: str,
        metadatas: List[dict] = None,
    ) -> None:
        """Add embeddings and documents to the store."""
        pass

    @abstractmethod
    async def search(
        self, query_embedding: List[float], k: int, filters: dict = None
    ) -> List[str]:
        """Search for similar documents with optional filters."""
        pass


class BaseLLM(ABC):
    def __init__(self, model_name: str):
        self.system_prompt = "You are an assistant for question-answering tasks."
        self.user_prompt = (
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know.\n"
            "ONLY output the final answer. Do NOT repeat the question, the context, "
            "or any part of this instruction.\n"
            "Question: {query}\n"
            "Context: {context}\n"
            "Answer:"
        )
        self.model_name = model_name

    @abstractmethod
    async def generate(self, query: str, context: str = "") -> str:
        """Generate a response based on query and optional context."""
        pass
