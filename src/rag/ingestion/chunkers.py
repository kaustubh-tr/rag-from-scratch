from typing import List
import tiktoken
from src.rag.core.interfaces import ChunkingStrategy, Document
from config.settings import Config


class CharacterChunker(ChunkingStrategy):
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        encoding_name: str = Config.TIKTOKEN_ENCODING_NAME,
    ):
        super().__init__(chunk_size, chunk_overlap, encoding_name)

    def chunk(self, document: Document) -> List[Document]:
        text = document.content
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunk_metadata = document.metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_strategy": "character",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "token_count": len(self.encoding.encode(chunk_text)),
                }
            )

            chunks.append(Document(content=chunk_text, metadata=chunk_metadata))
            start += self.chunk_size - self.chunk_overlap

        return chunks


class TokenChunker(ChunkingStrategy):
    def __init__(
        self,
        chunk_size: int = 250,
        chunk_overlap: int = 50,
        encoding_name: str = Config.TIKTOKEN_ENCODING_NAME,
    ):
        super().__init__(chunk_size, chunk_overlap, encoding_name)

    def chunk(self, document: Document) -> List[Document]:
        text = document.content
        tokens = self.encoding.encode(text)
        chunks = []
        start = 0
        tokens_len = len(tokens)

        while start < tokens_len:
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunk_metadata = document.metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_strategy": "token",
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                    "token_count": len(chunk_tokens),
                }
            )

            chunks.append(Document(content=chunk_text, metadata=chunk_metadata))
            start += self.chunk_size - self.chunk_overlap

        return chunks
