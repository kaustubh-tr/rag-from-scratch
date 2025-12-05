from typing import List, Dict, Any
from core.interfaces import VectorStore, EmbeddingModel


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: EmbeddingModel):
        self.vector_store = vector_store
        self.embedder = embedder

    async def retrieve(
        self, query: str, k: int = 5, filters: Dict[str, Any] = None
    ) -> List[str]:
        query_embedding = (await self.embedder.embed([query]))[0]
        results = await self.vector_store.search(query_embedding, k=k, filters=filters)
        return results
