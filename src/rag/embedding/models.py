from typing import List
import asyncio
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from src.rag.core.interfaces import EmbeddingModel
from config.settings import Config


class OpenAIEmbedder(EmbeddingModel):
    def __init__(
        self,
        model_name: str = Config.OPENAI_EMBEDDING_MODEL,
        dimensions: int = Config.EMBEDDING_DIMENSIONS,
    ):
        super().__init__(model_name, dimensions)
        self.client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        texts = [t.replace("\n", " ") for t in texts]
        response = await self.client.embeddings.create(input=texts, model=self.model_name, dimensions=self.dimensions)
        return [data.embedding for data in response.data]


class HuggingFaceEmbedder(EmbeddingModel):
    def __init__(
        self,
        model_name: str = Config.HF_EMBEDDING_MODEL,
        dimensions: int = Config.EMBEDDING_DIMENSIONS,
    ):
        super().__init__(model_name, dimensions)
        # Initialize synchronously as SentenceTransformer is not async native
        self.model = SentenceTransformer(model_name)

    async def embed(self, texts: List[str]) -> List[List[float]]:
        # Run blocking code in a thread
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.model.encode(texts, convert_to_numpy=True)
        )
        return embeddings.tolist()
