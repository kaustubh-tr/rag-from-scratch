from typing import List, Dict, Any
from core.interfaces import ChunkingStrategy, EmbeddingModel, VectorStore, BaseLLM
from ingestion.loaders import load_document
from retrieval.search import Retriever


class RAGPipeline:
    def __init__(
        self,
        chunker: ChunkingStrategy,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        llm: BaseLLM,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm
        self.retriever = Retriever(vector_store, embedder)

    async def ingest(self, file_path: str):
        import uuid

        job_id = str(uuid.uuid4())
        print(f"Starting ingestion job {job_id} for {file_path}...")

        documents = load_document(file_path)

        # Add job_id to document metadata
        for doc in documents:
            doc.metadata["ingestion_job_id"] = job_id

        print("Chunking...")
        all_chunks = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks)

        print(f"Embedding {len(all_chunks)} chunks...")
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = await self.embedder.embed(chunk_texts)

        print("Storing in Vector Store...")
        # Extract metadata from chunks
        metadatas = [chunk.metadata for chunk in all_chunks]

        # Get model name from embedder
        model_name = getattr(self.embedder, "model_name", "unknown")

        await self.vector_store.add(
            embeddings,
            chunk_texts,
            source=file_path,
            model_name=model_name,
            metadatas=metadatas,
        )
        print("Ingestion complete.")

    async def query(self, user_query: str, filters: Dict[str, Any] = None) -> str:
        print(f"Querying: {user_query}")

        print("Retrieving...")
        results = await self.retriever.retrieve(user_query, k=3, filters=filters)
        context = "\n\n".join(results)

        print("Generating Answer...")
        answer = await self.llm.generate(user_query, context)
        return answer
