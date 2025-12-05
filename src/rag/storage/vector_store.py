import asyncpg
from typing import List, Dict, Any
import json
from src.rag.core.interfaces import VectorStore
from config.settings import Config


class PostgresVectorStore(VectorStore):
    def __init__(self, db_url: str = Config.POSTGRES_DB_URL, dimension: int = Config.EMBEDDING_DIMENSIONS):
        if not db_url:
            raise ValueError("Database URL is not configured.")
        self.db_url = db_url
        self.pool = None
        self.dimension = dimension

    async def connect(self):
        if not self.pool:
            print("Creating asyncpg connection pool...")
            self.pool = await asyncpg.create_pool(self.db_url)
            await self._init_db()

    async def close(self):
        if self.pool:
            await self.pool.close()
            self.pool = None
            print("Connection pool closed.")

    async def _init_db(self):
        async with self.pool.acquire() as conn:
            # Enable extensions
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

            # Create tables
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    source_path TEXT NOT NULL,
                    title TEXT,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    deleted_at TIMESTAMPTZ
                );
            """
            )

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    deleted_at TIMESTAMPTZ
                );
            """
            )

            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    embedding vector({self.dimension}),
                    model TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    deleted_at TIMESTAMPTZ
                );
            """
            )

            # Trigger function
            await conn.execute("""
                CREATE OR REPLACE FUNCTION set_updated_at()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = NOW();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """
            )

            # Attach triggers
            await conn.execute("""
                DROP TRIGGER IF EXISTS documents_updated_at ON documents;
                CREATE TRIGGER documents_updated_at
                BEFORE UPDATE ON documents
                FOR EACH ROW
                EXECUTE FUNCTION set_updated_at();
            """
            )
            await conn.execute("""
                DROP TRIGGER IF EXISTS chunks_updated_at ON chunks;
                CREATE TRIGGER chunks_updated_at
                BEFORE UPDATE ON chunks
                FOR EACH ROW
                EXECUTE FUNCTION set_updated_at();
            """
            )
            await conn.execute("""
                DROP TRIGGER IF EXISTS embeddings_updated_at ON embeddings;
                CREATE TRIGGER embeddings_updated_at
                BEFORE UPDATE ON embeddings
                FOR EACH ROW
                EXECUTE FUNCTION set_updated_at();
            """
            )

            # Create Index
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw 
                ON embeddings 
                USING hnsw (embedding vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64);
            """
            )

    async def add(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        source: str,
        model_name: str,
        metadatas: List[Dict[str, Any]] = None,
    ) -> None:
        if not self.pool:
            await self.connect()

        if metadatas is None:
            metadatas = [{} for _ in documents]

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # 1. Insert Document
                # Extract document-level metadata from the first chunk's metadata.
                representative_meta = metadatas[0] if metadatas else {}

                # Define keys that belong to the document level
                doc_keys = {
                    "file_name",
                    "file_type",
                    "file_size",
                    "author",
                    "tags",
                    "ingestion_job_id",
                }

                doc_metadata = {
                    k: v for k, v in representative_meta.items() if k in doc_keys
                }

                doc_id = await conn.fetchval(
                    "INSERT INTO documents (source_path, metadata) VALUES ($1, $2) RETURNING id",
                    source,
                    json.dumps(doc_metadata),
                )

                # 2. Insert Chunks and Embeddings
                for i, (doc_content, full_meta, emb) in enumerate(
                    zip(documents, metadatas, embeddings)
                ):
                    # Filter out document-level keys for the chunk metadata
                    chunk_metadata = {
                        k: v for k, v in full_meta.items() if k not in doc_keys
                    }

                    chunk_id = await conn.fetchval("""
                        INSERT INTO chunks (document_id, chunk_index, content, metadata)
                        VALUES ($1, $2, $3, $4) RETURNING id
                    """,
                        doc_id,
                        i,
                        doc_content,
                        json.dumps(chunk_metadata),
                    )

                    emb_str = "[" + ",".join(f"{x:.8f}" for x in emb) + "]"
                    await conn.execute(
                        "INSERT INTO embeddings (chunk_id, embedding, model) VALUES ($1, $2, $3)",
                        chunk_id,
                        emb_str,
                        model_name,
                    )

    async def search(
        self, query_embedding: List[float], k: int = 5, filters: Dict[str, Any] = None
    ) -> List[str]:
        if not self.pool:
            await self.connect()
        
        async with self.pool.acquire() as conn:
            await conn.execute("SET hnsw.ef_search = 40")

            search_query = """
                SELECT c.content 
                FROM embeddings e
                JOIN chunks c ON e.chunk_id = c.id
                JOIN documents d ON c.document_id = d.id
            """

            params = [str(query_embedding), k]
            where_clauses = []
            param_index = 3

            if filters:
                for key, value in filters.items():
                    if key == "source_path":
                        where_clauses.append(f"d.source_path = ${param_index}")
                        params.append(value)
                        param_index += 1
                    else:
                        where_clauses.append(f"d.metadata->>'{key}' = ${param_index}")
                        params.append(str(value))
                        param_index += 1

            if where_clauses:
                search_query += " WHERE " + " AND ".join(where_clauses)

            search_query += " ORDER BY e.embedding <=> $1 LIMIT $2"

            rows = await conn.fetch(search_query, *params)
            return [row["content"] for row in rows]
