# RAG System From Scratch

A modular, asynchronous Retrieval-Augmented Generation (RAG) system built in Python from scratch. This project demonstrates how to build a RAG architecture without relying on high-level frameworks like LangChain or LlamaIndex. It uses PostgreSQL (`pgvector`) for vector storage and supports both OpenAI and Open Source (Hugging Face) models.

## Features

- **Async Architecture**: Built with `asyncio` and `asyncpg` for efficient I/O operations.
- **Modular Design**: Clean separation of concerns (Ingestion, Embedding, Storage, Retrieval, Generation).
- **Hybrid Model Support**:
  - **Closed Source**: OpenAI (GPT-4, text-embedding-3).
  - **Open Source**: Hugging Face (Sentence Transformers, TinyLlama/Phi-3).
- **Vector Store Implementation**:
  - Uses **PostgreSQL** with `pgvector` extension.
  - Normalized 3-table schema (`documents`, `chunks`, `embeddings`).
  - Metadata support (JSONB) for filtering.
- **Flexible Chunking**:
  - Character-based chunking.
  - Token-based chunking (using `tiktoken`).
- **CLI Interface**: Simple command-line interface for ingestion and querying.

## Architecture

The system follows a standard RAG pipeline:

1.  **Ingestion**: Load text or PDF documents.
2.  **Chunking**: Split documents into manageable pieces (preserving metadata like page numbers).
3.  **Embedding**: Convert text chunks into vector representations.
4.  **Storage**: Store vectors and metadata in Postgres.
5.  **Retrieval**: Search for relevant chunks using cosine similarity and metadata filters.
6.  **Generation**: Synthesize an answer using an LLM with the retrieved context.

## Prerequisites

- **Python 3.10+**
- **PostgreSQL** with `pgvector` and `uuid-ossp` extensions enabled.
- **OpenAI API Key** (optional, if using OpenAI models).

## Installation

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/kaustubh-tr/rag-from-scratch.git
    cd rag-from-scratch
    ```

2.  **Create a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up the database**:
    Create a database (replace dbname with your desired name):

    ```sql
    CREATE DATABASE dbname;
    ```

    Ensure your Postgres instance is running. The application will automatically create the necessary tables on the first run, provided the user has permissions to create extensions and tables.

    _Manual Extension Setup (if needed):_

    ```sql
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    ```

## Configuration

Create a `.env` file in the root directory:

```env
# Database
POSTGRES_DB_URL=postgresql://username:password@host:port/dbname

# OpenAI (if using)
OPENAI_API_KEY=sk-...

# Model Selection (openai or huggingface)
EMBEDDING_PROVIDER=openai
LLM_PROVIDER=openai

# Hugging Face Config (optional)
HF_TOKEN=your_hf_token
HF_EMBEDDING_MODEL=google/embeddinggemma-300m
HF_LLM_MODEL=google/gemma-3-1b-it
```

## Usage

The system provides a CLI via `main.py`.

### Command Line Arguments

| Argument               | Description                               | Options                 | Default     |
| :--------------------- | :---------------------------------------- | :---------------------- | :---------- |
| `--ingest`             | Path to the file to ingest (PDF or Text). | File path               | None        |
| `--query`              | Question to ask the system.               | Text string             | None        |
| `--embedding-provider` | Provider for embedding model.             | `openai`, `huggingface` | `openai`    |
| `--llm-provider`       | Provider for LLM.                         | `openai`, `huggingface` | `openai`    |
| `--chunking-strategy`  | Strategy for splitting text.              | `character`, `token`    | `character` |

### Examples

**Ingest a file using Hugging Face embeddings and Token chunking:**

```bash
python -m src.main --ingest data/sample1.txt --embedding-provider huggingface --chunking-strategy token
```

**Query the system using OpenAI LLM:**

```bash
python -m src.main --query "What is the main topic?" --llm-provider openai
```

## Project Structure

```
rag-from-scratch/
├── config/
│   └── settings.py              # Configuration management
├── data/                        # Sample data
│   ├── sample1.txt
│   └── sample2.txt
├── src/
│   ├── main.py                  # CLI Entry point
│   └── rag/
│       ├── pipeline.py          # RAG Orchestrator
│       ├── core/                # Abstract Base Classes & Data Structures
│       │   └── interfaces.py
│       ├── ingestion/           # Data Loading & Chunking
│       │   ├── loaders.py
│       │   └── chunkers.py
│       ├── embedding/           # Embedding Models
│       │   └── models.py
│       ├── storage/             # Vector Database
│       │   └── vector_store.py
│       ├── retrieval/           # Search Logic
│       │   └── search.py
│       └── generation/          # LLM Integration
│           └── llm.py
├── .env.example                 # Environment variables
├── .gitignore                   # Git ignore file
├── README.md                    # Project documentation
└── requirements.txt             # Project dependencies
```

## Database Schema

The system uses a normalized 3-table schema in PostgreSQL.

### 1. `documents`

Stores file-level metadata.

```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_path TEXT NOT NULL,
    title TEXT,
    metadata JSONB, -- Stores file_name, file_size, author, etc.
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);
```

### 2. `chunks`

Stores the split text content and chunk-specific metadata.

```sql
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB, -- Stores page_number, token_count, strategy
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);
```

### 3. `embeddings`

Stores the vector representation of chunks.

```sql
CREATE TABLE embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    embedding vector, -- The actual vector data
    model TEXT,       -- Name of the model used (e.g., text-embedding-3-small)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);
```
