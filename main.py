import asyncio
import os
import argparse
from config import Config
from ingestion.chunkers import CharacterChunker, TokenChunker
from embedding.models import OpenAIEmbedder, HuggingFaceEmbedder
from storage.vector_store import PostgresVectorStore
from generation.llm import OpenAILLM, HuggingFaceLLM
from pipeline import RAGPipeline


async def main():
    parser = argparse.ArgumentParser(description="RAG System CLI")
    parser.add_argument("--ingest", type=str, help="Path to the file to ingest")
    parser.add_argument("--query", type=str, help="Question to ask the system")
    parser.add_argument(
        "--embedding-provider",
        type=str,
        choices=["huggingface", "openai"],
        default="openai",
        help="Source for the embedding model (huggingface or openai). Defaults to 'openai'.",
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["huggingface", "openai"],
        default="openai",
        help="Source for the LLM (huggingface or openai). Defaults to 'openai'.",
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        choices=["character", "token"],
        default="character",
        help="Strategy for splitting text into chunks (character or token). Defaults to 'character'.",
    )

    args = parser.parse_args()

    # Initialize embedder
    if args.embedding_provider == "openai":
        embedder = OpenAIEmbedder()
        print("Using OpenAI Embedder.")
    else:
        embedder = HuggingFaceEmbedder()
        print("Using HuggingFace Embedder.")

    # Initialize LLM
    if args.llm_provider == "openai":
        llm = OpenAILLM()
        print("Using OpenAI LLM.")
    else:
        llm = HuggingFaceLLM()
        print("Using HuggingFace LLM.")

    # Initialize Chunker
    if args.chunking_strategy == "token":
        chunker = TokenChunker()
        print("Using TokenChunker.")
    else:
        chunker = CharacterChunker()
        print("Using CharacterChunker.")

    vector_store = PostgresVectorStore()

    pipeline = RAGPipeline(chunker, embedder, vector_store, llm)

    try:
        if args.ingest:
            if os.path.exists(args.ingest):
                print(f"Ingesting file: {args.ingest}")
                await pipeline.ingest(args.ingest)
            else:
                print(f"Error: File '{args.ingest}' not found.")
        elif args.query:
            print(f"Querying system: {args.query}")
            response = await pipeline.query(args.query)
            print(f"\nAnswer:\n{response}")
        else:
            parser.print_help()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await vector_store.close()


if __name__ == "__main__":
    asyncio.run(main())
