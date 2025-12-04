import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4.1-mini")
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "google/embeddinggemma-300m")
    HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "google/gemma-3-1b-it")
    EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", 768))
    TIKTOKEN_ENCODING_NAME = os.getenv("TIKTOKEN_ENCODING_NAME", "o200k_base")
    POSTGRES_DB_URL = os.getenv("POSTGRES_DB_URL")
