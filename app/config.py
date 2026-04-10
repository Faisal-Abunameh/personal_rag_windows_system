"""
Central configuration for the RAG system.
All tunables are defined here and can be overridden via environment variables.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
UPLOADS_DIR = DATA_DIR / "uploads"
DATABASE_PATH = DATA_DIR / "rag.db"
STATIC_DIR = BASE_DIR / "static"

# References directory on the Desktop
REFERENCES_DIR = Path(
    os.getenv("REFERENCES_DIR", os.path.expanduser("~/Desktop/references"))
)

# ──────────────────────────────────────────────
# LLM (Ollama — supports any model)
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gemma4")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))

# ──────────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────────
# NeMo Retriever NIM endpoint (Docker-based)
NEMO_RETRIEVER_URL = os.getenv("NEMO_RETRIEVER_URL", "http://localhost:8000/v1")
NEMO_RETRIEVER_MODEL = os.getenv(
    "NEMO_RETRIEVER_MODEL", "llama-3.2-nv-embedqa-1b-v2"
)

# Fallback: sentence-transformers model (CPU-friendly)
FALLBACK_EMBEDDING_MODEL = os.getenv(
    "FALLBACK_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

# Embedding dimension (384 for MiniLM, 768 for NeMo — auto-detected)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "0"))  # 0 = auto-detect

# ──────────────────────────────────────────────
# RAG Pipeline
# ──────────────────────────────────────────────
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "10"))
CHUNK_SIMILARITY_THRESHOLD = float(os.getenv("CHUNK_SIMILARITY_THRESHOLD", "0.65"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))  # max chars per chunk
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "100"))   # min chars per chunk
CHUNK_OVERLAP_SENTENCES = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "1"))

# ──────────────────────────────────────────────
# Conversation
# ──────────────────────────────────────────────
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))
CONVERSATION_TITLE_MAX_LEN = int(os.getenv("CONVERSATION_TITLE_MAX_LEN", "60"))

# ──────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────
EMBEDDING_CACHE_SIZE = int(os.getenv("EMBEDDING_CACHE_SIZE", "2048"))
QUERY_CACHE_SIZE = int(os.getenv("QUERY_CACHE_SIZE", "256"))
QUERY_CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", "300"))  # seconds

# ──────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ──────────────────────────────────────────────
# Supported file extensions for MarkItDown
# ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".html", ".htm", ".csv", ".json", ".xml", ".txt", ".md",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff",
    ".zip", ".epub",
}

# System prompt for the LLM
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """You are a helpful, knowledgeable AI assistant with access to a local knowledge base.
When answering questions, use the provided context from retrieved documents to give accurate, well-sourced answers.
If the context contains relevant information, cite the source document.
If you don't know the answer or the context doesn't contain relevant information, say so honestly.
Be concise but thorough. Use markdown formatting for readability.""")
