"""
Central configuration for the RAG system.
All tunables are defined here and can be overridden via environment variables.
"""

import os
import logging
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
# Primary: any Ollama model via /api/embed (auto-detected or user-selected)
# Fallback: sentence-transformers (local, runs on CPU or GPU)
FALLBACK_EMBEDDING_MODEL = os.getenv(
    "FALLBACK_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)

# Embedding dimension (384 for MiniLM, 768 for NeMo — auto-detected)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "0"))  # 0 = auto-detect

# ──────────────────────────────────────────────
# Device (GPU / CPU)
# ──────────────────────────────────────────────
# Options: "auto" (detect GPU, fallback CPU), "cuda", "cpu"
_DEVICE_SETTING = os.getenv("DEVICE", "auto").lower()

def resolve_device() -> str:
    """Resolve the compute device: auto-detect CUDA GPU, fallback to CPU."""
    _logger = logging.getLogger(__name__)
    if _DEVICE_SETTING == "cpu":
        _logger.info("Device forced to CPU via config")
        return "cpu"
    if _DEVICE_SETTING == "cuda":
        _logger.info("Device forced to CUDA via config")
        return "cuda"
    # Auto-detect by actually trying to use CUDA
    try:
        import torch
        # torch.cuda.is_available() can return True even for CPU-only builds,
        # so we verify by actually allocating a tensor on CUDA
        _ = torch.zeros(1, device="cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        _logger.info(f"GPU detected: {gpu_name} ({vram:.1f} GB VRAM) — using CUDA")
        return "cuda"
    except Exception:
        _logger.info("CUDA not available — using CPU")
        return "cpu"

DEVICE = resolve_device()

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
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", """You are a helpful, knowledgeable AI assistant with access to both a local knowledge base and live web search results.
When answering questions, use the provided context to give accurate, well-sourced answers.
If the context comes from a local document, cite the source filename.
If the context comes from a LIVE WEB SEARCH RESULT, cite the URL or source name provided.
If you don't know the answer or the context doesn't contain relevant information, say so honestly.
Be concise but thorough. Use markdown formatting for readability.""")
