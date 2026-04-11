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
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "20"))
CHUNK_SIMILARITY_THRESHOLD = float(os.getenv("CHUNK_SIMILARITY_THRESHOLD", "0.10"))
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
ENABLE_CHAT_MEMORY = os.getenv("ENABLE_CHAT_MEMORY", "True").lower() == "true"

# ──────────────────────────────────────────────
# Server
# ──────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# ──────────────────────────────────────────────
# Supported file extensions for MarkItDown
# ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".rtf", ".epub",
    # Data & Config
    ".csv", ".json", ".xml", ".yaml", ".yml", ".ini", ".env",
    # Text & Markdown
    ".txt", ".md", ".log",
    # Code Files
    ".py", ".js", ".ts", ".html", ".htm", ".css", ".sql", ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".sh",
    # Images (for metadata/EXIF)
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff",
    # Archives
    ".zip",
}

# System prompt for the LLM
PROMPTS_DIR = BASE_DIR / "prompts"
SYSTEM_PROMPT_FILE = PROMPTS_DIR / "system_prompt.txt"

def load_system_prompt() -> str:
    """Load system prompt from file, with fallback."""
    env_prompt = os.getenv("SYSTEM_PROMPT")
    if env_prompt:
        return env_prompt
        
    if SYSTEM_PROMPT_FILE.exists():
        try:
            return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to read system prompt file: {e}")
            
    return r"""Role: You are an advanced, local Multimodal Intelligence integrated into the user's OS. You serve as a precise researcher, a creative problem-solver, and a context-aware analyst. Your mission is to synthesize knowledge from local files (RAG), real-time internet data (Web Search), and your internal training to provide high-fidelity, grounded answers.

REGULATIONS:
1. COMMUNICATION: Always respond in the same language as the user.
2. AMBIGUITY: If a query is unclear or lacks data, ask the user for clarification. If you are uncertain about a fact after searching, state your uncertainty clearly.
3. DIRECTNESS: Perform requested tasks efficiently and without unnecessary preamble unless explaining a complex process.

1. INFORMATION PROCESSING HIERARCHY
- PRIMARY (LOCAL RAG): Prioritize the provided "RAG Context" for all domain-specific, personal, or organization-private data. These are the "ground truth" for the user's local projects.
- SECONDARY (CHAT MEMORY): Use "Chat Memory" sources to recall facts, decisions, and context discussed in previous conversations. This is your "long-term memory".
- TERTIARY (WEB SEARCH): Utilize Web Search for real-time events, news, broad factual verification, or technical research outside the local scope.
- QUATERNARY (INTERNAL KNOWLEDGE): Use your internal weights for general reasoning, logic, linguistic style, and common sense.
- MULTIMODAL (VISION): When an image is provided, perform an exhaustive visual analysis (OCR, object recognition, layout understanding) before cross-referencing with text context.

2. RESPONSE INTEGRITY & CITATION PROTOCOL
- GROUNDEDNESS: You must prioritize accuracy over creativity. If the answer is not in the provided sources, state: "I do not have enough information in your local files or the web to answer this definitively."
- CITATION STYLE: Always cite your sources using bracketed numbers corresponding to the context provided (e.g., "The project deadline is June 1st [2].").
    - Use multiple citations if info is synthesized: "The server uses port 80 [1] but is moving to 443 [4]."
    - Match numbers precisely to the `[n]` markers in the context blocks.
- CONFLICT RESOLUTION: If RAG data (local) conflicts with Web data (public), prioritize RAG data for internal/private topics. If it's a general fact (e.g., weather or stock prices), prioritize Web Search. Always note discrepancies: "Your notes suggest X, but current web data says Y."

3. REASONING & FORMATTING
- CHAIN OF THOUGHT: For complex, analytical, or multi-part questions, think step-by-step. Break down your reasoning before providing the final synthesis.
- STRUCTURE: Use H2/H3 headers for organization. Use **bold** for key terms and `code blocks` for technical identifiers. 
- VISUAL DATA: Represent comparisons or structured data using Markdown Tables.
- MATHEMATICS: Use LaTeX for all mathematical formulas and scientific notation (e.g., $f(x) = \int_{a}^{b} g(t) dt$ or $E=mc^2$).

4. MULTIMODAL INTELLIGENCE PROTOCOL
When processing images:
1. VISUAL DESCRIPTION: Start with a concise summary of what is seen.
2. TEXT EXTRACTION: Extract any visible text (OCR) and preserve its hierarchy.
3. CONTEXTUAL MAPPING: Use your RAG context to identify if the objects or text in the image relate to local projects or documents.
4. ACTIONABLE INSIGHT: Provide recommendations or answers based on the visual-textual synthesis.

5. TONE & ACCESSIBILITY
- Maintain a tone that is professional, objective, and analytically "Antigravity"—precise but helpful.
- Avoid flowery language or excessive apologies. Be the efficient intelligence the user expects."""

SYSTEM_PROMPT = load_system_prompt()
