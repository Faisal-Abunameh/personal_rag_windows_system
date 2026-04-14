"""
Central configuration for the RAG system.
All tunables are defined here and can be overridden via settings.json or environment variables.
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (.env acts as a fallback/system-level override)
load_dotenv()

# ────────────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
UPLOADS_DIR = DATA_DIR / "uploads"
DATABASE_PATH = DATA_DIR / "rag.db"
MEM0_DIR = DATA_DIR / "mem0"
STATIC_DIR = BASE_DIR / "static"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".py", ".js", ".html", ".css"}

SETTINGS_FILE = DATA_DIR / "settings.json"

def load_user_settings():
    """Load settings from the JSON file."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load settings.json: {e}")
    return {}

USER_SETTINGS = load_user_settings()

def get_setting(key, default):
    """Priority: settings.json > environment variable > default."""
    if key in USER_SETTINGS:
        return USER_SETTINGS[key]
    val = os.getenv(key)
    if val is not None and val.strip() != "":
        # Simple type inference for env vars
        try:
            if isinstance(default, bool): return val.lower() == "true"
            if isinstance(default, int): return int(val)
            if isinstance(default, float): return float(val)
        except (ValueError, TypeError):
            return default
        return val
    return default

# References directory on the Desktop (Auto-detect & Create)
desktop_path = Path.home() / "Desktop"
REFERENCES_DIR = desktop_path / "references"
try:
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logging.warning(f"Could not create references directory on Desktop: {e}. Falling back to app data.")
    REFERENCES_DIR = DATA_DIR / "references"
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# LLM (Ollama — supports any model)
# ────────────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = get_setting("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = get_setting("LLM_MODEL", "gemma4")
LLM_TEMPERATURE = float(get_setting("LLM_TEMPERATURE", 0.7))
LLM_TOP_P = float(get_setting("LLM_TOP_P", 0.9))
LLM_MAX_TOKENS = int(get_setting("LLM_MAX_TOKENS", 4096))
LLM_CONTEXT_WINDOW = int(get_setting("LLM_CONTEXT_WINDOW", 8192))

# ────────────────────────────────────────────────────────────────────────────
# Embeddings
# ────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = get_setting("EMBEDDING_MODEL", "")
FALLBACK_EMBEDDING_MODEL = get_setting("FALLBACK_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(get_setting("EMBEDDING_DIM", 0))  # 0 = auto-detect

# ────────────────────────────────────────────────────────────────────────────
# Device (GPU / CPU)
# ────────────────────────────────────────────────────────────────────────────
_DEVICE_SETTING = get_setting("DEVICE", "auto").lower()

def resolve_device() -> str:
    """Resolve the compute device: auto-detect CUDA GPU, fallback to CPU."""
    _logger = logging.getLogger(__name__)
    if _DEVICE_SETTING == "cpu":
        return "cpu"
    if _DEVICE_SETTING == "cuda":
        return "cuda"
    try:
        import torch
        _ = torch.zeros(1, device="cuda")
        return "cuda"
    except Exception:
        return "cpu"

DEVICE = resolve_device()

# ────────────────────────────────────────────────────────────────────────────
# RAG Pipeline
# ────────────────────────────────────────────────────────────────────────────
TOP_K_CHUNKS = int(get_setting("TOP_K_CHUNKS", 20))
CHUNK_SIMILARITY_THRESHOLD = float(get_setting("CHUNK_SIMILARITY_THRESHOLD", 0.10))
MAX_CHUNK_SIZE = int(get_setting("MAX_CHUNK_SIZE", 1000))
MIN_CHUNK_SIZE = int(get_setting("MIN_CHUNK_SIZE", 100))
CHUNK_OVERLAP_SENTENCES = int(get_setting("CHUNK_OVERLAP_SENTENCES", 1))

# ────────────────────────────────────────────────────────────────────────────
# Conversation
# ────────────────────────────────────────────────────────────────────────────
MAX_CONVERSATION_HISTORY = int(get_setting("MAX_CONVERSATION_HISTORY", 20))
CONVERSATION_TITLE_MAX_LEN = int(get_setting("CONVERSATION_TITLE_MAX_LEN", 60))
ENABLE_CHAT_MEMORY = get_setting("ENABLE_CHAT_MEMORY", True)
ENABLE_MEM0 = get_setting("ENABLE_MEM0", True)

# ────────────────────────────────────────────────────────────────────────────
# Cache
# ────────────────────────────────────────────────────────────────────────────
EMBEDDING_CACHE_SIZE = int(get_setting("EMBEDDING_CACHE_SIZE", 1024))
QUERY_CACHE_SIZE = int(get_setting("QUERY_CACHE_SIZE", 128))
QUERY_CACHE_TTL = int(get_setting("QUERY_CACHE_TTL", 3600))

# ────────────────────────────────────────────────────────────────────────────
# Configuration Metadata (for UI)
# ────────────────────────────────────────────────────────────────────────────
CONFIG_METADATA = {
    "LLM_MODEL": {
        "type": "select",
        "description": "The active LLM model from Ollama.",
        "category": "Model Settings",
        "options": []
    },
    "EMBEDDING_MODEL": {
        "type": "select",
        "description": "Model used for document vector indexing.",
        "category": "Model Settings",
        "options": []
    },
    "LLM_TEMPERATURE": {
        "type": "float",
        "min": 0,
        "max": 2.0,
        "step": 0.05,
        "recommended": 0.7,
        "description": "Controls creativity. Higher is more creative, lower is more focused.",
        "category": "Model Settings"
    },
    "LLM_TOP_P": {
        "type": "float",
        "min": 0,
        "max": 1.0,
        "step": 0.05,
        "recommended": 0.9,
        "description": "Nucleus sampling threshold.",
        "category": "Model Settings"
    },
    "LLM_MAX_TOKENS": {
        "type": "int",
        "min": 256,
        "max": 8192,
        "step": 256,
        "recommended": 4096,
        "description": "Maximum tokens in the response.",
        "category": "Model Settings"
    },
    "LLM_CONTEXT_WINDOW": {
        "type": "int",
        "min": 1024,
        "max": 128000,
        "step": 1024,
        "recommended": 8192,
        "description": "Total context window size (tokens).",
        "category": "Model Settings"
    },
    "TOP_K_CHUNKS": {
        "type": "int",
        "min": 1,
        "max": 100,
        "step": 1,
        "recommended": 20,
        "description": "Number of document chunks to retrieve.",
        "category": "RAG Settings"
    },
    "CHUNK_SIMILARITY_THRESHOLD": {
        "type": "float",
        "min": 0,
        "max": 1,
        "step": 0.01,
        "recommended": 0.10,
        "description": "Minimum similarity score for document retrieval.",
        "category": "RAG Settings"
    },
    "MAX_CHUNK_SIZE": {
        "type": "int",
        "min": 200,
        "max": 10000,
        "step": 100,
        "recommended": 1000,
        "description": "Maximum characters per document chunk.",
        "category": "RAG Settings"
    },
    "MIN_CHUNK_SIZE": {
        "type": "int",
        "min": 0,
        "max": 5000,
        "step": 50,
        "recommended": 100,
        "description": "Minimum characters per document chunk.",
        "category": "RAG Settings"
    },
    "CHUNK_OVERLAP_SENTENCES": {
        "type": "int",
        "min": 0,
        "max": 20,
        "step": 1,
        "recommended": 1,
        "description": "Number of sentences to overlap between chunks.",
        "category": "RAG Settings"
    },
    "MAX_CONVERSATION_HISTORY": {
        "type": "int",
        "min": 0,
        "max": 200,
        "step": 1,
        "recommended": 20,
        "description": "Number of past messages to send to the LLM.",
        "category": "Conversation"
    },
    "ENABLE_CHAT_MEMORY": {
        "type": "bool",
        "recommended": True,
        "description": "Store and retrieve past conversations using basic RAG.",
        "category": "Conversation"
    },
    "ENABLE_MEM0": {
        "type": "bool",
        "recommended": True,
        "description": "Enable advanced long-term memory using mem0.",
        "category": "Conversation"
    },
    "EMBEDDING_CACHE_SIZE": {
        "type": "int",
        "min": 0,
        "max": 20000,
        "step": 128,
        "recommended": 1024,
        "description": "Number of embeddings to keep in memory.",
        "category": "Cache"
    },
    "QUERY_CACHE_SIZE": {
        "type": "int",
        "min": 0,
        "max": 2000,
        "step": 16,
        "recommended": 128,
        "description": "Number of query results to keep in memory.",
        "category": "Cache"
    },
    "QUERY_CACHE_TTL": {
        "type": "int",
        "min": 0,
        "max": 604800,
        "step": 60,
        "recommended": 3600,
        "description": "How long to keep query results in cache (seconds).",
        "category": "Cache"
    },
    "DEVICE": {
        "type": "choice",
        "options": ["auto", "cpu", "cuda"],
        "recommended": "auto",
        "description": "Hardware accelerator for embeddings.",
        "category": "Hardware"
    },
    "REFERENCES_DIR": {
        "type": "string",
        "description": "Path to your local reference documents.",
        "category": "Paths",
        "hide_recommended": True
    }
}


def load_system_prompt():
    """Load system prompt from external file."""
    prompt_path = BASE_DIR / "prompts" / "system_prompt.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return "You are a helpful AI assistant."

SYSTEM_PROMPT = load_system_prompt()

def get_config_state():
    """Returns the current configuration values and their metadata."""
    return {
        "values": {
            "LLM_MODEL": LLM_MODEL,
            "EMBEDDING_MODEL": EMBEDDING_MODEL,
            "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
            "LLM_TEMPERATURE": LLM_TEMPERATURE,
            "LLM_TOP_P": LLM_TOP_P,
            "LLM_MAX_TOKENS": LLM_MAX_TOKENS,
            "LLM_CONTEXT_WINDOW": LLM_CONTEXT_WINDOW,
            "TOP_K_CHUNKS": TOP_K_CHUNKS,
            "CHUNK_SIMILARITY_THRESHOLD": CHUNK_SIMILARITY_THRESHOLD,
            "MAX_CHUNK_SIZE": MAX_CHUNK_SIZE,
            "MIN_CHUNK_SIZE": MIN_CHUNK_SIZE,
            "CHUNK_OVERLAP_SENTENCES": CHUNK_OVERLAP_SENTENCES,
            "MAX_CONVERSATION_HISTORY": MAX_CONVERSATION_HISTORY,
            "ENABLE_CHAT_MEMORY": ENABLE_CHAT_MEMORY,
            "ENABLE_MEM0": ENABLE_MEM0,
            "EMBEDDING_CACHE_SIZE": EMBEDDING_CACHE_SIZE,
            "QUERY_CACHE_SIZE": QUERY_CACHE_SIZE,
            "QUERY_CACHE_TTL": QUERY_CACHE_TTL,
            "DEVICE": DEVICE,
            "REFERENCES_DIR": str(REFERENCES_DIR),
        },
        "metadata": CONFIG_METADATA
    }

def update_config(key: str, value: str):
    """Updates a configuration key in settings.json and reloads the environment."""
    # Ensure value is correctly typed
    meta = CONFIG_METADATA.get(key)
    typed_value = value
    if meta:
        try:
            if meta["type"] == "int": typed_value = int(value)
            if meta["type"] == "float": typed_value = float(value)
            if meta["type"] == "bool": typed_value = str(value).lower() == "true"
        except (ValueError, TypeError):
            logging.error(f"Invalid type for {key}: {value}")
            return

    # Load existing
    current_settings = load_user_settings()
    current_settings[key] = typed_value

    # Save to JSON
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Convert all values to JSON-serializable types just in case
        serializable_settings = {k: v for k, v in current_settings.items()}
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_settings, f, indent=4)
    except Exception as e:
        import traceback
        logging.error(f"Failed to save {SETTINGS_FILE}: {e}")
        logging.error(traceback.format_exc())
        raise e # Re-raise to let the router handle it correctly

    # Update global variables in memory for immediate effect
    if key in globals():
        globals()[key] = typed_value
    
    # Special handling for Path objects
    if key == "REFERENCES_DIR":
        globals()[key] = Path(typed_value)
