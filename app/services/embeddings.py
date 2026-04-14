"""
Embedding service — supports Ollama embedding models and sentence-transformers fallback.
Auto-detects available backend on initialization.
"""

import logging
from typing import Optional

import numpy as np
import httpx

import app.config as config
from app.services.cache import embedding_cache

logger = logging.getLogger(__name__)

# Well-known Ollama embedding model families (used to auto-select from pulled models)
KNOWN_EMBEDDING_FAMILIES = {
    "nomic-embed-text", "mxbai-embed-large", "all-minilm",
    "snowflake-arctic-embed", "bge-m3", "bge-large",
    "paraphrase-multilingual", "multilingual-e5",
}


class EmbeddingService:
    """
    Unified embedding service with Ollama embedding models and
    sentence-transformers fallback.
    """

    def __init__(self):
        self._mode: str = ""  # "ollama" or "sentence-transformers"
        self._model = None
        self._model_name: str = ""
        self._dim: int = 0
        self._initialized = False

    async def initialize(self, preferred_model: str = ""):
        """
        Initialize the embedding backend.
        Priority: preferred Ollama model → auto-detect Ollama embedding model →
                  sentence-transformers fallback.
        """
        if self._initialized:
            return

        # Try Ollama embedding models
        ollama_model = preferred_model or await self._find_ollama_embedding_model()
        if ollama_model and await self._try_ollama(ollama_model):
            self._mode = "ollama"
            self._model_name = ollama_model
            logger.info(
                f"Using Ollama embeddings ({ollama_model}), dim={self._dim}"
            )
        else:
            # Fallback to sentence-transformers
            self._init_sentence_transformers()
            self._mode = "sentence-transformers"
            self._model_name = config.FALLBACK_EMBEDDING_MODEL
            logger.info(
                f"Using sentence-transformers ({config.FALLBACK_EMBEDDING_MODEL}), "
                f"dim={self._dim}"
            )

        self._initialized = True

    async def _find_ollama_embedding_model(self) -> Optional[str]:
        """Auto-detect an embedding model from Ollama's pulled models."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    for m in models:
                        name = m.get("name", "").split(":")[0]
                        if name in KNOWN_EMBEDDING_FAMILIES:
                            return m.get("name", "")
        except Exception:
            pass
        return None

    async def _try_ollama(self, model_name: str) -> bool:
        """Test an Ollama model for embedding support via /api/embed."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{config.OLLAMA_BASE_URL}/api/embed",
                    json={"model": model_name, "input": "test"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = data.get("embeddings", [])
                    if embeddings and len(embeddings) > 0:
                        self._dim = len(embeddings[0])
                        return True
        except Exception as e:
            logger.debug(f"Ollama embedding failed for {model_name}: {e}")
        return False

    def _init_sentence_transformers(self):
        """Initialize sentence-transformers fallback model."""
        from sentence_transformers import SentenceTransformer

        logger.info(
            f"Loading sentence-transformers model: {config.FALLBACK_EMBEDDING_MODEL} "
            f"on {config.DEVICE}"
        )
        self._model = SentenceTransformer(config.FALLBACK_EMBEDDING_MODEL, device=config.DEVICE)
        self._dim = self._model.get_sentence_embedding_dimension()

    async def switch_model(self, model_name: str, mode: str = "ollama") -> dict:
        """
        Switch to a different embedding model at runtime.
        Returns dict with success status and new dimension.
        Note: Changing dimensions requires re-indexing.
        """
        old_dim = self._dim
        old_mode = self._mode
        old_model = self._model_name

        if mode == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(model_name, device=config.DEVICE)
                self._dim = self._model.get_sentence_embedding_dimension()
                self._mode = "sentence-transformers"
                self._model_name = model_name
            except Exception as e:
                return {"success": False, "error": str(e)}
        else:
            # Ollama embedding model
            if await self._try_ollama(model_name):
                self._mode = "ollama"
                self._model_name = model_name
                self._model = None  # Not needed for Ollama
            else:
                return {"success": False, "error": f"Model '{model_name}' failed embedding test"}

        # Clear embedding cache since model changed
        embedding_cache.clear()

        dim_changed = old_dim != self._dim
        logger.info(
            f"Switched embedding model: {old_model} -> {self._model_name} "
            f"(dim: {old_dim} -> {self._dim})"
        )

        return {
            "success": True,
            "model_name": self._model_name,
            "mode": self._mode,
            "dim": self._dim,
            "dim_changed": dim_changed,
            "needs_reindex": dim_changed,
        }

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of texts and return numpy array of shape (n, dim).
        Uses caching to avoid re-computing embeddings.
        """
        if not self._initialized:
            raise RuntimeError("EmbeddingService not initialized. Call initialize() first.")

        results = []
        uncached_indices = []
        uncached_texts = []

        # Check cache first
        for i, text in enumerate(texts):
            cached = embedding_cache.get(text)
            if cached is not None:
                results.append((i, cached))
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Compute uncached embeddings
        if uncached_texts:
            if self._mode == "ollama":
                new_embeddings = self._embed_ollama(uncached_texts)
            else:
                new_embeddings = self._embed_st(uncached_texts)

            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                embedding_cache.put(text, emb)
                results.append((idx, emb))

        # Sort by original index and stack
        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results], dtype=np.float32)

    def _embed_ollama(self, texts: list[str]) -> list[np.ndarray]:
        """Embed texts using Ollama /api/embed endpoint."""
        embeddings = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = httpx.post(
                f"{config.OLLAMA_BASE_URL}/api/embed",
                json={"model": self._model_name, "input": batch},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            for emb in data.get("embeddings", []):
                embeddings.append(np.array(emb, dtype=np.float32))

        return embeddings

    def _embed_st(self, texts: list[str]) -> list[np.ndarray]:
        """Embed texts using sentence-transformers."""
        embeddings = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [embeddings[i] for i in range(len(texts))]

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def dim(self) -> int:
        if config.EMBEDDING_DIM > 0:
            return config.EMBEDDING_DIM
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
