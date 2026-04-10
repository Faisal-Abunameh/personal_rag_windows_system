"""
Embedding service — dual-mode support for NeMo Retriever NIM and sentence-transformers.
Auto-detects available backend on initialization.
"""

import logging
from typing import Optional

import numpy as np
import httpx

from app.config import (
    NEMO_RETRIEVER_URL,
    NEMO_RETRIEVER_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    EMBEDDING_DIM,
)
from app.services.cache import embedding_cache

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Unified embedding service with auto-fallback.
    Primary: NVIDIA NeMo Retriever NIM (Docker)
    Fallback: sentence-transformers (local, CPU-friendly)
    """

    def __init__(self):
        self._mode: str = ""  # "nemo" or "sentence-transformers"
        self._model = None
        self._dim: int = 0
        self._initialized = False

    async def initialize(self):
        """Auto-detect and initialize the best available embedding backend."""
        if self._initialized:
            return

        # Try NeMo Retriever first
        if await self._try_nemo():
            self._mode = "nemo"
            logger.info(
                f"Using NeMo Retriever embeddings ({NEMO_RETRIEVER_MODEL}), "
                f"dim={self._dim}"
            )
        else:
            self._init_sentence_transformers()
            self._mode = "sentence-transformers"
            logger.info(
                f"Using sentence-transformers ({FALLBACK_EMBEDDING_MODEL}), "
                f"dim={self._dim}"
            )

        self._initialized = True

    async def _try_nemo(self) -> bool:
        """Try to connect to NeMo Retriever NIM endpoint."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{NEMO_RETRIEVER_URL}/embeddings",
                    json={
                        "input": ["test"],
                        "model": NEMO_RETRIEVER_MODEL,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    embedding = data["data"][0]["embedding"]
                    self._dim = len(embedding)
                    return True
        except Exception as e:
            logger.debug(f"NeMo Retriever not available: {e}")
        return False

    def _init_sentence_transformers(self):
        """Initialize sentence-transformers fallback model."""
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading sentence-transformers model: {FALLBACK_EMBEDDING_MODEL}")
        self._model = SentenceTransformer(FALLBACK_EMBEDDING_MODEL)
        self._dim = self._model.get_sentence_embedding_dimension()

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
            if self._mode == "nemo":
                new_embeddings = self._embed_nemo(uncached_texts)
            else:
                new_embeddings = self._embed_st(uncached_texts)

            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                embedding_cache.put(text, emb)
                results.append((idx, emb))

        # Sort by original index and stack
        results.sort(key=lambda x: x[0])
        return np.array([r[1] for r in results], dtype=np.float32)

    def _embed_nemo(self, texts: list[str]) -> list[np.ndarray]:
        """Embed texts using NeMo Retriever NIM."""
        import httpx as httpx_sync

        embeddings = []
        # Batch in groups of 32
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = httpx_sync.post(
                f"{NEMO_RETRIEVER_URL}/embeddings",
                json={"input": batch, "model": NEMO_RETRIEVER_MODEL},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data["data"]:
                embeddings.append(np.array(item["embedding"], dtype=np.float32))

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
        if EMBEDDING_DIM > 0:
            return EMBEDDING_DIM
        return self._dim

    @property
    def model_name(self) -> str:
        if self._mode == "nemo":
            return NEMO_RETRIEVER_MODEL
        return FALLBACK_EMBEDDING_MODEL


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get the singleton embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
