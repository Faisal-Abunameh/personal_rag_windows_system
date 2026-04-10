"""
FAISS vector store management.
Handles indexing, searching, and persistence of document embeddings.
"""

import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Optional

import faiss
import numpy as np

from app.config import FAISS_INDEX_DIR, TOP_K_CHUNKS

logger = logging.getLogger(__name__)

# Metadata file stores chunk info alongside the FAISS index
METADATA_FILE = FAISS_INDEX_DIR / "metadata.json"
INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"


class VectorStore:
    """
    FAISS-backed vector store with metadata mapping.
    Uses IndexFlatIP (inner product) for cosine similarity with normalized vectors.
    """

    def __init__(self):
        self._index: Optional[faiss.IndexFlatIP] = None
        self._metadata: list[dict] = []  # parallel array: index i → metadata[i]
        self._dim: int = 0
        self._lock = Lock()

    def initialize(self, dim: int):
        """Initialize or load the FAISS index."""
        self._dim = dim
        FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

        if INDEX_FILE.exists() and METADATA_FILE.exists():
            self._load()
        else:
            self._index = faiss.IndexFlatIP(dim)
            self._metadata = []
            logger.info(f"Created new FAISS index with dim={dim}")

    def _load(self):
        """Load index and metadata from disk."""
        try:
            self._index = faiss.read_index(str(INDEX_FILE))
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
            self._dim = self._index.d
            logger.info(
                f"Loaded FAISS index: {self._index.ntotal} vectors, dim={self._dim}"
            )
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self._index = faiss.IndexFlatIP(self._dim)
            self._metadata = []

    def save(self):
        """Persist index and metadata to disk."""
        with self._lock:
            FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(INDEX_FILE))
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, ensure_ascii=False)
            logger.debug(f"Saved FAISS index: {self._index.ntotal} vectors")

    def add(self, embeddings: np.ndarray, metadata_list: list[dict]):
        """
        Add vectors with metadata to the index.

        Args:
            embeddings: numpy array of shape (n, dim)
            metadata_list: list of dicts with keys like 'text', 'source_file', 'chunk_index'
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embeddings and metadata must have the same length")

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = (embeddings / norms).astype(np.float32)

        with self._lock:
            self._index.add(normalized)
            self._metadata.extend(metadata_list)

        logger.info(f"Added {len(embeddings)} vectors to FAISS index")

    def search(self, query_embedding: np.ndarray, top_k: Optional[int] = None) -> list[dict]:
        """
        Search for the most similar chunks.

        Args:
            query_embedding: numpy array of shape (dim,) or (1, dim)
            top_k: number of results to return

        Returns:
            List of dicts with 'text', 'source_file', 'chunk_index', 'score'
        """
        k = top_k or TOP_K_CHUNKS

        if self._index is None or self._index.ntotal == 0:
            return []

        # Ensure correct shape
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize query
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        query_embedding = query_embedding.astype(np.float32)

        # Clamp k to available vectors
        k = min(k, self._index.ntotal)

        with self._lock:
            scores, indices = self._index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._metadata):
                continue
            meta = self._metadata[idx].copy()
            meta["score"] = float(score)
            results.append(meta)

        return results

    def remove_by_source(self, source_file: str):
        """
        Remove all vectors from a specific source file.
        Note: FAISS IndexFlatIP doesn't support removal, so we rebuild.
        """
        with self._lock:
            keep_indices = [
                i for i, m in enumerate(self._metadata)
                if m.get("source_file") != source_file
            ]

            if len(keep_indices) == len(self._metadata):
                return  # Nothing to remove

            if not keep_indices:
                self._index = faiss.IndexFlatIP(self._dim)
                self._metadata = []
                return

            # Reconstruct vectors for kept indices
            vectors = np.array(
                [self._index.reconstruct(i) for i in keep_indices],
                dtype=np.float32,
            )
            new_metadata = [self._metadata[i] for i in keep_indices]

            self._index = faiss.IndexFlatIP(self._dim)
            self._index.add(vectors)
            self._metadata = new_metadata

        logger.info(f"Removed vectors for source: {source_file}")

    @property
    def total_vectors(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal

    @property
    def index_size_bytes(self) -> int:
        if INDEX_FILE.exists():
            return INDEX_FILE.stat().st_size
        return 0

    @property
    def dim(self) -> int:
        return self._dim

    def get_indexed_sources(self) -> list[str]:
        """Get unique source files in the index."""
        sources = set()
        for m in self._metadata:
            src = m.get("source_file", "")
            if src:
                sources.add(src)
        return sorted(sources)

    def get_chunks_by_source(self, source_file: str) -> list[dict]:
        """Get all chunks for a specific source file, sorted by chunk_index."""
        chunks = [
            m for m in self._metadata
            if m.get("source_file") == source_file
        ]
        # Sort by chunk_index if available
        return sorted(chunks, key=lambda x: x.get("chunk_index", 0))


# Singleton instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get the singleton vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
