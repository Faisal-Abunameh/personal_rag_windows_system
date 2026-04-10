"""
Caching utilities — LRU cache for embeddings and query results.
"""

import hashlib
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

from app.config import EMBEDDING_CACHE_SIZE, QUERY_CACHE_SIZE, QUERY_CACHE_TTL


class LRUCache:
    """Thread-safe LRU cache with optional TTL."""

    def __init__(self, max_size: int = 1024, ttl: Optional[int] = None):
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        hashed = self._make_key(key)
        with self._lock:
            if hashed not in self._cache:
                self._misses += 1
                return None
            value, ts = self._cache[hashed]
            if self._ttl and (time.time() - ts) > self._ttl:
                del self._cache[hashed]
                self._misses += 1
                return None
            self._cache.move_to_end(hashed)
            self._hits += 1
            return value

    def put(self, key: str, value: Any):
        hashed = self._make_key(key)
        with self._lock:
            if hashed in self._cache:
                self._cache.move_to_end(hashed)
                self._cache[hashed] = (value, time.time())
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[hashed] = (value, time.time())

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0,
            }


# Singleton caches
embedding_cache = LRUCache(max_size=EMBEDDING_CACHE_SIZE)
query_cache = LRUCache(max_size=QUERY_CACHE_SIZE, ttl=QUERY_CACHE_TTL)
