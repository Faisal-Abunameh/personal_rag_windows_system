"""
Memory service using mem0ai.
Provides per-chat isolated long-term memory.
"""

import logging
from typing import Optional, List, Dict
from mem0 import Memory
import app.config as config
from app.services.llm import get_llm_client
from app.services.embeddings import get_embedding_service

logger = logging.getLogger(__name__)

class MemoryService:
    """
    Service wrapper for mem0ai.
    Integrates with Ollama for LLM and Embeddings.
    """

    def __init__(self):
        self._memory: Optional[Memory] = None
        self._initialized = False

    async def initialize(self):
        """Initialize mem0 with Ollama configuration."""
        if self._initialized:
            return

        if not config.ENABLE_MEM0:
            logger.info("mem0 memory is disabled in config.")
            return

        try:
            # Get current embedding info
            embed_svc = get_embedding_service()
            await embed_svc.initialize()
            
            # mem0 configuration
            mem0_config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "personal_rag_memory",
                        "path": str(config.MEM0_DIR),
                        "embedding_model_dims": embed_svc.dim,
                    },
                },
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": "llama3.2:latest",
                        "temperature": 0,
                        "ollama_base_url": config.OLLAMA_BASE_URL,
                    },
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": embed_svc.model_name,
                        "ollama_base_url": config.OLLAMA_BASE_URL,
                    },
                },
            }

            # If embedding mode is sentence-transformers, we might need a different config
            # but mem0's Ollama provider expects a model name that Ollama knows.
            # If the user is using sentence-transformers, we'll try to use the same model name
            # but it might fail if Ollama doesn't have it.
            # For now, we assume Ollama has the model.

            self._memory = Memory.from_config(mem0_config)
            self._initialized = True
            logger.info(f"mem0 initialized successfully using {embed_svc.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize mem0: {e}")
            self._memory = None

    async def add(self, messages: List[Dict[str, str]], conversation_id: str):
        """Add a conversation turn to memory."""
        if not self._initialized or not self._memory:
            await self.initialize()
            if not self._memory:
                return

        try:
            # mem0 uses user_id for isolation
            self._memory.add(messages, user_id=conversation_id)
            logger.debug(f"Added interaction to mem0 for chat {conversation_id}")
        except Exception as e:
            logger.error(f"Error adding to mem0: {e}")

    async def search(self, query: str, conversation_id: str, limit: int = 5) -> List[str]:
        """Search memory for relevant facts."""
        if not self._initialized or not self._memory:
            await self.initialize()
            if not self._memory:
                return []

        try:
            results = self._memory.search(query, user_id=conversation_id, limit=limit)
            logger.debug(f"Raw mem0 results for {conversation_id}: {results}")
            
            # mem0 returns a dict with 'results' key in some versions/providers
            if isinstance(results, dict) and "results" in results:
                results = results["results"]

            # Extract memory content from results
            memories = []
            for res in results:
                if isinstance(res, dict) and "memory" in res:
                    memories.append(res["memory"])
                elif isinstance(res, str):
                    memories.append(res)
            return memories
        except Exception as e:
            logger.error(f"Error searching mem0: {e}")
            return []

    async def delete_chat_memory(self, conversation_id: str):
        """Delete all memories for a specific chat."""
        if not self._initialized or not self._memory:
            return

        try:
            self._memory.delete_all(user_id=conversation_id)
            logger.info(f"Deleted all mem0 memories for chat {conversation_id}")
        except Exception as e:
            logger.error(f"Error deleting mem0 memory: {e}")

# Singleton
_memory_service: Optional[MemoryService] = None

def get_memory_service() -> MemoryService:
    """Get the singleton memory service."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service
