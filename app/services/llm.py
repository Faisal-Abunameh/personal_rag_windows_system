"""
Ollama / Gemma 4 LLM client.
Handles streaming chat completions and model management.
"""

import json
import logging
from typing import AsyncGenerator, Optional

import httpx

from app.config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TOP_P,
    LLM_MAX_TOKENS,
    SYSTEM_PROMPT,
    MAX_CONVERSATION_HISTORY,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client for Ollama API with streaming support."""

    def __init__(self):
        self._base_url = OLLAMA_BASE_URL
        self._model = LLM_MODEL
        self._available = False

    async def check_health(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    model_names = [m.get("name", "").split(":")[0] for m in models]
                    self._available = True
                    if self._model.split(":")[0] in model_names:
                        return True
                    logger.warning(
                        f"Model '{self._model}' not found. "
                        f"Available: {model_names}"
                    )
                    return False
        except Exception as e:
            logger.error(f"Ollama not reachable: {e}")
            self._available = False
            return False

    async def pull_model(self) -> bool:
        """Pull the configured model from Ollama."""
        logger.info(f"Pulling model: {self._model}")
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    f"{self._base_url}/api/pull",
                    json={"name": self._model},
                )
                return resp.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False

    async def stream_chat(
        self,
        user_message: str,
        context: str = "",
        conversation_history: list[dict] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat completion from the LLM.

        Args:
            user_message: The user's message
            context: Retrieved RAG context (formatted chunks)
            conversation_history: List of {"role": "...", "content": "..."} dicts
            system_prompt: Override the default system prompt

        Yields:
            Token strings as they arrive
        """
        messages = self._build_messages(
            user_message, context, conversation_history, system_prompt
        )

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/api/chat",
                    json={
                        "model": self._model,
                        "messages": messages,
                        "stream": True,
                        "options": {
                            "temperature": LLM_TEMPERATURE,
                            "top_p": LLM_TOP_P,
                            "num_predict": LLM_MAX_TOKENS,
                        },
                    },
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                token = data["message"]["content"]
                                if token:
                                    yield token
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield f"\n\n⚠️ Error communicating with LLM: {str(e)}"

    def _build_messages(
        self,
        user_message: str,
        context: str,
        conversation_history: list[dict] = None,
        system_prompt: Optional[str] = None,
    ) -> list[dict]:
        """Build the messages array for the Ollama API."""
        messages = []

        # System prompt
        sys_prompt = system_prompt or SYSTEM_PROMPT
        if context:
            sys_prompt += (
                "\n\n--- RETRIEVED CONTEXT ---\n"
                "Use the following context to answer the user's question. "
                "Cite the source document when referencing specific information.\n\n"
                f"{context}\n"
                "--- END CONTEXT ---"
            )
        messages.append({"role": "system", "content": sys_prompt})

        # Conversation history (sliding window)
        if conversation_history:
            # Keep last N messages to fit context window
            history = conversation_history[-MAX_CONVERSATION_HISTORY:]
            messages.extend(history)

        # Current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def is_available(self) -> bool:
        return self._available


# Singleton
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get the singleton LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
