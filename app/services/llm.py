"""
Ollama LLM client — supports any Ollama model.
Handles streaming chat completions and model management.
"""

import json
import logging
from typing import AsyncGenerator, Optional

import httpx

import app.config as config

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client for Ollama API with streaming support."""

    def __init__(self):
        # We don't bind to config at init, but use config.VALUE in methods
        # to ensure we always use the latest values.
        pass

    async def check_health(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    model_names = [m.get("name", "").split(":")[0] for m in models]
                    self._available = True
                    if config.LLM_MODEL.split(":")[0] in model_names:
                        self._model_loaded = True
                        return True
                    logger.warning(
                        f"Model '{config.LLM_MODEL}' not found. "
                        f"Available: {model_names}"
                    )
                    self._model_loaded = False
                    return False
        except Exception as e:
            logger.error(f"Ollama not reachable: {e}")
            self._available = False
            self._model_loaded = False
            return False

    async def list_available_models(self) -> list[dict]:
        """List all models available in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    return [
                        {
                            "name": m.get("name", ""),
                            "size": m.get("size", 0),
                            "modified_at": m.get("modified_at", ""),
                            "family": m.get("details", {}).get("family", ""),
                            "parameter_size": m.get("details", {}).get("parameter_size", ""),
                            "quantization": m.get("details", {}).get("quantization_level", ""),
                        }
                        for m in models
                    ]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []

    def set_model(self, model_name: str):
        """Switch the active model at runtime."""
        logger.info(f"Switching model: {config.LLM_MODEL} -> {model_name}")
        config.update_config("LLM_MODEL", model_name)
        self._model_loaded = False  # Will be verified on next health check

    async def pull_model(self) -> bool:
        """Pull the configured model from Ollama."""
        logger.info(f"Pulling model: {self._model}")
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    f"{config.OLLAMA_BASE_URL}/api/pull",
                    json={"name": config.LLM_MODEL},
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
        
        logger.info(f"LLM Chat Request: model={config.LLM_MODEL}, history_len={len(conversation_history or [])}")
        logger.debug(f"LLM Full Messages: {json.dumps(messages, ensure_ascii=False)}")

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "POST",
                    f"{config.OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": config.LLM_MODEL,
                        "messages": messages,
                        "stream": True,
                        "options": {
                            "temperature": config.LLM_TEMPERATURE,
                            "top_p": config.LLM_TOP_P,
                            "num_predict": config.LLM_MAX_TOKENS,
                        },
                    },
                ) as resp:
                    if resp.status_code != 200:
                        err_text = await resp.aread()
                        logger.error(f"Ollama API error ({resp.status_code}): {err_text}")
                        yield f"\n\n⚠️ Ollama error {resp.status_code}: {err_text.decode()}"
                        return

                    token_count = 0
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                token = data["message"]["content"]
                                if token:
                                    token_count += 1
                                    logger.debug(f"LLM token [{token_count}]: {repr(token)}")
                                    yield token
                            if data.get("done", False):
                                logger.info(f"LLM generation finished. Total tokens: {token_count}")
                                break
                        except json.JSONDecodeError as je:
                            logger.error(f"JSON decode error in LLM stream: {je} | Line: {line}")
                            continue
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield f"\n\n⚠️ Error communicating with LLM: {str(e)}"

    async def generate_title(self, user_message: str, assistant_response: str) -> str:
        """
        Generate a short, descriptive title for a conversation based on the first interaction.
        """
        prompt = (
            "Summarize the following chat interaction into a short, concise title (max 5 words). "
            "Do not use quotes, punctuation, or special characters. Use Title Case.\n\n"
            f"User: {user_message}\n"
            f"Assistant: {assistant_response}"
        )

        messages = [{"role": "user", "content": prompt}]

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{config.OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": config.LLM_MODEL,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 20,
                        },
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    title = data.get("message", {}).get("content", "").strip()
                    # Clean up quotes if the LLM included them despite instructions
                    title = title.strip('"').strip("'").strip()
                    logger.info(f"Generated conversation title: {title}")
                    return title if title else "New Chat"
        except Exception as e:
            logger.error(f"Title generation error: {e}")
        
        return "New Chat"

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
        sys_prompt = system_prompt or config.SYSTEM_PROMPT
        if context:
            sys_prompt += (
                "\n\n--- PROVIDED SOURCE LIST ---\n"
                f"{context}\n"
                "--- END SOURCE LIST ---\n"
            )
        else:
            sys_prompt += "\n\n(Note: No specific local context found.)"
        
        messages.append({"role": "system", "content": sys_prompt})

        # Conversation history (sliding window)
        if conversation_history:
            # Keep last N messages to fit context window
            history = conversation_history[-config.MAX_CONVERSATION_HISTORY:]
            messages.extend(history)

        # Current user message
        messages.append({"role": "user", "content": user_message})

        return messages

    @property
    def model_name(self) -> str:
        return config.LLM_MODEL

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
