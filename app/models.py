"""
Pydantic models for API request/response contracts.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Chat
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    web_search: bool = False


class SourceReference(BaseModel):
    filename: str
    chunk_text: str
    relevance_score: float = 0.0
    chunk_index: int = 0


class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    sources: list[SourceReference] = []


class StreamChunk(BaseModel):
    type: str  # "token", "sources", "done", "error"
    content: str = ""
    sources: list[SourceReference] = []
    conversation_id: str = ""
    message_id: str = ""
    generation_time: Optional[float] = None


# ──────────────────────────────────────────────
# Conversations
# ──────────────────────────────────────────────
class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class Message(BaseModel):
    id: str
    role: str
    content: str
    sources: list[SourceReference] = []
    created_at: str
    parent_id: Optional[str] = None
    generation_time: Optional[float] = None


class ConversationDetail(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[Message] = []


class RenameRequest(BaseModel):
    title: str


# ──────────────────────────────────────────────
# Documents
# ──────────────────────────────────────────────
class DocumentInfo(BaseModel):
    id: str
    filename: str
    filepath: str
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    chunk_count: int = 0
    indexed_at: str
    source: str = "upload"


class IndexStatus(BaseModel):
    total_documents: int = 0
    total_chunks: int = 0
    index_size_bytes: int = 0
    embedding_model: str = ""
    embedding_dim: int = 0


# ──────────────────────────────────────────────
# System
# ──────────────────────────────────────────────
class SystemStatus(BaseModel):
    ollama_available: bool = False
    model_loaded: bool = False
    model_name: str = ""
    embedding_mode: str = ""  # "nemo" or "sentence-transformers"
    total_documents: int = 0
    total_chunks: int = 0
    references_dir: str = ""
