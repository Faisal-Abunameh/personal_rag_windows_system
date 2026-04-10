"""
SQLite-backed conversation persistence.
Manages conversations and messages with async operations.
"""

import json
import logging
import uuid
from typing import Optional

from app.database import get_db
from app.models import (
    ConversationSummary,
    ConversationDetail,
    Message,
    SourceReference,
)

logger = logging.getLogger(__name__)


async def create_conversation(title: str = "New Chat") -> str:
    """Create a new conversation and return its ID."""
    conv_id = str(uuid.uuid4())
    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            (conv_id, title),
        )
        await db.commit()
    finally:
        await db.close()
    return conv_id


async def list_conversations() -> list[ConversationSummary]:
    """List all conversations ordered by most recent activity."""
    db = await get_db()
    try:
        cursor = await db.execute("""
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        """)
        rows = await cursor.fetchall()
        return [
            ConversationSummary(
                id=row["id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["message_count"],
            )
            for row in rows
        ]
    finally:
        await db.close()


async def get_conversation(conv_id: str) -> Optional[ConversationDetail]:
    """Get a conversation with all its messages."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conv_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        cursor = await db.execute(
            "SELECT id, role, content, sources, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY created_at ASC",
            (conv_id,),
        )
        msg_rows = await cursor.fetchall()

        messages = []
        for mr in msg_rows:
            sources = []
            try:
                raw = json.loads(mr["sources"] or "[]")
                sources = [SourceReference(**s) for s in raw]
            except (json.JSONDecodeError, TypeError):
                pass

            messages.append(Message(
                id=mr["id"],
                role=mr["role"],
                content=mr["content"],
                sources=sources,
                created_at=mr["created_at"],
            ))

        return ConversationDetail(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            messages=messages,
        )
    finally:
        await db.close()


async def add_message(
    conv_id: str,
    role: str,
    content: str,
    sources: list[dict] = None,
) -> str:
    """Add a message to a conversation. Returns message ID."""
    msg_id = str(uuid.uuid4())
    sources_json = json.dumps(sources or [], ensure_ascii=False)

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, sources) "
            "VALUES (?, ?, ?, ?, ?)",
            (msg_id, conv_id, role, content, sources_json),
        )
        await db.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
            (conv_id,),
        )
        await db.commit()
    finally:
        await db.close()
    return msg_id


async def get_conversation_history(conv_id: str) -> list[dict]:
    """Get conversation history formatted for the LLM."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT role, content FROM messages "
            "WHERE conversation_id = ? AND role IN ('user', 'assistant') "
            "ORDER BY created_at ASC",
            (conv_id,),
        )
        rows = await cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]
    finally:
        await db.close()


async def update_conversation_title(conv_id: str, title: str):
    """Rename a conversation."""
    db = await get_db()
    try:
        await db.execute(
            "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ?",
            (title, conv_id),
        )
        await db.commit()
    finally:
        await db.close()


async def delete_conversation(conv_id: str):
    """Delete a conversation and all its messages."""
    db = await get_db()
    try:
        await db.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        await db.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        await db.commit()
    finally:
        await db.close()


async def auto_title(conv_id: str, first_message: str):
    """Generate a title from the first user message."""
    title = first_message.strip()[:60]
    if len(first_message) > 60:
        title = title.rsplit(" ", 1)[0] + "..."
    await update_conversation_title(conv_id, title)
