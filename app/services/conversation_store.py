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

async def _delete_mem0_memory(conv_id: str):
    """Helper to delete mem0 memory for a conversation."""
    from app.services.memory import get_memory_service
    memory_svc = get_memory_service()
    await memory_svc.delete_chat_memory(conv_id)



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
            "SELECT id, role, content, sources, created_at, parent_id, generation_time FROM messages "
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

            attachments = []
            try:
                # sqlite3.Row doesn't have .get(), use indexing
                raw_att = json.loads(mr["attachments"] if "attachments" in mr.keys() else "[]")
                attachments = [AttachmentInfo(**a) for a in raw_att]
            except (json.JSONDecodeError, TypeError, IndexError):
                pass

            messages.append(Message(
                id=mr["id"],
                role=mr["role"],
                content=mr["content"],
                sources=sources,
                attachments=attachments,
                created_at=mr["created_at"],
                parent_id=mr["parent_id"],
                generation_time=mr["generation_time"],
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
    attachments: list[dict] = None,
    parent_id: Optional[str] = None,
    generation_time: Optional[float] = None,
) -> str:
    """Add a message to a conversation. Returns message ID."""
    if not conv_id or conv_id in ("undefined", "null", "None"):
        raise ValueError(f"Invalid conversation_id '{conv_id}' passed to add_message")
        
    msg_id = str(uuid.uuid4())
    sources_json = json.dumps(sources or [], ensure_ascii=False)
    attachments_json = json.dumps(attachments or [], ensure_ascii=False)

    db = await get_db()
    try:
        await db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, sources, attachments, parent_id, generation_time) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (msg_id, conv_id, role, content, sources_json, attachments_json, parent_id, generation_time),
        )
        await db.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
            (conv_id,),
        )
        await db.commit()
    finally:
        await db.close()
    return msg_id


async def get_message(msg_id: str) -> Optional[Message]:
    """Get a single message by ID."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, role, content, sources, created_at, parent_id, generation_time FROM messages WHERE id = ?",
            (msg_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None

        sources = []
        try:
            raw = json.loads(row["sources"] or "[]")
            sources = [SourceReference(**s) for s in raw]
        except (json.JSONDecodeError, TypeError):
            pass

        attachments = []
        try:
            # sqlite3.Row doesn't have .get(), use indexing
            raw_att = json.loads(row["attachments"] if "attachments" in row.keys() else "[]")
            attachments = [AttachmentInfo(**a) for a in raw_att]
        except (json.JSONDecodeError, TypeError, IndexError):
            pass

        return Message(
            id=row["id"],
            role=row["role"],
            content=row["content"],
            sources=sources,
            attachments=attachments,
            created_at=row["created_at"],
            parent_id=row["parent_id"],
            generation_time=row["generation_time"],
        )
    finally:
        await db.close()


async def get_conversation_history(conv_id: str, leaf_message_id: Optional[str] = None) -> list[dict]:
    """
    Get conversation history formatted for the LLM.
    If leaf_message_id is provided, it follows the chain of parents to build the specific branch.
    """
    db = await get_db()
    try:
        if leaf_message_id:
            # Use Recursive CTE to find all ancestors of the leaf message
            cursor = await db.execute("""
                WITH RECURSIVE message_chain(id, role, content, parent_id, created_at) AS (
                    SELECT id, role, content, parent_id, created_at FROM messages WHERE id = ?
                    UNION ALL
                    SELECT m.id, m.role, m.content, m.parent_id, m.created_at
                    FROM messages m
                    JOIN message_chain mc ON m.id = mc.parent_id
                )
                SELECT role, content FROM message_chain
                WHERE role IN ('user', 'assistant')
                ORDER BY created_at ASC
            """, (leaf_message_id,))
        else:
            # Fallback to linear history if no leaf is specified (for legacy or simple chats)
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
    
    # Also delete mem0 memory
    try:
        await _delete_mem0_memory(conv_id)
    except Exception as e:
        logger.error(f"Failed to delete mem0 memory for {conv_id}: {e}")



