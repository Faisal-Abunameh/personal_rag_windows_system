"""
Conversation CRUD API endpoints.
"""

from fastapi import APIRouter, HTTPException

from app.models import RenameRequest
from app.services import conversation_store

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("")
async def list_conversations():
    """List all conversations, most recent first."""
    convs = await conversation_store.list_conversations()
    return [c.model_dump() for c in convs]


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a conversation with all messages."""
    conv = await conversation_store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv.model_dump()


@router.put("/{conversation_id}")
async def rename_conversation(conversation_id: str, req: RenameRequest):
    """Rename a conversation."""
    conv = await conversation_store.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await conversation_store.update_conversation_title(conversation_id, req.title)
    return {"status": "ok"}


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation and all its messages."""
    await conversation_store.delete_conversation(conversation_id)
    return {"status": "ok"}
