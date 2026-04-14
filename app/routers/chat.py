"""
Chat API endpoints with Server-Sent Events (SSE) streaming.
"""

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse

import app.config as config
from app.services.rag_pipeline import chat_with_rag

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("")
@router.post("/{conversation_id}")
async def chat(
    conversation_id: Optional[str] = None,
    message: str = Form(...),
    web_search: bool = Form(False),
    parent_id: Optional[str] = Form(None),
    attachment: Optional[UploadFile] = File(None),
):
    """
    Send a chat message and receive a streaming response via SSE.
    Supports optional file attachment.
    """
    attachment_path = None

    # Handle file upload
    if attachment and attachment.filename:
        config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        unique_name = f"{uuid.uuid4().hex[:8]}_{attachment.filename}"
        attachment_path = str(config.UPLOADS_DIR / unique_name)
        with open(attachment_path, "wb") as f:
            shutil.copyfileobj(attachment.file, f)
        logger.info(f"Saved attachment: {unique_name}")

    async def event_stream():
        try:
            async for chunk in chat_with_rag(
                user_message=message,
                conversation_id=conversation_id,
                attachment_path=attachment_path,
                web_search=web_search,
                parent_id=parent_id,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            import traceback
            logger.error(f"Chat stream error: {e}")
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'content': str(e) or 'Internal Server Error'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
