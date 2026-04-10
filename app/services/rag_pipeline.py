"""
Full RAG pipeline orchestration.
Ties together document parsing, chunking, embedding, FAISS retrieval, and LLM generation.
"""

import logging
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

from app.config import REFERENCES_DIR, UPLOADS_DIR
from app.models import SourceReference
from app.services.document_parser import parse_document, scan_directory
from app.services.chunker import semantic_chunk
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.llm import get_llm_client
from app.services import conversation_store
from app.services.cache import query_cache

logger = logging.getLogger(__name__)


async def index_document(file_path: str | Path, source: str = "upload") -> dict:
    """
    Parse, chunk, embed, and index a single document.

    Returns:
        Dict with document info and chunk count.
    """
    embedding_svc = get_embedding_service()
    vector_store = get_vector_store()

    # Parse document
    doc = parse_document(file_path)
    if not doc["text"]:
        logger.warning(f"No text extracted from {doc['filename']}")
        return {"filename": doc["filename"], "chunks": 0, "error": "No text extracted"}

    # Semantic chunking
    chunks = semantic_chunk(
        text=doc["text"],
        embed_fn=embedding_svc.embed_texts,
        source_file=doc["filename"],
    )

    if not chunks:
        return {"filename": doc["filename"], "chunks": 0, "error": "No chunks created"}

    # Embed all chunk texts
    chunk_texts = [c.text for c in chunks]
    embeddings = embedding_svc.embed_texts(chunk_texts)

    # Prepare metadata
    metadata_list = [
        {
            "text": c.text,
            "source_file": c.source_file,
            "chunk_index": c.chunk_index,
            "source": source,
        }
        for c in chunks
    ]

    # Clear any existing chunks for this file to prevent duplicates/stale data
    vector_store.remove_by_source(doc["filename"])

    # Add to FAISS
    vector_store.add(embeddings, metadata_list)
    vector_store.save()

    # Register in database
    from app.database import get_db
    doc_id = str(uuid.uuid4())
    db = await get_db()
    try:
        await db.execute(
            "INSERT OR REPLACE INTO documents (id, filename, filepath, file_type, file_size, chunk_count, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, doc["filename"], doc["filepath"], doc["file_type"],
             doc["file_size"], len(chunks), source),
        )
        await db.commit()
    finally:
        await db.close()

    logger.info(f"Indexed {doc['filename']}: {len(chunks)} chunks")
    return {
        "id": doc_id,
        "filename": doc["filename"],
        "chunks": len(chunks),
        "source": source,
    }


async def index_references_directory() -> list[dict]:
    """Scan and index all documents in the references directory."""
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    files = scan_directory(REFERENCES_DIR)
    if not files:
        logger.info("No files found in references directory")
        return []

    # Check which files are already indexed
    from app.database import get_db
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT filepath FROM documents WHERE source = 'reference'"
        )
        indexed = {row["filepath"] for row in await cursor.fetchall()}
    finally:
        await db.close()

    results = []
    for f in files:
        if str(f) in indexed:
            continue
        try:
            result = await index_document(f, source="reference")
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to index {f}: {e}")
            results.append({"filename": f.name, "chunks": 0, "error": str(e)})

    return results


def retrieve_context(query: str, top_k: int = None) -> tuple[str, list[SourceReference]]:
    """
    Retrieve relevant context for a query.

    Returns:
        Tuple of (formatted context string, list of source references)
    """
    # Check cache
    cached = query_cache.get(query)
    if cached is not None:
        return cached

    embedding_svc = get_embedding_service()
    vector_store = get_vector_store()

    # Embed query
    query_embedding = embedding_svc.embed_texts([query])[0]

    # Search FAISS
    results = vector_store.search(query_embedding, top_k=top_k)

    if not results:
        return "", []

    # Format context
    context_parts = []
    sources = []
    seen_texts = set()

    for r in results:
        text = r.get("text", "")
        if text in seen_texts:
            continue
        seen_texts.add(text)

        source_file = r.get("source_file", "Unknown")
        score = r.get("score", 0.0)

        context_parts.append(
            f"[Source: {source_file}]\n{text}"
        )
        sources.append(SourceReference(
            filename=source_file,
            chunk_text=text[:300] + ("..." if len(text) > 300 else ""),
            relevance_score=round(score, 4),
            chunk_index=r.get("chunk_index", 0),
        ))

    context = "\n\n---\n\n".join(context_parts)

    # Cache result
    query_cache.put(query, (context, sources))

    return context, sources


import time

async def chat_with_rag(
    user_message: str,
    conversation_id: Optional[str] = None,
    attachment_path: Optional[str] = None,
    web_search: bool = False,
    parent_id: Optional[str] = None,
) -> AsyncGenerator[dict, None]:
    """
    Full RAG chat pipeline with streaming.
    """
    start_time = time.time()
    llm = get_llm_client()

    # Create or load conversation
    if not conversation_id or conversation_id in ("undefined", "null", "None"):
        conversation_id = await conversation_store.create_conversation()
        is_new = True
    else:
        is_new = False


    # Handle attachment
    if attachment_path:
        try:
            yield {"type": "info", "content": "📄 Processing attachment..."}
            result = await index_document(attachment_path, source="attachment")
            yield {
                "type": "info",
                "content": f"✅ Indexed {result['filename']} ({result['chunks']} chunks)",
            }
        except Exception as e:
            yield {"type": "info", "content": f"⚠️ Failed to process attachment: {e}"}

    # Save user message (unless regenerating)
    user_msg_id = parent_id
    if not user_message and parent_id:
        # Regeneration case: use parent message's content
        parent_msg = await conversation_store.get_message(parent_id)
        if parent_msg:
            user_message = parent_msg.content
        else:
            yield {"type": "error", "content": "Could not find parent message for regeneration."}
            return
    else:
        # Normal case: Save new user message
        user_msg_id = await conversation_store.add_message(conversation_id, "user", user_message, parent_id=parent_id)


    # Auto-title on first message
    if is_new:
        await conversation_store.auto_title(conversation_id, user_message)

    # Web search
    web_context = ""
    if web_search:
        yield {"type": "info", "content": "🌍 Searching the web..."}
        try:
            from app.services.web_search import search_web
            # Ideally this would be async, but duckduckgo-search is fast enough for now
            web_results = search_web(user_message)
            if web_results:
                web_context = "--- LIVE WEB SEARCH RESULTS ---\n" + web_results + "\n\n"
                yield {"type": "info", "content": "✅ Web search complete"}
            else:
                yield {"type": "info", "content": "⚠️ No web results found"}
        except Exception as e:
            logger.error(f"Web search error in pipeline: {e}")
            yield {"type": "info", "content": f"⚠️ Web search failed: {e}"}

    # Retrieve context from FAISS
    context, sources = retrieve_context(user_message)
    
    # Combine web and local context
    full_context = web_context + context

    # Send sources to frontend
    if sources:
        yield {
            "type": "sources",
            "sources": [s.model_dump() for s in sources],
        }

    # Get conversation history ONLY FOR THIS BRANCH
    history = await conversation_store.get_conversation_history(conversation_id, leaf_message_id=user_msg_id)

    # Stream LLM response
    full_response = ""
    async for token in llm.stream_chat(
        user_message=user_message,
        context=full_context,
        conversation_history=history[:-1],  # Exclude the message we just added
    ):
        full_response += token
        yield {"type": "token", "content": token}

    # Save assistant response
    generation_time = round(time.time() - start_time, 2)
    sources_data = [s.model_dump() for s in sources] if sources else []
    ast_msg_id = await conversation_store.add_message(
        conversation_id, "assistant", full_response, sources=sources_data, parent_id=user_msg_id, generation_time=generation_time
    )

    yield {
        "type": "done",
        "conversation_id": conversation_id,
        "message_id": ast_msg_id,
        "generation_time": generation_time,
    }
