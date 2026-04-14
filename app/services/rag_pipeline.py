"""
Full RAG pipeline orchestration.
Ties together document parsing, chunking, embedding, FAISS retrieval, and LLM generation.
"""

import logging
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

import app.config as config
from app.models import SourceReference, AttachmentInfo
from app.services.document_parser import parse_document, scan_directory
from app.services.chunker import semantic_chunk
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.llm import get_llm_client
from app.services import conversation_store
from app.services.cache import query_cache
from app.services.memory import get_memory_service

logger = logging.getLogger(__name__)


async def index_document(file_path: str | Path, source: str = "upload", custom_source_id: Optional[str] = None) -> dict:
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

    # Source tracking: use custom ID if provided, otherwise filename
    source_id = custom_source_id or doc["filename"]

    # Semantic chunking
    chunks = semantic_chunk(
        text=doc["text"],
        embed_fn=embedding_svc.embed_texts,
        source_file=source_id,
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
    vector_store.remove_by_source(source_id)

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
        "filepath": str(file_path),
        "file_type": doc["file_type"],
        "file_size": doc["file_size"],
        "chunks": len(chunks),
        "source": source,
    }


async def index_references_directory() -> list[dict]:
    """Scan and index all documents in the references directory."""
    config.REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    files = scan_directory(config.REFERENCES_DIR)
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
            # Use relative path as source ID for references to handle deep structures correctly
            rel_path = str(f.relative_to(config.REFERENCES_DIR))
            result = await index_document(f, source="reference", custom_source_id=rel_path)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to index {f}: {e}")
            results.append({"filename": f.name, "chunks": 0, "error": str(e)})

    return results
    

async def reindex_all_references() -> dict:
    """Clear and fully rebuild the reference index."""
    vector_store = get_vector_store()
    
    # 1. Clear database entries for references
    from app.database import get_db
    db = await get_db()
    try:
        await db.execute("DELETE FROM documents WHERE source = 'reference'")
        await db.commit()
    finally:
        await db.close()
        
    # 2. Clear vectors
    vector_store.clear()
    vector_store.save()
    
    # 3. Scan and index again
    results = await index_references_directory()
    
    return {
        "success": True,
        "indexed_count": len(results),
        "results": results
    }


def retrieve_context(query: str, top_k: int = None) -> tuple[list[dict], list[SourceReference]]:
    """
    Retrieve relevant context for a query.

    Returns:
        Tuple of (list of chunk dictionaries, list of source references)
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
        logger.info("RAG search: No chunks found in FAISS index.")
        return [], []

    # Log best score for debugging
    top_score = results[0].get("score", 0.0)
    logger.info(f"RAG search: Top similarity score found: {top_score:.4f} (query: '{query[:50]}...')")

    # Filter by similarity threshold
    threshold = config.CHUNK_SIMILARITY_THRESHOLD
    results = [r for r in results if r.get("score", 0.0) >= threshold]
    
    if not results:
        logger.info(f"RAG Filter: All chunks below threshold {threshold}. Best was {top_score:.4f}")
        return [], []

    logger.info(f"RAG Filter: {len(results)} chunks passed threshold {threshold}")

    # Prepare structured chunks and sources
    chunks = []
    sources = []
    seen_texts = set()

    for r in results:
        text = r.get("text", "")
        if text in seen_texts:
            continue
        seen_texts.add(text)

        source_file = r.get("source_file", "Unknown")
        score = r.get("score", 0.0)

        chunks.append({
            "text": text,
            "source_file": source_file,
            "score": score,
            "source_type": "Knowledge Base"
        })
        
        sources.append(SourceReference(
            filename=source_file,
            chunk_text=text[:300] + ("..." if len(text) > 300 else ""),
            relevance_score=round(score, 4),
            chunk_index=r.get("chunk_index", 0),
        ))

    # Cache result
    query_cache.put(query, (chunks, sources))

    return chunks, sources


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

    # Clear query cache to ensure the new similarity threshold is applied immediately
    from app.services.cache import query_cache
    query_cache.clear()

    # Create or load conversation
    if not conversation_id or conversation_id in ("undefined", "null", "None"):
        conversation_id = await conversation_store.create_conversation()
        is_new = True
    else:
        is_new = False


    # Handle attachment
    attachments_data = []
    if attachment_path:
        try:
            yield {"type": "info", "content": "📄 Processing attachment..."}
            result = await index_document(attachment_path, source="attachment")
            
            # Save attachment detail for message persistence
            attachments_data.append({
                "filename": result["filename"],
                "file_type": result["file_type"],
                "file_size": result["file_size"],
                "chunk_count": result["chunks"],
                "filepath": result["filepath"]
            })

            yield {
                "type": "info",
                "content": f"✅ Indexed {result['filename']} ({result['chunks']} chunks)",
            }
        except Exception as e:
            err_msg = str(e) or repr(e)
            yield {"type": "info", "content": f"⚠️ Failed to process attachment: {err_msg}"}

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
        user_msg_id = await conversation_store.add_message(
            conversation_id, "user", user_message, attachments=attachments_data, parent_id=parent_id
        )
        yield {
            "type": "user_meta",
            "message_id": user_msg_id,
            "attachments": attachments_data
        }


    # Web search
    all_source_data = []
    if web_search:
        yield {"type": "info", "content": "🌍 Searching the web..."}
        try:
            from app.services.web_search import search_web
            web_results = search_web(user_message)
            if web_results:
                all_source_data.extend(web_results)
                yield {"type": "info", "content": f"✅ Found {len(web_results)} web results"}
            else:
                yield {"type": "info", "content": "⚠️ No web results found"}
        except Exception as e:
            err_msg = str(e) or repr(e)
            logger.error(f"Web search error in pipeline: {err_msg}")
            yield {"type": "info", "content": f"⚠️ Web search failed: {err_msg}"}

    # Retrieve context from FAISS
    try:
        chunks, sources = retrieve_context(user_message)
        if chunks:
            all_source_data.extend(chunks)
    except Exception as e:
        err_msg = str(e) or repr(e)
        logger.error(f"Context retrieval error: {err_msg}")
        yield {"type": "info", "content": f"⚠️ Context retrieval failed: {err_msg}"}
        chunks, sources = [], []
    
    # Retrieve mem0 memories
    mem0_context = ""
    if config.ENABLE_MEM0:
        yield {"type": "info", "content": "🧠 Recalling relevant memories..."}
        try:
            memory_svc = get_memory_service()
            memories = await memory_svc.search(user_message, conversation_id)
            if memories:
                mem0_context = "\n".join([f"- {m}" for m in memories])
                yield {"type": "info", "content": f"✅ Recalled {len(memories)} relevant memories"}
        except Exception as e:
            import traceback
            err_msg = str(e) or repr(e)
            logger.error(f"Mem0 search error: {err_msg}")
            logger.error(traceback.format_exc())
            yield {"type": "info", "content": f"⚠️ Memory recall failed: {err_msg}"}
    
    # Format unified context with numbering [1], [2], etc.
    formatted_parts = []
    for i, src in enumerate(all_source_data, 1):
        source_label = src.get("source_type", "Source")
        name = src.get("source_file") or src.get("title") or "Unknown"
        url_part = f" (URL: {src['href']})" if src.get("href") else ""
        
        formatted_parts.append(
            f"[{i}] {source_label}: {name}{url_part}\n"
            f"Content: {src['text'] if 'text' in src else src.get('body', '')}"
        )

    full_context = "\n\n---\n\n".join(formatted_parts)

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
    
    # Final context formatting: Combine RAG docs + Web + mem0
    context_to_send = full_context
    if mem0_context:
        context_to_send += f"\n\n--- RELEVANT MEMORIES ---\n{mem0_context}\n--- END MEMORIES ---\n"

    try:
        async for token in llm.stream_chat(
            user_message=user_message,
            context=context_to_send,
            conversation_history=history[:-1],  # Exclude the message we just added
        ):
            full_response += token
            yield {"type": "token", "content": token}
    except Exception as e:
        import traceback
        logger.error(f"LLM Stream error: {e}")
        logger.error(traceback.format_exc())
        yield {"type": "error", "content": f"LLM error: {e}"}

    # Fallback for completely empty responses
    import re
    cleaned_resp = re.sub(r'[\s\u200b\ufeff\x00-\x1f]+', '', full_response)
    is_empty = not cleaned_resp
    
    if is_empty:
        logger.warning(f"⚠️  LLM returned an empty or whitespace-only response for query: '{user_message}'")
        fallback_msg = "I'm here to help! "
        if full_context:
            fallback_msg += "I found some relevant notes in your knowledge base, but I couldn't generate a specific answer. Feel free to ask a more specific question about them!"
        else:
            fallback_msg += "How can I assist you today?"
        
        full_response = fallback_msg
        yield {"type": "token", "content": fallback_msg}

    # Save assistant response
    generation_time = round(time.time() - start_time, 2)
    sources_data = [s.model_dump() for s in sources] if sources else []
    ast_msg_id = await conversation_store.add_message(
        conversation_id, "assistant", full_response, sources=sources_data, parent_id=user_msg_id, generation_time=generation_time
    )

    # Auto-title on first message using LLM
    if is_new:
        try:
            new_title = await llm.generate_title(user_message, full_response)
            await conversation_store.update_conversation_title(conversation_id, new_title)
        except Exception as e:
            logger.error(f"Failed to auto-generate title: {e}")

    # Indexed turn in Eternal Memory if enabled
    if config.ENABLE_CHAT_MEMORY:
        try:
            await index_chat_turn(user_message, full_response, conversation_id, ast_msg_id, attachments_data)
        except Exception as e:
            logger.error(f"Failed to index chat turn in memory: {e}")

    # Save turn to mem0
    if config.ENABLE_MEM0:
        try:
            memory_svc = get_memory_service()
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": full_response}
            ]
            await memory_svc.add(messages, conversation_id)
        except Exception as e:
            logger.error(f"Failed to save turn to mem0: {e}")

    yield {
        "type": "done",
        "conversation_id": conversation_id,
        "message_id": ast_msg_id,
        "generation_time": generation_time,
    }


async def index_chat_turn(user_message: str, assistant_response: str, conversation_id: str, message_id: str, attachments: list[dict]):
    """Index a conversation turn into FAISS for long-term memory."""
    vector_store = get_vector_store()
    embedding_svc = get_embedding_service()

    # Format the memory text
    attachment_names = ", ".join([a["filename"] for a in attachments]) if attachments else "None"
    memory_text = (
        f"[Chat Memory] User: {user_message}\n"
        f"(Attachments: {attachment_names})\n"
        f"Assistant: {assistant_response}"
    )

    # Embed
    embedding = embedding_svc.embed_texts([memory_text])

    # Metadata
    from datetime import datetime
    timestamp = datetime.now().strftime("%b %d, %H:%M")
    
    metadata = [{
        "text": memory_text,
        "source_file": f"Chat History ({timestamp})",
        "chunk_index": 0,
        "source": "memory",
        "conversation_id": conversation_id,
        "message_id": message_id,
        "source_type": "Chat Memory"
    }]

    # Add to FAISS
    vector_store.add(embedding, metadata)
    # Note: We don't necessarily need to save to disk every single turn to avoid I/O overhead,
    # main.py lifespan saves it on shutdown. But for safety:
    vector_store.save()
    logger.info(f"Indexed conversation turn {message_id} into Chat Memory")
