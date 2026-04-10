"""
Document upload and indexing API endpoints.
"""

import logging
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.config import UPLOADS_DIR
from app.services.rag_pipeline import index_document
from app.services.vector_store import get_vector_store
from app.services.embeddings import get_embedding_service
from app.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = UPLOADS_DIR / unique_name

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = await index_document(str(file_path), source="upload")
        return result
    except Exception as e:
        logger.error(f"Failed to index upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def list_documents():
    """List all indexed documents."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT id, filename, filepath, file_type, file_size, chunk_count, indexed_at, source "
            "FROM documents ORDER BY indexed_at DESC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


@router.get("/status")
async def index_status():
    """Get FAISS index statistics."""
    vs = get_vector_store()
    es = get_embedding_service()

    db = await get_db()
    try:
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM documents")
        doc_count = (await cursor.fetchone())["cnt"]
    finally:
        await db.close()

    return {
        "total_documents": doc_count,
        "total_chunks": vs.total_vectors,
        "index_size_bytes": vs.index_size_bytes,
        "embedding_model": es.model_name,
        "embedding_dim": es.dim,
    }


@router.post("/reindex")
async def force_reindex():
    """Force re-index all referenced documents."""
    from app.services.rag_pipeline import index_references_directory
    results = await index_references_directory()
    return {"indexed": len(results), "details": results}


@router.get("/{doc_id}/chunks")
async def get_document_chunks(doc_id: str):
    """Retrieve all chunks for a specific document."""
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT filename FROM documents WHERE id = ?", (doc_id,)
        )
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Document not found")
        
        filename = row["filename"]
        vs = get_vector_store()
        chunks = vs.get_chunks_by_source(filename)
        return {"filename": filename, "chunks": chunks}
    finally:
        await db.close()
