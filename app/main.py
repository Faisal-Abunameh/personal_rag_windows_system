"""
FastAPI application entry point.
Sets up routes, middleware, static files, and startup/shutdown events.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import STATIC_DIR, REFERENCES_DIR, DATA_DIR
from app.database import init_db
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.llm import get_llm_client
from app.routers import chat, conversations, documents, references

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    logger.info("=" * 60)
    logger.info("  Local RAG System — Starting Up")
    logger.info("=" * 60)

    # Create directories
    REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize database
    await init_db()
    logger.info("✅ Database initialized")

    # Initialize embedding service
    embedding_svc = get_embedding_service()
    await embedding_svc.initialize()
    logger.info(f"✅ Embeddings ready ({embedding_svc.mode}: {embedding_svc.model_name})")

    # Initialize FAISS vector store
    vector_store = get_vector_store()
    vector_store.initialize(dim=embedding_svc.dim)
    logger.info(f"✅ FAISS index ready ({vector_store.total_vectors} vectors)")

    # Check LLM availability
    llm = get_llm_client()
    model_ok = await llm.check_health()
    if model_ok:
        logger.info(f"✅ LLM ready ({llm.model_name})")
    else:
        logger.warning(
            f"⚠️  LLM not available. Make sure Ollama is running and "
            f"'{llm.model_name}' is pulled."
        )

    # Index references directory (background-safe)
    try:
        from app.services.rag_pipeline import index_references_directory
        results = await index_references_directory()
        if results:
            logger.info(f"✅ Indexed {len(results)} reference documents")
    except Exception as e:
        logger.warning(f"⚠️  Reference indexing skipped: {e}")

    logger.info("=" * 60)
    logger.info("  🚀 Ready at http://localhost:8000")
    logger.info("=" * 60)

    yield

    # Shutdown
    vector_store.save()
    logger.info("FAISS index saved. Goodbye!")


# Create FastAPI app
app = FastAPI(
    title="Local RAG System",
    description="A local RAG system with Gemma 4, FAISS, and NeMo Retriever",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all (no auth)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Include API routers
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(documents.router)
app.include_router(references.router)


@app.get("/")
async def root():
    """Serve the main page."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
async def system_status():
    """Get system status including LLM, embeddings, and index info."""
    from app.services.embeddings import get_embedding_service
    from app.services.vector_store import get_vector_store
    from app.services.llm import get_llm_client
    from app.database import get_db

    es = get_embedding_service()
    vs = get_vector_store()
    llm = get_llm_client()
    model_ok = await llm.check_health()

    db = await get_db()
    try:
        cursor = await db.execute("SELECT COUNT(*) as cnt FROM documents")
        doc_count = (await cursor.fetchone())["cnt"]
    finally:
        await db.close()

    return {
        "ollama_available": llm.is_available,
        "model_loaded": model_ok,
        "model_name": llm.model_name,
        "embedding_mode": es.mode,
        "embedding_model": es.model_name,
        "total_documents": doc_count,
        "total_chunks": vs.total_vectors,
        "references_dir": str(REFERENCES_DIR),
    }
