"""
FastAPI application entry point.
Sets up routes, middleware, static files, and startup/shutdown events.
"""

import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import app.config as config
from app.database import init_db
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store
from app.services.llm import get_llm_client
from app.routers import chat, conversations, documents, references, settings
from app.services.file_watcher import ReferenceWatcher

# Global watcher instance
watcher = ReferenceWatcher()

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
    config.REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize database
    await init_db()
    logger.info("✅ Database initialized")

    # Initialize embedding service
    embedding_svc = get_embedding_service()
    await embedding_svc.initialize()
    logger.info(f"✅ Embeddings ready ({embedding_svc.mode}: {embedding_svc.model_name}) on {config.DEVICE}")

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

    # Index references directory (Full re-index on every start)
    try:
        from app.services.rag_pipeline import reindex_all_references
        reindex_data = await reindex_all_references()
        results = reindex_data.get("results", [])
        if results:
            logger.info(f"✅ Successfully re-indexed {len(results)} reference documents")
    except Exception as e:
        logger.warning(f"⚠️  Reference indexing skipped: {e}")

    logger.info("=" * 60)
    logger.info("  🚀 Ready at http://localhost:8000")
    logger.info("=" * 60)

    # Start file watcher for real-time indexing
    loop = asyncio.get_running_loop()
    watcher.start(loop)

    yield

    # Shutdown
    watcher.stop()
    vector_store.save()
    logger.info("FAISS index saved. Goodbye!")


# Create FastAPI app
app = FastAPI(
    title="Local RAG System",
    description="A local RAG system using Ollama models and FAISS",
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
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

# Include API routers
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(documents.router)
app.include_router(references.router)
app.include_router(settings.router)


@app.get("/")
async def root():
    """Serve the main page."""
    return FileResponse(str(config.STATIC_DIR / "index.html"))


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
        "references_dir": str(config.REFERENCES_DIR),
        "device": config.DEVICE,
    }


@app.get("/api/models")
async def list_models():
    """List all models available in Ollama."""
    from app.services.llm import get_llm_client

    llm = get_llm_client()
    models = await llm.list_available_models()

    return {
        "models": models,
        "current_model": llm.model_name,
    }


@app.post("/api/models/switch")
async def switch_model(body: dict):
    """Switch the active LLM model."""
    from app.services.llm import get_llm_client

    model_name = body.get("model")
    if not model_name:
        return {"error": "Model name is required"}, 400

    llm = get_llm_client()
    llm.set_model(model_name)

    # Verify the model is available
    model_ok = await llm.check_health()

    return {
        "success": True,
        "model_name": llm.model_name,
        "model_loaded": model_ok,
    }


@app.get("/api/embeddings/models")
async def list_embedding_models():
    """List all available embedding model options (Ollama models + sentence-transformers)."""
    from app.services.llm import get_llm_client
    from app.services.embeddings import get_embedding_service

    llm = get_llm_client()
    es = get_embedding_service()

    # Get all Ollama models (any can be used for embeddings)
    ollama_models = await llm.list_available_models()

    options = []
    for m in ollama_models:
        options.append({
            "name": m["name"],
            "mode": "ollama",
            "size": m.get("size", 0),
            "parameter_size": m.get("parameter_size", ""),
            "family": m.get("family", ""),
        })

    # Add sentence-transformers fallback option
    options.append({
        "name": "all-MiniLM-L6-v2",
        "mode": "sentence-transformers",
        "size": 90_000_000,  # ~90 MB
        "parameter_size": "22M",
        "family": "sentence-transformers",
    })

    return {
        "models": options,
        "current_model": es.model_name,
        "current_mode": es.mode,
        "current_dim": es.dim,
    }


@app.post("/api/embeddings/switch")
async def switch_embedding_model(body: dict):
    """Switch the active embedding model. May trigger re-indexing if dimensions change."""
    from app.services.embeddings import get_embedding_service
    from app.services.vector_store import get_vector_store

    model_name = body.get("model")
    mode = body.get("mode", "ollama")
    if not model_name:
        return {"error": "Model name is required"}, 400

    es = get_embedding_service()
    result = await es.switch_model(model_name, mode=mode)

    if not result.get("success"):
        return {"error": result.get("error", "Switch failed")}, 500

    # If dimensions changed, rebuild the FAISS index
    if result.get("dim_changed"):
        vs = get_vector_store()
        vs.initialize(dim=result["dim"])
        vs.save()
        logger.info(f"FAISS index rebuilt with new dim={result['dim']}")

    return result

