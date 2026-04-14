"""
References directory management endpoints.
"""

import logging
from pathlib import Path

from fastapi import APIRouter

import app.config as config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/references", tags=["references"])


@router.get("")
async def list_references():
    """List all files in the references directory."""
    config.REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    files = []
    for f in sorted(config.REFERENCES_DIR.rglob("*")):
        if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS:
            files.append({
                "filename": f.name,
                "filepath": str(f),
                "file_type": f.suffix.lower(),
                "file_size": f.stat().st_size,
            })

    return {
        "directory": str(config.REFERENCES_DIR),
        "file_count": len(files),
        "files": files,
    }


@router.post("/scan")
async def scan_references():
    """Trigger a scan and index of the references directory."""
    from app.services.rag_pipeline import index_references_directory
    results = await index_references_directory()
    return {
        "directory": str(config.REFERENCES_DIR),
        "indexed": len(results),
        "details": results,
    }


@router.get("/status")
async def references_status():
    """Get indexing status for the references directory."""
    from app.services.vector_store import get_vector_store
    vs = get_vector_store()

    config.REFERENCES_DIR.mkdir(parents=True, exist_ok=True)

    total_files = sum(
        1 for f in config.REFERENCES_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )
    indexed_sources = vs.get_indexed_sources()
    ref_indexed = [s for s in indexed_sources if s in [f.name for f in config.REFERENCES_DIR.rglob("*")]]

    return {
        "directory": str(config.REFERENCES_DIR),
        "total_files": total_files,
        "indexed_files": len(ref_indexed),
        "pending": total_files - len(ref_indexed),
    }
