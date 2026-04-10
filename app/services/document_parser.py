"""
Document parser using Microsoft MarkItDown.
Converts various file formats to markdown for processing.
"""

import logging
from pathlib import Path
from typing import Optional
from markitdown import MarkItDown

from app.config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)

# Singleton MarkItDown instance
_md_instance: Optional[MarkItDown] = None


def get_markitdown() -> MarkItDown:
    """Get or create the singleton MarkItDown instance."""
    global _md_instance
    if _md_instance is None:
        _md_instance = MarkItDown()
    return _md_instance


def parse_document(file_path: str | Path) -> dict:
    """
    Parse a document file and return its content as markdown.

    Returns:
        dict with keys: text, filename, filepath, file_type, file_size
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    logger.info(f"Parsing document: {path.name}")

    md = get_markitdown()
    try:
        result = md.convert(str(path))
        text = result.text_content or ""
    except Exception as e:
        logger.error(f"Failed to parse {path.name}: {e}")
        # Fallback: try reading as plain text
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            raise RuntimeError(f"Cannot parse {path.name}: {e}")

    return {
        "text": text.strip(),
        "filename": path.name,
        "filepath": str(path),
        "file_type": ext,
        "file_size": path.stat().st_size,
    }


def parse_documents_batch(file_paths: list[str | Path]) -> list[dict]:
    """Parse multiple documents and return list of results."""
    results = []
    for fp in file_paths:
        try:
            results.append(parse_document(fp))
        except Exception as e:
            logger.warning(f"Skipping {fp}: {e}")
            results.append({
                "text": "",
                "filename": Path(fp).name,
                "filepath": str(fp),
                "file_type": Path(fp).suffix.lower(),
                "file_size": 0,
                "error": str(e),
            })
    return results


def scan_directory(directory: str | Path) -> list[Path]:
    """Scan a directory for supported documents."""
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.warning(f"Directory does not exist: {dir_path}")
        return []

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(dir_path.rglob(f"*{ext}"))

    logger.info(f"Found {len(files)} supported files in {dir_path}")
    return sorted(files)
