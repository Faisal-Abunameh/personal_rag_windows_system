"""
Document parser using Microsoft MarkItDown.
Converts various file formats to markdown for processing.
"""

import logging
from pathlib import Path
from typing import Optional
from markitdown import MarkItDown

import app.config as config

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

logger = logging.getLogger(__name__)

# Singleton MarkItDown instance
_md_instance: Optional[MarkItDown] = None


def get_markitdown() -> MarkItDown:
    """Get or create the singleton MarkItDown instance."""
    global _md_instance
    if _md_instance is None:
        _md_instance = MarkItDown()
    return _md_instance


def _parse_pdf_robustly(path: Path) -> str:
    """Directly extract text from PDF using PyMuPDF."""
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is not installed.")

    text = []
    try:
        with fitz.open(str(path)) as doc:
            for page in doc:
                text.append(page.get_text())
        return "\n\n".join(text)
    except Exception as e:
        logger.error(f"PyMuPDF failed to parse {path.name}: {e}")
        raise


def parse_document(file_path: str | Path) -> dict:
    """
    Parse a document file and return its content as markdown.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {ext}. "
            f"Supported: {', '.join(sorted(config.SUPPORTED_EXTENSIONS))}"
        )

    logger.info(f"Parsing document: {path.name} ({ext})")

    text = ""
    error = None

    # Priority 1: Specialized PDF parsing (PyMuPDF is fast and reliable for text)
    if ext == ".pdf" and fitz:
        try:
            text = _parse_pdf_robustly(path)
        except Exception as e:
            error = e
            logger.warning(f"PyMuPDF failed, falling back to MarkItDown: {e}")

    # Priority 2: General MarkItDown parsing
    if not text:
        md = get_markitdown()
        try:
            result = md.convert(str(path))
            text = result.text_content or ""
        except Exception as e:
            error = e
            # Log specific advice if dependencies are missing
            err_msg = str(e).lower()
            if "dependency" in err_msg or "not installed" in err_msg:
                logger.error(f"MarkItDown missing dependencies for {ext}: {e}")
            else:
                logger.error(f"MarkItDown failed to parse {path.name}: {e}")

    # Priority 3: Robust Text Fallback (Handle code, markdown, logs, and data)
    if not text:
        # Define extensions that are safe to read as plain text
        text_safe = {
            ".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm", ".log",
            ".py", ".js", ".ts", ".css", ".sql", ".yaml", ".yml", ".ini",
            ".env", ".c", ".cpp", ".h", ".cs", ".go", ".rs", ".sh"
        }
        if ext in text_safe:
            try:
                # Use a larger lookahead for encoding detection if needed
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                error = e
                logger.error(f"Plain text fallback failed for {path.name}: {e}")

    if not text and error:
        # If we failed on a binary format like Excel, explain why
        if ext in {".xlsx", ".xls", ".docx", ".doc", ".pptx", ".ppt"}:
             raise RuntimeError(
                 f"Cannot parse {path.name}. This usually requires additional libraries "
                 f"(openpyxl, python-docx, etc.) or the file is corrupted. Original error: {error}"
             )
        raise RuntimeError(f"Cannot parse {path.name}: {error}")

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
    for ext in config.SUPPORTED_EXTENSIONS:
        files.extend(dir_path.rglob(f"*{ext}"))

    logger.info(f"Found {len(files)} supported files in {dir_path}")
    return sorted(files)
