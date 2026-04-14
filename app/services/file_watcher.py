"""
Real-time file system monitoring for the references directory.
Automatically indexes added/modified files and removes deleted ones.
"""

import logging
import time
import asyncio
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import app.config as config
from app.services.rag_pipeline import index_document
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

class ReferenceHandler(FileSystemEventHandler):
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self._last_processed = {}  # path -> timestamp to debounce

    def on_created(self, event):
        if event.is_directory:
            return
        self._handle_change(event.src_path, "created")

    def on_modified(self, event):
        if event.is_directory:
            return
        self._handle_change(event.src_path, "modified")

    def on_deleted(self, event):
        if event.is_directory:
            return
        self._handle_change(event.src_path, "deleted")

    def on_moved(self, event):
        if event.is_directory:
            return
        logger.info(f"🚚 Reference moved: {Path(event.src_path).name} -> {Path(event.dest_path).name}")
        self._handle_change(event.src_path, "deleted")
        self._handle_change(event.dest_path, "created")

    def _handle_change(self, file_path_str: str, event_type: str):
        path = Path(file_path_str)
        
        # Check extension
        if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            return

        # For "deleted" from moved/deleted events, the file doesn't exist anymore
        # but we still need the relative path to remove it from the index.
        try:
            rel_path = str(path.relative_to(config.REFERENCES_DIR))
        except ValueError:
            # Path might be outside references dir if moved out
            rel_path = path.name

        # Debounce (some OS events fire twice)
        now = time.time()
        if rel_path in self._last_processed and (now - self._last_processed[rel_path]) < 1.0:
            return
        self._last_processed[rel_path] = now

        if event_type == "deleted":
            logger.info(f"🗑️ Reference deleted: {rel_path}")
            try:
                get_vector_store().remove_by_source(rel_path)
                get_vector_store().save()
            except Exception as e:
                logger.error(f"Failed to remove {rel_path} from index: {e}")
        else:
            logger.info(f"🔄 Reference {event_type}: {rel_path}")
            # Indexing is async, schedule on the main loop
            asyncio.run_coroutine_threadsafe(
                self._async_index(path, rel_path), 
                self.loop
            )

    async def _async_index(self, path: Path, rel_path: str):
        """Indexes a document with retries for OS locking."""
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Wait for the file to be fully written/released by OS
                await asyncio.sleep(retry_delay * (attempt + 1))
                
                if not path.exists():
                    logger.warning(f"File vanished before indexing: {rel_path}")
                    return
                
                # Try to open it to check if it's locked
                try:
                    with open(path, "rb"):
                        pass
                except (IOError, OSError):
                    logger.debug(f"File locked, retrying {attempt+1}/{max_retries}: {rel_path}")
                    continue

                await index_document(path, source="reference", custom_source_id=rel_path)
                return # Success
            except Exception as e:
                err_msg = str(e) or repr(e)
                if "Dimension mismatch" in err_msg:
                    logger.error(f"Auto-index failed for {rel_path}: {err_msg}")
                    return # Don't retry for fatal dimension errors
                
                if attempt < max_retries - 1:
                    logger.debug(f"Retry {attempt+1} indexing {rel_path} due to error: {err_msg}")
                    continue
                logger.error(f"Auto-index failed for {rel_path} after {max_retries} attempts: {err_msg}")

class ReferenceWatcher:
    def __init__(self):
        self.observer = None
        self.handler = None

    def start(self, loop: asyncio.AbstractEventLoop):
        if not config.REFERENCES_DIR.exists():
            config.REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
            
        self.handler = ReferenceHandler(loop)
        self.observer = Observer()
        self.observer.schedule(self.handler, str(config.REFERENCES_DIR), recursive=True)
        self.observer.start()
        logger.info(f"👀 Reference watcher started on {config.REFERENCES_DIR}")

    def stop(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("🛑 Reference watcher stopped")
