import sys
import os
import time
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import REFERENCES_DIR
from app.services.vector_store import get_vector_store

def check_index(source_name):
    vs = get_vector_store()
    vs.initialize(384) # Assuming dim 384 for MiniLM
    sources = vs.get_indexed_sources()
    print(f"Indexed sources: {sources}")
    return source_name in sources

def main():
    test_file = REFERENCES_DIR / "test_watcher.txt"
    moved_file = REFERENCES_DIR / "test_watcher_moved.txt"
    
    print(f"--- Verification Test ---")
    print(f"References dir: {REFERENCES_DIR}")
    
    # 1. Create file
    print("\n1. Creating test file...")
    test_file.write_text("This is a test document for the file watcher logic.")
    print("Wait for watcher...")
    time.sleep(2)
    
    # 2. Rename file
    print("\n2. Renaming test file...")
    if test_file.exists():
        test_file.rename(moved_file)
    print("Wait for watcher...")
    time.sleep(2)
    
    # 3. Delete file
    print("\n3. Deleting test file...")
    if moved_file.exists():
        moved_file.unlink()
    print("Wait for watcher...")
    time.sleep(2)
    
    print("\nVerification complete. Check application logs for emoji indicators (🚚, 🗑️, 🔄).")

if __name__ == "__main__":
    main()
