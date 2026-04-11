import asyncio
import shutil
import os
from pathlib import Path
import sqlite3

# Set environment to allow importing app
os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent)

async def wipe_and_rescan():
    print("--- Starting Full Index Clean and Rescan ---")
    
    # 1. Paths
    index_dir = Path("data/faiss_index")
    db_path = Path("data/rag.db")
    
    # 2. Clear FAISS files
    if index_dir.exists():
        print(f"Cleaning vector index directory: {index_dir}")
        shutil.rmtree(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Clear Database document records
    if db_path.exists():
        print(f"Wiping document records from database: {db_path}")
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("DELETE FROM documents")
            conn.commit()
            print("Database records cleared")
        except Exception as e:
            print(f"Database error: {e}")
        finally:
            conn.close()

    # 4. Trigger Rescan
    print("\nScanning for references and rebuilding index...")
    from app.services.rag_pipeline import index_references_directory
    from app.services.vector_store import get_vector_store
    from app.services.embeddings import get_embedding_service
    
    # Initialize embedding service first
    embed_svc = get_embedding_service()
    await embed_svc.initialize()
    
    # Initialize store with dimension
    v_store = get_vector_store()
    v_store.initialize(embed_svc.dim)
    
    results = await index_references_directory()
    
    print(f"\nRe-indexed {len(results)} files.")
    for res in results:
        if "error" in res:
            print(f"  [ERROR] {res['filename']}: {res['error']}")
        else:
            print(f"  [OK] {res['filename']} ({res['chunks']} chunks)")

    print("\nFull re-index complete! You can now restart your application.")

if __name__ == "__main__":
    asyncio.run(wipe_and_rescan())
