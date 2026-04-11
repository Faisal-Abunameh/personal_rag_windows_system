import asyncio
import sys
from pathlib import Path

# Add app directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.services.rag_pipeline import chat_with_rag
from app.services.embeddings import get_embedding_service
from app.services.vector_store import get_vector_store
from app.database import init_db

async def test_chat():
    print("Initializng...")
    await init_db()
    get_embedding_service().initialize_sync() # Use sync if available or assume it works
    get_vector_store().initialize(384)
    
    print("Testing chat with 'yo'...")
    async for event in chat_with_rag("yo"):
        print(f"Event: {event}")

if __name__ == "__main__":
    asyncio.run(test_chat())
