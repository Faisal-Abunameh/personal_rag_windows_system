
import os
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent))

import asyncio
from app.services.rag_pipeline import retrieve_context
from app.config import CHUNK_SIMILARITY_THRESHOLD

async def debug_rag():
    query = "Tell me a long story about a robot."
    print(f"DEBUG: Query: {query}")
    print(f"DEBUG: Threshold: {CHUNK_SIMILARITY_THRESHOLD}")
    
    context, sources = retrieve_context(query)
    
    print(f"DEBUG: Results Found: {len(sources)}")
    for s in sources:
        print(f"  - {s.filename}: {s.relevance_score}")

if __name__ == "__main__":
    asyncio.run(debug_rag())
