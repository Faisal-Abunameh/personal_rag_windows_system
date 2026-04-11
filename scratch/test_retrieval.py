import asyncio
from app.services.rag_pipeline import retrieve_context
from app.services.vector_store import get_vector_store
from app.services.embeddings import get_embedding_service
import os

# Set up logging to see what's happening
import logging
logging.basicConfig(level=logging.INFO)

async def test_retrieval():
    # Initialize services
    embedding_svc = get_embedding_service()
    await embedding_svc.initialize()
    
    vector_store = get_vector_store()
    vector_store.initialize(embedding_svc.dim) 
    
    queries = [
        "summarize my statement of purpose",
        "what is in my etf excel sheet?",
        "Statement of Purpose",
        "ETF watchlist"
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        chunks, sources = retrieve_context(q)
        print(f"Results found: {len(chunks)}")
        for i, c in enumerate(chunks[:3]):
            print(f"  [{i+1}] Score: {c['score']:.4f} | Source: {c['source_file']}")
            print(f"      Text: {c['text'][:100]}...")

if __name__ == "__main__":
    asyncio.run(test_retrieval())
