import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import app.config as config
from app.services.memory import get_memory_service

# Configure root logger
logging.basicConfig(level=logging.DEBUG)

async def debug_mem0():
    print("--- Starting mem0 debug ---")
    memory_svc = get_memory_service()
    
    print("Initializing...")
    try:
        await memory_svc.initialize()
    except Exception as e:
        print(f"FAILED INITIALIZE: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Searching...")
    try:
        results = await memory_svc.search("test query", "debug-chat")
        print(f"SEARCH RESULTS: {results}")
    except Exception as e:
        print(f"FAILED SEARCH: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_mem0())
