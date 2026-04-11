import sqlite3
import os

db_path = 'data/rag.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT content FROM messages WHERE content LIKE '%PDF%' LIMIT 2")
    print("Sample messages with PDF markers:", cursor.fetchall())
    
    # Check chunks table (this is what the RAG uses)
    cursor = conn.execute("SELECT content FROM chunks WHERE content LIKE '%PDF%' LIMIT 2")
    print("\nSample chunks with PDF markers:", cursor.fetchall())
    
    conn.close()
else:
    print(f"DB not found at {db_path}")
