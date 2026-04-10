"""
SQLite database setup and connection management.
Uses aiosqlite for async operations.
"""

import aiosqlite
from app.config import DATABASE_PATH

DATABASE_URL = str(DATABASE_PATH)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New Chat',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content         TEXT NOT NULL,
    sources         TEXT DEFAULT '[]',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    parent_id       TEXT,
    generation_time REAL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    filepath    TEXT NOT NULL,
    file_type   TEXT,
    file_size   INTEGER,
    chunk_count INTEGER DEFAULT 0,
    indexed_at  TEXT NOT NULL DEFAULT (datetime('now')),
    source      TEXT DEFAULT 'upload'
);

CREATE INDEX IF NOT EXISTS idx_documents_filename
    ON documents(filename);
"""


async def get_db() -> aiosqlite.Connection:
    """Get an async database connection."""
    db = await aiosqlite.connect(DATABASE_URL)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    return db


async def init_db():
    """Initialize the database schema."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = await get_db()
    try:
        await db.executescript(SCHEMA_SQL)
        
        # Auto-migration for existing DBs
        try:
            await db.execute("ALTER TABLE messages ADD COLUMN parent_id TEXT;")
        except aiosqlite.OperationalError:
            pass  # column exists
            
        try:
            await db.execute("ALTER TABLE messages ADD COLUMN generation_time REAL;")
        except aiosqlite.OperationalError:
            pass  # column exists

        await db.commit()
    finally:
        await db.close()
