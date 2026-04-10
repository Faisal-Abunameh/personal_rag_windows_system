# 🤖 Local RAG System

A fully local **Retrieval-Augmented Generation** system with a ChatGPT-like interface, powered by **Google Gemma 4**, **FAISS** vector search, **NVIDIA NeMo Retriever** embeddings, and **MarkItDown** document parsing.

> **100% local & private** — No data leaves your machine. No authentication required.

![Architecture](https://img.shields.io/badge/Architecture-FastAPI%20%2B%20FAISS%20%2B%20Ollama-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [API Reference](#-api-reference)
- [Usage Guide](#-usage-guide)
- [Performance & Caching](#-performance--caching)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

| Feature | Description |
|---|---|
| **ChatGPT-like UI** | Premium dark-mode interface with streaming responses, markdown rendering, and code highlighting |
| **RAG Pipeline** | Semantic search over your documents using FAISS vector similarity |
| **Gemma 4 LLM** | Google's state-of-the-art model running locally via Ollama |
| **NeMo Retriever** | NVIDIA's embedding model for high-quality vector representations (with sentence-transformers fallback) |
| **MarkItDown** | Microsoft's document parser supporting PDF, DOCX, PPTX, XLSX, HTML, CSV, JSON, and more |
| **Semantic Chunking** | Intelligent text splitting based on meaning, not arbitrary character counts |
| **Conversation Memory** | Persistent chat history with SQLite storage |
| **File Attachments** | Upload and index documents directly through the chat interface |
| **References Directory** | Auto-indexes documents from `Desktop/references/` |
| **Caching** | LRU caches for embeddings and query results for fast responses |
| **Source Citations** | Every answer shows which documents were referenced |
| **No Authentication** | Zero-config, just start and use |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Frontend (HTML/CSS/JS)                     │
│  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌──────────────┐   │
│  │ Sidebar │  │ Chat Area│  │ Upload │  │ Markdown/SSE │   │
│  └────┬────┘  └────┬─────┘  └───┬────┘  └──────┬───────┘   │
└───────┼────────────┼────────────┼───────────────┼────────────┘
        │            │            │               │
        ▼            ▼            ▼               ▼
┌──────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                             │
│  ┌─────────────┐  ┌────────────┐  ┌───────────────────────┐ │
│  │ Conversations│  │    Chat    │  │     Documents         │ │
│  │    CRUD      │  │ (Streaming)│  │ (Upload/Index/Refs)   │ │
│  └──────┬──────┘  └─────┬──────┘  └──────────┬────────────┘ │
│         │               │                    │              │
│         ▼               ▼                    ▼              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  RAG Pipeline                           │ │
│  │  MarkItDown → Semantic Chunker → Embeddings → FAISS    │ │
│  │                                                         │ │
│  │  Query → Embed → Search Top-10 → Context → LLM Stream  │ │
│  └──────────────────────────┬──────────────────────────────┘ │
└─────────────────────────────┼────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────────┐
        │  SQLite  │   │  FAISS   │   │  Ollama      │
        │  (Chats) │   │  (Index) │   │  (Gemma 4)   │
        └──────────┘   └──────────┘   └──────────────┘
```

---

## 📦 Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| **Python** | 3.10+ | Runtime |
| **Ollama** | Latest | Local LLM hosting |
| **Docker** (optional) | Latest | NeMo Retriever NIM embeddings |
| **NVIDIA GPU** (optional) | — | Required for NeMo; app works with CPU otherwise |

### Install Ollama

1. Download from [ollama.com/download](https://ollama.com/download)
2. Install and start Ollama
3. Pull Gemma 4:
   ```bash
   ollama pull gemma4
   ```

### (Optional) NeMo Retriever NIM

If you have Docker + NVIDIA GPU:
```bash
# Pull and run the NeMo Retriever embedding NIM
docker run --gpus all -p 8000:8000 \
    nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:latest
```

> Without Docker/GPU, the system automatically falls back to `sentence-transformers` (all-MiniLM-L6-v2) — still excellent quality.

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd personal_rag_windows_system

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure Ollama is running with Gemma 4
ollama pull gemma4

# 5. Launch the app
python run.py
```

The app will:
- ✅ Check Python version
- ✅ Create required directories
- ✅ Check Ollama availability
- ✅ Pull Gemma 4 if not present
- ✅ Initialize databases and FAISS index
- ✅ Scan and index your references directory
- ✅ Open `http://localhost:8000` in your browser

### Using the Desktop Shortcut

Double-click `Local LLM.lnk` in the project directory to launch the app.

---

## ⚙️ Configuration

All settings are in `app/config.py` and can be overridden via environment variables or a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `LLM_MODEL` | `gemma4` | Ollama model name |
| `LLM_TEMPERATURE` | `0.7` | Response creativity (0.0–1.0) |
| `LLM_MAX_TOKENS` | `4096` | Max response length |
| `REFERENCES_DIR` | `~/Desktop/references` | Path to reference documents |
| `TOP_K_CHUNKS` | `10` | Number of chunks retrieved per query |
| `CHUNK_SIMILARITY_THRESHOLD` | `0.65` | Semantic chunking split threshold |
| `MAX_CHUNK_SIZE` | `1000` | Maximum characters per chunk |
| `MIN_CHUNK_SIZE` | `100` | Minimum characters per chunk |
| `MAX_CONVERSATION_HISTORY` | `20` | Messages included in LLM context |
| `EMBEDDING_CACHE_SIZE` | `2048` | LRU cache size for embeddings |
| `NEMO_RETRIEVER_URL` | `http://localhost:8000/v1` | NeMo NIM endpoint |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

### Example `.env` file

```env
LLM_MODEL=gemma4:12b
LLM_TEMPERATURE=0.5
TOP_K_CHUNKS=15
REFERENCES_DIR=D:\my_documents\references
```

---

## 📁 Project Structure

```
personal_rag_windows_system/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app with lifespan events
│   ├── config.py               # Central configuration
│   ├── database.py             # SQLite schema and connection
│   ├── models.py               # Pydantic request/response models
│   ├── routers/
│   │   ├── chat.py             # POST /api/chat — SSE streaming
│   │   ├── conversations.py    # CRUD for conversations
│   │   ├── documents.py        # Upload and index documents
│   │   └── references.py       # References directory management
│   └── services/
│       ├── cache.py            # LRU caching (embeddings + queries)
│       ├── chunker.py          # Semantic text chunking
│       ├── conversation_store.py # SQLite conversation persistence
│       ├── document_parser.py  # MarkItDown document conversion
│       ├── embeddings.py       # NeMo / sentence-transformers
│       ├── llm.py              # Ollama streaming client
│       ├── rag_pipeline.py     # Full RAG orchestration
│       └── vector_store.py     # FAISS index management
├── static/
│   ├── index.html              # Single-page application
│   ├── css/styles.css          # Premium dark-mode styling
│   └── js/
│       ├── app.js              # State management & API client
│       ├── chat.js             # Chat UI with SSE streaming
│       ├── sidebar.js          # Conversation sidebar
│       ├── upload.js           # File upload & drag-and-drop
│       └── markdown.js         # Markdown rendering
├── data/
│   ├── faiss_index/            # Persisted FAISS vectors
│   ├── uploads/                # Uploaded documents
│   └── rag.db                  # SQLite database
├── requirements.txt            # Python dependencies
├── run.py                      # App launcher
├── README.md                   # This file
└── Local LLM.lnk              # Windows shortcut
```

---

## 📡 API Reference

### Chat

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Send message (new conversation), returns SSE stream |
| `POST` | `/api/chat/{id}` | Continue conversation, returns SSE stream |

**Request** (multipart/form-data):
- `message` (required): User message text
- `attachment` (optional): File to upload and index

**SSE Events**:
```json
{"type": "info", "content": "Processing attachment..."}
{"type": "sources", "sources": [{"filename": "...", "chunk_text": "...", "relevance_score": 0.85}]}
{"type": "token", "content": "Hello"}
{"type": "done", "conversation_id": "uuid"}
```

### Conversations

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/conversations` | List all conversations |
| `GET` | `/api/conversations/{id}` | Get conversation with messages |
| `PUT` | `/api/conversations/{id}` | Rename conversation |
| `DELETE` | `/api/conversations/{id}` | Delete conversation |

### Documents

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/documents/upload` | Upload and index a document |
| `GET` | `/api/documents` | List indexed documents |
| `GET` | `/api/documents/status` | FAISS index statistics |
| `POST` | `/api/documents/reindex` | Force re-index references |

### References

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/references` | List files in references dir |
| `POST` | `/api/references/scan` | Trigger reference scanning |
| `GET` | `/api/references/status` | Indexing status |

### System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | System health (LLM, embeddings, index) |

---

## 💡 Usage Guide

### 1. Chat with your documents

1. Place documents in `Desktop/references/` (or your configured dir)
2. Start the app — they'll be auto-indexed on startup
3. Ask questions! The system retrieves relevant chunks and feeds them to Gemma 4

### 2. Upload documents via chat

Click the 📎 button or drag-and-drop files onto the chat. The system will:
1. Parse the document with MarkItDown
2. Split into semantic chunks
3. Embed and store in FAISS
4. Use for answering your question

### 3. Supported file types

PDF, DOCX, PPTX, XLSX, HTML, CSV, JSON, XML, TXT, MD, PNG, JPG, ZIP, EPUB

### 4. Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Enter` | Send message |
| `Shift+Enter` | New line in message |
| `Ctrl+N` | New chat |
| `Escape` | Stop generation |

---

## ⚡ Performance & Caching

The system uses multiple optimization layers:

1. **Embedding Cache** — LRU cache (2048 entries) avoids re-computing embeddings for previously seen text
2. **Query Cache** — Recent query results cached with 5-minute TTL
3. **FAISS IndexFlatIP** — In-memory inner-product search for fastest retrieval
4. **WAL Mode SQLite** — Concurrent reads for conversation history
5. **Batch Embedding** — Sentences are embedded in batches (64 at a time)
6. **Streaming SSE** — Tokens streamed as they're generated, no waiting for full response
7. **Persistent Index** — FAISS index saved to disk, loaded on startup (no re-indexing)

---

## 🔧 Troubleshooting

### Ollama not found

```
⚠️ Ollama is not running!
```

**Fix**: Install Ollama from [ollama.com/download](https://ollama.com/download), start it, and run `ollama pull gemma4`.

### Model not available

```
Model 'gemma4' not found
```

**Fix**: Run `ollama pull gemma4` in your terminal. For a smaller model: `ollama pull gemma4:4b`.

### Import errors

```
❌ Dependencies not installed
```

**Fix**: Activate your virtual environment and run `pip install -r requirements.txt`.

### Slow first response

The first query after startup may be slow because:
- The embedding model needs to warm up (cached after first use)
- Ollama may need to load the model into memory

Subsequent queries will be much faster due to caching.

### Port already in use

**Fix**: Set a different port: `set PORT=8001` then `python run.py`

---

## 🔒 Privacy

- **100% local** — All processing happens on your machine
- **No external API calls** — LLM and embeddings run locally
- **No telemetry** — No usage data is collected
- **No authentication** — Direct access, zero config

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.