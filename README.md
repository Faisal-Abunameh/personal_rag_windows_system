# 🤖 Local RAG System

A fully local **Retrieval-Augmented Generation** system with a ChatGPT-like interface. Works with **any Ollama model** (Gemma 4, LLaMA 3, Mistral, Phi, Qwen, etc.), **FAISS** vector search, **NVIDIA NeMo Retriever** embeddings, and **MarkItDown** document parsing.

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
| **Any Ollama Model** | Supports any model from the Ollama library — Gemma 4, LLaMA 3, Mistral, Phi, Qwen, and more |
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
│                    Frontend (HTML/CSS/JS)                    │
│  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌──────────────┐     │
│  │ Sidebar │  │ Chat Area│  │ Upload │  │ Markdown/SSE │     │
│  └────┬────┘  └────┬─────┘  └───┬────┘  └──────┬───────┘     │
└───────┼────────────┼────────────┼───────────────┼────────────┘
        │            │            │               │
        ▼            ▼            ▼               ▼
┌──────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                            │
│  ┌─────────────┐  ┌────────────┐  ┌───────────────────────┐  │
│  │Conversations│  │    Chat    │  │     Documents         │  │
│  │   CRUD      │  │(Streaming) │  │ (Upload/Index/Refs)   │  │
│  └──────┬──────┘  └─────┬──────┘  └──────────┬────────────┘  │
│         │               │                    │               │
│         ▼               ▼                    ▼               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  RAG Pipeline                           │ │
│  │  MarkItDown → Semantic Chunker → Embeddings → FAISS     │ │
│  │                                                         │ │
│  │  Query → Embed → Search Top-10 → Context → LLM Stream   │ │
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

## 📦 Prerequisites — What You Need Before Starting

Before running the app, make sure you have the following installed on your system.

### Step 1: Python 3.10 or later (REQUIRED)

Download and install Python from [python.org/downloads](https://www.python.org/downloads/).

> ⚠️ During installation, **check the box** that says "Add Python to PATH".

Verify your installation:
```bash
python --version
# Expected: Python 3.10.x or higher
```

---

### Step 2: Git (REQUIRED)

Download and install Git from [git-scm.com/downloads](https://git-scm.com/downloads).

Verify:
```bash
git --version
```

---

### Step 3: Ollama — Local LLM Runtime (REQUIRED)

Ollama runs any LLM locally on your machine. This project supports **any model** from the [Ollama library](https://ollama.com/library).

1. **Download** Ollama from [ollama.com/download](https://ollama.com/download)
2. **Install** by running the downloaded installer
3. **Start Ollama** — it runs as a system service automatically after install
4. **Verify** it's running:
   ```bash
   # In a terminal / PowerShell:
   ollama --version
   ```
5. **Pull a model** — pick any model you want from the table below:
   ```bash
   # Recommended default:
   ollama pull gemma4

   # Or pick any other model:
   ollama pull llama3.2
   ollama pull mistral
   ollama pull phi4
   ollama pull qwen3
   ```

   **Popular models:**

   | Model | Command | Size | Notes |
   |---|---|---|---|
   | Gemma 4 | `ollama pull gemma4` | ~5–17 GB | Google's latest, great quality |
   | Gemma 4 (small) | `ollama pull gemma4:4b` | ~3 GB | Fast, works on 8 GB RAM |
   | LLaMA 3.2 | `ollama pull llama3.2` | ~2 GB | Meta's compact model |
   | Mistral | `ollama pull mistral` | ~4 GB | Excellent for general tasks |
   | Phi 4 | `ollama pull phi4` | ~9 GB | Microsoft's reasoning model |
   | Qwen 3 | `ollama pull qwen3` | ~5 GB | Alibaba's multilingual model |
   | DeepSeek R1 | `ollama pull deepseek-r1` | ~4–8 GB | Strong reasoning |

   > 💡 Browse all models at [ollama.com/library](https://ollama.com/library)

6. **Verify** the model is available:
   ```bash
   ollama list
   # You should see your model in the list
   ```

---

### Step 4 (OPTIONAL): Docker + NVIDIA GPU — For NeMo Retriever Embeddings

This is **completely optional**. If you skip this, the system uses `sentence-transformers` (all-MiniLM-L6-v2) for embeddings, which works great on CPU.

If you have an NVIDIA GPU and want premium embeddings:
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
3. Pull and run the NeMo Retriever NIM:
   ```bash
   docker run --gpus all -p 8080:8080 \
       nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2:latest
   ```

---

## 🚀 Quick Start — Step-by-Step Setup

Follow these steps **in order** to get the app running.

### Step 1: Clone the Repository

```bash
git clone <repo-url>
cd personal_rag_windows_system
```

### Step 2: Create a Python Virtual Environment

This isolates the app's dependencies from your system Python.

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> 💡 If you get an execution policy error, run:  
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs FastAPI, FAISS, sentence-transformers, MarkItDown, and all other dependencies. It may take a few minutes the first time.

### Step 4: Ensure Ollama is Running with Your Model

Open a **separate terminal** and verify:
```bash
ollama list
# Should show your chosen model in the list
```

If not, pull it now:
```bash
# Default (Gemma 4):
ollama pull gemma4

# Or any other model:
ollama pull llama3.2
```

### Step 5: (Optional) Add Reference Documents

Place any documents you want the system to know about into:
```
C:\Users\<YourUsername>\Desktop\references\
```

Supported formats: **PDF, DOCX, PPTX, XLSX, HTML, CSV, JSON, XML, TXT, MD, PNG, JPG, ZIP, EPUB**

The app will automatically scan and index this folder on startup.

### Step 6: Launch the App

```bash
python run.py
```

**What happens on launch:**
1. ✅ Checks your Python version
2. ✅ Creates required data directories
3. ✅ Creates `Desktop/references/` if it doesn't exist
4. ✅ Checks Ollama availability and pulls the configured model if needed
5. ✅ Initializes SQLite database
6. ✅ Loads the embedding model (first run downloads ~90 MB)
7. ✅ Initializes the FAISS vector index
8. ✅ Scans and indexes your references directory
9. ✅ Opens `http://localhost:8000` in your browser automatically

### Step 7: Start Chatting!

The app should open in your browser. If not, navigate to:
```
http://localhost:8000
```

### Alternative: Use the Desktop Shortcut

Double-click **`Local LLM.lnk`** in the project directory to launch the app without a terminal.

---

## ⚙️ Configuration

All settings are in `app/config.py` and can be overridden via environment variables or a `.env` file.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `LLM_MODEL` | `gemma4` | Ollama model name (any model from `ollama list`) |
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
# Switch to a different model:
LLM_MODEL=llama3.2
LLM_TEMPERATURE=0.5
TOP_K_CHUNKS=5
REFERENCES_DIR=D:\my_documents\references
```

### Switching Models

To use a different model, just:
1. Pull it: `ollama pull <model-name>`
2. Set it in `.env`: `LLM_MODEL=<model-name>`
3. Restart the app

The UI automatically displays the active model name.

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

**Fix**: Install Ollama from [ollama.com/download](https://ollama.com/download), start it, and run `ollama pull <your-model>`.

### Model not available

```
Model 'gemma4' not found
```

**Fix**: Run `ollama pull <model-name>` in your terminal. Set `LLM_MODEL` in `.env` to match. Check available models: `ollama list`.

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