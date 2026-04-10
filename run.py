"""
Application launcher script.
Checks prerequisites, sets up directories, and starts the server.
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

# ─── Constants ───
BASE_DIR = Path(__file__).resolve().parent
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
OLLAMA_MODEL = os.getenv("LLM_MODEL", "gemma4")


def check_python():
    """Ensure Python 3.10+."""
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print(f"❌ Python 3.10+ required. You have {v.major}.{v.minor}.{v.micro}")
        sys.exit(1)
    print(f"✅ Python {v.major}.{v.minor}.{v.micro}")


def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            print(f"✅ Ollama running — models: {', '.join(model_names) or 'none'}")

            if OLLAMA_MODEL.split(":")[0] not in model_names:
                print(f"\n📥 Model '{OLLAMA_MODEL}' not found. Pulling...")
                print("   This may take several minutes for the first download.\n")
                subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
                print(f"✅ Model '{OLLAMA_MODEL}' ready")
            return True
        return False
    except Exception:
        print("\n" + "=" * 60)
        print("  ⚠️  Ollama is not running!")
        print("=" * 60)
        print("\n  To install Ollama:")
        print("    1. Download from https://ollama.com/download")
        print("    2. Install and run Ollama")
        print(f"    3. Pull the model: ollama pull {OLLAMA_MODEL}")
        print("    4. Re-run this script\n")
        print("  The app will start without LLM capabilities.\n")
        return False


def create_directories():
    """Create required directories."""
    dirs = [
        BASE_DIR / "data" / "faiss_index",
        BASE_DIR / "data" / "uploads",
    ]

    # References directory
    refs_dir = Path(os.getenv(
        "REFERENCES_DIR",
        os.path.expanduser("~/Desktop/references")
    ))
    dirs.append(refs_dir)

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print(f"✅ Directories ready")
    print(f"   📁 References: {refs_dir}")


def open_browser():
    """Open the app in the default browser after a delay."""
    time.sleep(2)
    url = f"http://localhost:{PORT}"
    webbrowser.open(url)
    print(f"\n🌐 Opened {url} in browser")


def main():
    print("\n" + "=" * 60)
    print("  🚀 Local RAG System — Launcher")
    print("=" * 60 + "\n")

    check_python()
    create_directories()
    check_ollama()

    print(f"\n🔧 Starting server on http://localhost:{PORT}...")
    print("   Press Ctrl+C to stop.\n")

    # Open browser in background
    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Start uvicorn
    try:
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host=HOST,
            port=PORT,
            reload=False,
            log_level="info",
            access_log=False,
        )
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except ImportError:
        print("\n❌ Dependencies not installed. Run:")
        print(f"   pip install -r {BASE_DIR / 'requirements.txt'}")
        sys.exit(1)


if __name__ == "__main__":
    main()
