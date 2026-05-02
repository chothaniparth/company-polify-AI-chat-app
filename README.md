# 📋 ACME Policy AI — Company Policy RAG Chat App

A production-ready **Retrieval-Augmented Generation (RAG)** chat application built with:

- 🤖 **Google Gemini 2.0 Flash** — LLM for answers
- 🔢 **Google `text-embedding-004`** — State-of-the-art embeddings
- 🗃️ **ChromaDB** — Persistent local vector store
- 🦜 **LangChain** — RAG pipeline + history-aware retrieval
- 🎈 **Streamlit** — Interactive chat UI
- ⚡ **uv** — Fast Python package manager

---

## 🚀 Quickstart

### 1. Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- A [Google AI Studio](https://aistudio.google.com) API key

### 2. Clone & set up

```bash
# Install dependencies with uv
uv sync

# Copy env file and add your key
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_key_here
```

### 3. Run

```bash
uv run streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
[History-Aware Condenser]  ← Chat History
     │  (Gemini rephrases follow-ups into standalone questions)
     ▼
[Chroma MMR Retriever]  →  Top-5 relevant policy chunks
     │
     ▼
[QA Chain]  (Gemini 2.0 Flash + system prompt + context)
     │
     ▼
Answer + Source sections
```

### Key components

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI, chat rendering, sidebar controls |
| `rag_engine.py` | RAG pipeline: loader, vector store, chain |
| `docs/` | Put your `.pdf` or `.txt` policy files here |
| `.chroma_db/` | Auto-created persistent vector database |

---

## 📄 Adding Your Own Policies

1. Drop `.pdf` or `.txt` files into the `docs/` folder  
   **OR** use the file uploader in the app sidebar
2. Click **Rebuild DB** in the sidebar to re-index

---

## ⚙️ Configuration

Edit constants in `rag_engine.py`:

```python
CHUNK_SIZE = 800      # Characters per chunk
CHUNK_OVERLAP = 120   # Overlap between chunks
TOP_K = 5             # Retrieved chunks per query
CHAT_MODEL = "gemini-2.0-flash"
EMBED_MODEL = "models/text-embedding-004"
```

---

## 🔒 Security Notes

- The API key is stored in `.env` (never commit this file)
- ChromaDB is stored locally in `.chroma_db/` — add to `.gitignore`
- For production, use environment secrets management (e.g., GCP Secret Manager)
