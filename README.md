# GitLab Handbook AI 🦊

A production-grade RAG chatbot that answers questions about the [GitLab Handbook](https://handbook.gitlab.com/) and [GitLab Direction](https://about.gitlab.com/direction/) pages — powered by FAISS semantic search and Claude Sonnet.

---

## ✨ Features

- **Semantic RAG search** — FAISS vector index over ~80 scraped GitLab pages
- **Local embeddings** — `all-MiniLM-L6-v2` via sentence-transformers (no extra API key)
- **Claude Sonnet answers** — grounded responses, never hallucinates beyond the context
- **Confidence scoring** — every answer shows High / Medium / Low + percentage
- **Cited sources** — clickable links to the exact Handbook pages used
- **Chat history sidebar** — session-scoped conversation tracking
- **One-command deployment** — Render.com via `render.yaml`

---

## 🏗️ Architecture

```
User question
      │
      ▼
┌─────────────┐     embed (MiniLM)    ┌──────────────────┐
│  FastAPI     │ ────────────────────► │  FAISS Index      │
│  /ask        │ ◄──── top-6 chunks ── │  (384-dim cosine) │
└─────────────┘                        └──────────────────┘
      │
      │  chunks + question
      ▼
┌─────────────┐
│  Claude      │  ← system prompt grounds it on GitLab context
│  Sonnet      │
└─────────────┘
      │
      ▼
{ answer, confidence, sources }
```

**Stack:**
| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | FAISS (faiss-cpu) |
| LLM | Anthropic Claude Sonnet |
| Scraping | requests + BeautifulSoup4 |
| Frontend | Vanilla HTML/CSS/JS (zero dependencies) |

---

## 📁 Project Structure

```
gitlab-handbook-ai/
├── backend/
│   ├── main.py              ← FastAPI app
│   ├── requirements.txt
│   └── rag/
│       ├── scraper.py       ← Web crawler (handbook + direction)
│       ├── embedder.py      ← MiniLM sentence embeddings
│       ├── vector_store.py  ← FAISS index wrapper
│       ├── llm.py           ← Claude Sonnet client
│       └── pipeline.py      ← RAG orchestration
├── frontend/
│   └── index.html           ← Chat UI (no framework, no build step)
├── scripts/
│   └── build_index.py       ← One-time index builder
├── data/
│   └── index/               ← FAISS index files (git-ignored)
├── Dockerfile
├── render.yaml              ← Render.com deploy config
├── .env.example
└── .gitignore
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/gitlab-handbook-ai.git
cd gitlab-handbook-ai

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r backend/requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Open `.env` and set your key:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Build the Index

This scrapes ~80 GitLab pages and builds the FAISS vector index. Run once (takes ~5–10 minutes depending on your connection).

```bash
python scripts/build_index.py
```

You should see output like:
```
2024-01-15 [INFO] Scraping https://handbook.gitlab.com/handbook/values/ ...
...
✅ Index built — 3847 chunks stored.
```

To force a rebuild (fresh data):
```bash
python scripts/build_index.py --force
```

### 4. Start the Server

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) 🎉

---

## 🌐 Deployment

### Option A — Render.com (Recommended, Free Tier)

Render is the easiest option: persistent disk for the index, automatic deploys on git push.

1. Push your repo to GitHub
2. Go to [render.com](https://render.com) → **New** → **Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Apply**
5. In the Render dashboard → **Environment** → add:
   ```
   ANTHROPIC_API_KEY = sk-ant-...
   ```
6. Click **Deploy** — your URL will be `https://gitlab-handbook-ai.onrender.com`

> **Note:** First deploy runs the scraper and builds the index (~10 min). Subsequent deploys are fast. The persistent disk keeps the index across deploys.

### Option B — Docker (Self-hosted / VPS)

```bash
# Build image
docker build -t gitlab-handbook-ai .

# Run (index builds on first startup if not cached)
docker run -d \
  -p 8000:8000 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -v $(pwd)/data:/app/data \
  --name handbook-ai \
  gitlab-handbook-ai
```

### Option C — Railway / Fly.io

Both support Docker deployments out of the box. Point at the `Dockerfile`, set `ANTHROPIC_API_KEY` in their env panel, and deploy.

### Option D — Vercel (Frontend-only static deploy)

If you want to host the backend separately (e.g., on Railway) and just deploy the frontend to Vercel:

1. Set `API_BASE` in `frontend/index.html` to your backend URL:
   ```js
   const API_BASE = 'https://your-backend.railway.app';
   ```
2. Deploy the `frontend/` folder to Vercel as a static site.

---

## 🔌 API Reference

### `POST /ask`

Ask a question about the GitLab Handbook.

**Request:**
```json
{
  "question": "What are GitLab's core values?",
  "conversation_id": "optional-string-for-continuity"
}
```

**Response:**
```json
{
  "answer": "GitLab's six core values form the acronym **CREDIT**: ...",
  "confidence": 0.84,
  "sources": [
    { "title": "GitLab Values", "url": "https://handbook.gitlab.com/handbook/values/" }
  ],
  "conversation_id": "abc123def456"
}
```

### `GET /health`

```json
{ "status": "ok", "timestamp": 1705312800.0 }
```

### `POST /reindex`

Force a fresh scrape and rebuild of the FAISS index (admin use).

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `INDEX_DIR` | `data/index` | Where FAISS index files are stored |
| `PORT` | `8000` | Server port |

---

## 🛠️ Development Notes

### Adjusting the scraper

Edit `backend/rag/scraper.py`:
- `SEED_URLS` — add/remove pages to crawl
- `MAX_PAGES` — cap on total pages followed
- `CHUNK_SIZE` — words per chunk (default: 400)
- `CHUNK_OVERLAP` — overlap between chunks (default: 60)

### Changing the LLM

Edit `backend/rag/llm.py` — swap `MODEL` and update the client. The system prompt is also there.

### Changing the embedding model

Edit `backend/rag/embedder.py` — swap `MODEL_NAME`. Update `dim` property to match. Rebuild the index after any change.

---

## 📄 License

MIT — see [LICENSE](LICENSE)
