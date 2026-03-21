"""
GitLab Handbook AI — FastAPI Backend
RAG pipeline: scrape → chunk → embed → FAISS → Claude answer
"""

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from rag.pipeline import RAGPipeline



# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("main")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="GitLab Handbook AI",
    description="RAG-powered chatbot over GitLab Handbook & Direction pages",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── RAG pipeline (lazy-loaded once on first request) ─────────────────────────
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising RAG pipeline …")
        _pipeline = RAGPipeline()
        _pipeline.ensure_ready()
        logger.info("RAG pipeline ready.")
    return _pipeline


# ─── Schemas ──────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None


class Source(BaseModel):
    title: str
    url: str


class AskResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[Source]
    conversation_id: str


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    pipeline = get_pipeline()
    result = pipeline.query(req.question.strip())

    conv_id = req.conversation_id or hashlib.md5(
        (req.question + str(time.time())).encode()
    ).hexdigest()[:12]

    return AskResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        sources=[Source(**s) for s in result["sources"]],
        conversation_id=conv_id,
    )


@app.post("/reindex")
def reindex():
    """Force a fresh scrape + re-index (admin endpoint)."""
    pipeline = get_pipeline()
    pipeline.build_index(force=True)
    return {"status": "reindexed", "chunks": pipeline.num_chunks}


# ─── Serve frontend (for local dev & single-container deploys) ────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
