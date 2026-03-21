"""
pipeline.py — Orchestrates scraping, indexing, retrieval, and generation.
"""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from .scraper import scrape_all, TextChunk
from .embedder import Embedder
from .vector_store import VectorStore
from .llm import LLMClient

logger = logging.getLogger(__name__)

INDEX_DIR = Path(os.environ.get("INDEX_DIR", "data/index"))
TOP_K = 6


class RAGPipeline:
    def __init__(self):
        self._embedder: Embedder | None = None
        self._store = VectorStore(INDEX_DIR)
        self._llm: LLMClient | None = None
        self._ready = False

    # ── Lazy getters ──────────────────────────────────────────────────────

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    @property
    def num_chunks(self) -> int:
        return self._store.size

    # ── Indexing ─────────────────────────────────────────────────────────

    def build_index(self, force: bool = False) -> None:
        if not force and self._store.exists():
            logger.info("Index already exists at %s — skipping build.", INDEX_DIR)
            return

        logger.info("Starting full scrape + index build …")
        chunks: list[TextChunk] = scrape_all(follow_links=False)

        if not chunks:
            raise RuntimeError("Scraper returned 0 chunks. Check network access.")

        texts = [c.text for c in chunks]
        logger.info("Embedding %d chunks …", len(texts))
        vectors = self.embedder.embed(texts)

        metadata = [
            {
                "text": c.text,
                "url": c.url,
                "title": c.title,
                "section": c.section,
            }
            for c in chunks
        ]
        self._store.build(vectors, metadata)
        logger.info("Index built: %d vectors.", self._store.size)

    def ensure_ready(self) -> None:
        if self._ready:
            return
        if not self._store.load():
            self.build_index()
        self._ready = True

    # ── Querying ──────────────────────────────────────────────────────────

    def query(self, question: str) -> dict[str, Any]:
        """
        Full RAG query: embed → retrieve → generate → return structured result.

        Returns:
            {
                "answer":     str,
                "confidence": float  (0-1),
                "sources":    [{"title": str, "url": str}, ...]
            }
        """
        self.ensure_ready()

        # 1. Embed query
        q_vec = self.embedder.embed_one(question)

        # 2. Retrieve top-k chunks
        hits = self._store.search(q_vec, k=TOP_K)
        if not hits:
            return {
                "answer": "I could not find any relevant information in the GitLab Handbook for your question.",
                "confidence": 0.0,
                "sources": [],
            }

        # 3. Compute confidence from top-hit cosine similarity
        top_score = hits[0]["score"]          # already 0-1 (normalised IP)
        confidence = float(np.clip(top_score, 0.0, 1.0))

        # 4. Deduplicate sources
        passages = [h["text"] for h in hits]
        seen_urls: set[str] = set()
        sources = []
        for h in hits:
            if h["url"] not in seen_urls:
                seen_urls.add(h["url"])
                sources.append({"title": h["title"], "url": h["url"]})

        # 5. Generate answer
        try:
            answer = self.llm.answer(question, passages)
        except Exception as exc:
            logger.error("LLM error: %s", exc)
            answer = (
                "I found relevant passages but could not generate an answer right now. "
                "Please check the sources below."
            )

        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
        }
