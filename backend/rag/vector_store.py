"""
vector_store.py — FAISS-backed vector index for fast nearest-neighbour search.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Wraps a FAISS flat L2/cosine index with a parallel metadata list.

    Files on disk
    -------------
    <index_dir>/faiss.index   — FAISS binary index
    <index_dir>/meta.json     — JSON array of per-chunk metadata
    """

    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.index_dir / "faiss.index"
        self._meta_path  = self.index_dir / "meta.json"
        self._index = None
        self._meta: list[dict[str, Any]] = []

    # ── Build ─────────────────────────────────────────────────────────────

    def build(self, vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        """Build a fresh index from vectors + matching metadata list."""
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss-cpu not installed. Run: pip install faiss-cpu") from exc

        assert len(vectors) == len(metadata), "vectors and metadata must have same length"
        dim = vectors.shape[1]

        logger.info("Building FAISS index: %d vectors × %d dims", len(vectors), dim)
        index = faiss.IndexFlatIP(dim)   # Inner-product = cosine when vectors are normalised
        index.add(vectors)

        faiss.write_index(index, str(self._index_path))
        self._meta_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")

        self._index = index
        self._meta = metadata
        logger.info("FAISS index saved to %s", self.index_dir)

    # ── Load ──────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load a previously built index from disk. Returns True on success."""
        if not self._index_path.exists() or not self._meta_path.exists():
            return False
        try:
            import faiss
            self._index = faiss.read_index(str(self._index_path))
            self._meta  = json.loads(self._meta_path.read_text(encoding="utf-8"))
            logger.info("Loaded FAISS index: %d vectors", self._index.ntotal)
            return True
        except Exception as exc:
            logger.warning("Could not load FAISS index: %s", exc)
            return False

    # ── Query ─────────────────────────────────────────────────────────────

    def search(self, query_vec: np.ndarray, k: int = 6) -> list[dict[str, Any]]:
        """
        Return the top-k metadata dicts for a single query vector.
        Each dict has an added 'score' key (cosine similarity 0-1).
        """
        if self._index is None:
            raise RuntimeError("Index not loaded. Call load() or build() first.")

        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = dict(self._meta[idx])
            meta["score"] = float(score)
            results.append(meta)
        return results

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    def exists(self) -> bool:
        return self._index_path.exists() and self._meta_path.exists()
