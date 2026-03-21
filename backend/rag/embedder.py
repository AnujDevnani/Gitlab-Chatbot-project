"""
embedder.py — Turn text into dense vectors using a sentence-transformer model.

Model: sentence-transformers/all-MiniLM-L6-v2
  • 384-dim, fast, freely available, MIT-licensed
  • No API key required — runs entirely locally
"""

import logging
import numpy as np
from typing import Sequence

logger = logging.getLogger(__name__)


class Embedder:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        logger.info("Loading embedding model: %s", self.MODEL_NAME)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
        logger.info("Embedding model loaded.")

    def embed(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """
        Returns a float32 numpy array of shape (len(texts), 384).
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,   # cosine sim via dot product
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        return 384
