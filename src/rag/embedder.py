"""
src/rag/embedder.py
───────────────────
Thin wrapper around SentenceTransformer.

Keeping the embedding model isolated here means you can swap it out
(e.g., upgrade to a multilingual model) without touching the pipeline.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Embedder:
    """Encodes text into dense vector representations."""

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of strings into a 2D numpy array of embeddings.

        Args:
            texts: List of strings to encode.

        Returns:
            numpy array of shape (len(texts), embedding_dim).
        """
        if not texts:
            raise ValueError("Cannot encode an empty list of texts.")
        return self._model.encode(texts, show_progress_bar=False)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single string into a 1D embedding vector."""
        return self.encode([text])[0]
