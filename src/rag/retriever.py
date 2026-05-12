"""
src/rag/retriever.py
────────────────────
FAISS-based vector retriever with index persistence.

Key improvements over the original:
- Index is saved to disk after first build, loaded on subsequent starts.
- Separates indexing logic from search logic cleanly.
- Validates data integrity on load.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved document with its similarity score."""

    question: str
    answer: str
    score: float
    index: int


class FAISSRetriever:
    """
    Builds and queries a FAISS flat L2 index.

    Persists the index and the associated text corpus to disk so the
    expensive encoding step only runs once.
    """

    def __init__(self, index_path: str) -> None:
        self._index_path = Path(index_path)
        self._meta_path = self._index_path.with_suffix(".pkl")
        self._index: faiss.IndexFlatL2 | None = None
        self._questions: list[str] = []
        self._answers: list[str] = []

    @property
    def is_built(self) -> bool:
        return self._index is not None and len(self._questions) > 0

    def build(
        self,
        questions: list[str],
        answers: list[str],
        vectors: np.ndarray,
    ) -> None:
        """
        Build a FAISS index from precomputed embedding vectors.

        Args:
            questions:  List of question strings (used for display/retrieval).
            answers:    Corresponding answers aligned with questions.
            vectors:    2D numpy array of shape (n_questions, embedding_dim).
        """
        if len(questions) != len(answers) != len(vectors):
            raise ValueError(
                f"Mismatched lengths: questions={len(questions)}, "
                f"answers={len(answers)}, vectors={len(vectors)}"
            )

        dimension = vectors.shape[1]
        self._index = faiss.IndexFlatL2(dimension)
        self._index.add(vectors.astype(np.float32))
        self._questions = questions
        self._answers = answers

        logger.info(
            "FAISS index built with %d vectors (dim=%d).", len(questions), dimension
        )

    def save(self) -> None:
        """Persist the index and metadata to disk."""
        if not self.is_built:
            raise RuntimeError("Cannot save: index has not been built yet.")

        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))

        with open(self._meta_path, "wb") as f:
            pickle.dump({"questions": self._questions, "answers": self._answers}, f)

        logger.info("FAISS index saved to %s.", self._index_path)

    def load(self) -> bool:
        """
        Attempt to load a pre-built index from disk.

        Returns:
            True if loaded successfully, False if no saved index exists.
        """
        if not self._index_path.exists() or not self._meta_path.exists():
            return False

        self._index = faiss.read_index(str(self._index_path))

        with open(self._meta_path, "rb") as f:
            meta = pickle.load(f)

        self._questions = meta["questions"]
        self._answers = meta["answers"]

        logger.info(
            "FAISS index loaded from disk (%d documents).", len(self._questions)
        )
        return True

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[RetrievalResult]:
        """
        Find the top-k most similar documents to a query vector.

        Args:
            query_vector:  1D numpy array of the query embedding.
            top_k:         Number of results to return.

        Returns:
            List of RetrievalResult objects, ordered by similarity (closest first).
        """
        if not self.is_built:
            raise RuntimeError(
                "Index has not been built or loaded. Call build() or load() first."
            )

        query = np.array([query_vector], dtype=np.float32)
        distances, indices = self._index.search(query, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._questions):
                continue  # FAISS can return -1 for unfilled slots
            results.append(
                RetrievalResult(
                    question=self._questions[idx],
                    answer=self._answers[idx],
                    score=float(dist),
                    index=int(idx),
                )
            )
        return results
