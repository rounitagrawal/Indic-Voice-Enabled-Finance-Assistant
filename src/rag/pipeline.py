"""
src/rag/pipeline.py
───────────────────
Orchestrates the full RAG pipeline: data loading, indexing, and querying.

This is the single entry point for all RAG operations. The Flask routes
should call this and nothing else — they don't need to know about FAISS
or embeddings directly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import RAGConfig
from src.rag.embedder import Embedder
from src.rag.retriever import FAISSRetriever, RetrievalResult

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end retrieval-augmented generation pipeline."""

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._embedder = Embedder(config.embedding_model)
        self._retriever = FAISSRetriever(config.faiss_index_path)

    def initialize(self) -> None:
        """
        Load data and prepare the FAISS index.

        Tries to load a cached index from disk first. Only rebuilds
        (and re-encodes all documents) if no cached index exists.
        """
        if self._retriever.load():
            logger.info("Using cached FAISS index — skipping re-encoding.")
            return

        logger.info("No cached index found. Building from %s …", self._config.data_path)
        questions, answers = self._load_data(self._config.data_path)

        logger.info("Encoding %d documents …", len(questions))
        vectors = self._embedder.encode(questions)

        self._retriever.build(questions, answers, vectors)
        self._retriever.save()
        logger.info("Index built and saved.")

    def query(self, user_input: str) -> list[RetrievalResult]:
        """
        Retrieve the top-k most relevant Q&A pairs for a user query.

        Args:
            user_input: The user's question in English (after translation if needed).

        Returns:
            Ordered list of RetrievalResult objects (most similar first).
        """
        query_vector = self._embedder.encode_single(user_input)
        return self._retriever.search(query_vector, top_k=self._config.top_k)

    def get_answer_by_index(
        self, result_index: int, results: list[RetrievalResult]
    ) -> RetrievalResult:
        """
        Fetch a specific result from a previous query by its 1-based position.

        Args:
            result_index:  1-based position (1 = first result, matching user choice).
            results:       The list returned by a previous call to query().

        Returns:
            The selected RetrievalResult.

        Raises:
            IndexError: If result_index is out of range.
        """
        if not 1 <= result_index <= len(results):
            raise IndexError(
                f"Choice {result_index} is out of range. "
                f"Valid options are 1–{len(results)}."
            )
        return results[result_index - 1]

    @staticmethod
    def _load_data(file_path: str) -> tuple[list[str], list[str]]:
        """
        Load Q&A pairs from a CSV file.

        Expected CSV format:
            Column 0: question
            Column 1: answer

        Args:
            file_path: Path to the CSV file.

        Returns:
            Tuple of (questions, answers) lists.

        Raises:
            FileNotFoundError: If the CSV does not exist.
            ValueError: If the CSV is missing required columns.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{file_path}'. "
                f"Please place your finance_qa.csv in the data/ directory. "
                f"See data/README.md for the expected format."
            )

        df = pd.read_csv(file_path)

        if df.shape[1] < 2:
            raise ValueError(
                f"CSV at '{file_path}' must have at least 2 columns "
                f"(question, answer). Found: {df.columns.tolist()}"
            )

        questions = df.iloc[:, 0].dropna().tolist()
        answers = df.iloc[:, 1].dropna().tolist()

        if len(questions) != len(answers):
            raise ValueError(
                f"Column length mismatch: {len(questions)} questions vs "
                f"{len(answers)} answers. Check for missing values."
            )

        logger.info("Loaded %d Q&A pairs from %s.", len(questions), file_path)
        return questions, answers
