"""
tests/test_rag.py
─────────────────
Unit tests for the RAG pipeline.

These tests use only in-memory data — no external APIs, no files required.
They run in CI without any credentials.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.rag.embedder import Embedder
from src.rag.retriever import FAISSRetriever, RetrievalResult
from src.rag.pipeline import RAGPipeline
from src.config import RAGConfig

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_questions() -> list[str]:
    return [
        "What is the difference between NPV and IRR?",
        "How does compound interest work?",
        "What is a mutual fund?",
        "How do I calculate EMI?",
        "What is a credit score?",
    ]


@pytest.fixture
def sample_answers() -> list[str]:
    return [
        "NPV measures net dollar value; IRR is the percentage return.",
        "Compound interest earns returns on both principal and accumulated interest.",
        "A mutual fund pools money from many investors to buy a diversified portfolio.",
        "EMI is calculated using principal, interest rate, and loan tenure.",
        "A credit score is a number rating your creditworthiness, typically 300–900.",
    ]


@pytest.fixture
def sample_vectors(sample_questions) -> np.ndarray:
    """Deterministic fake vectors — no actual model needed."""
    rng = np.random.default_rng(seed=42)
    return rng.random((len(sample_questions), 384)).astype(np.float32)


@pytest.fixture
def tmp_index_path(tmp_path) -> str:
    return str(tmp_path / "test.faiss")


@pytest.fixture
def rag_config(tmp_path, tmp_index_path) -> RAGConfig:
    csv_path = str(tmp_path / "qa.csv")
    return RAGConfig(
        top_k=3,
        embedding_model="paraphrase-MiniLM-L6-v2",
        similarity_metric="l2",
        data_path=csv_path,
        faiss_index_path=tmp_index_path,
    )


# ── Retriever Tests ───────────────────────────────────────────────────────────


class TestFAISSRetriever:

    def test_build_and_search(
        self, sample_questions, sample_answers, sample_vectors, tmp_index_path
    ):
        retriever = FAISSRetriever(tmp_index_path)
        retriever.build(sample_questions, sample_answers, sample_vectors)

        query = sample_vectors[0]  # Exact match should be closest
        results = retriever.search(query, top_k=3)

        assert len(results) == 3
        assert isinstance(results[0], RetrievalResult)
        assert results[0].index == 0  # Exact match is first
        assert results[0].score < results[1].score  # Lower L2 = more similar

    def test_save_and_load(
        self, sample_questions, sample_answers, sample_vectors, tmp_index_path
    ):
        retriever = FAISSRetriever(tmp_index_path)
        retriever.build(sample_questions, sample_answers, sample_vectors)
        retriever.save()

        loaded = FAISSRetriever(tmp_index_path)
        assert loaded.load() is True
        assert len(loaded._questions) == len(sample_questions)

        # Results should be identical
        query = sample_vectors[2]
        original_results = retriever.search(query, top_k=2)
        loaded_results = loaded.search(query, top_k=2)

        assert [r.index for r in original_results] == [r.index for r in loaded_results]

    def test_load_returns_false_when_no_index(self, tmp_index_path):
        retriever = FAISSRetriever(tmp_index_path)
        assert retriever.load() is False

    def test_search_before_build_raises(self, tmp_index_path):
        retriever = FAISSRetriever(tmp_index_path)
        with pytest.raises(RuntimeError, match="not been built"):
            retriever.search(np.zeros(384, dtype=np.float32))

    def test_mismatched_lengths_raise(self, tmp_index_path, sample_vectors):
        retriever = FAISSRetriever(tmp_index_path)
        with pytest.raises(ValueError, match="Mismatched lengths"):
            retriever.build(["q1"], ["a1", "a2"], sample_vectors)


# ── Pipeline Tests ────────────────────────────────────────────────────────────


class TestRAGPipeline:

    def _make_csv(self, path: str, questions: list[str], answers: list[str]) -> None:
        pd.DataFrame({"question": questions, "answer": answers}).to_csv(
            path, index=False
        )

    def test_load_data_success(self, rag_config, sample_questions, sample_answers):
        self._make_csv(rag_config.data_path, sample_questions, sample_answers)
        qs, ans = RAGPipeline._load_data(rag_config.data_path)
        assert len(qs) == len(sample_questions)
        assert qs[0] == sample_questions[0]

    def test_load_data_missing_file_raises(self, rag_config):
        with pytest.raises(FileNotFoundError):
            RAGPipeline._load_data("/nonexistent/path/qa.csv")

    def test_load_data_insufficient_columns_raises(self, rag_config):
        pd.DataFrame({"question": ["q1"]}).to_csv(rag_config.data_path, index=False)
        with pytest.raises(ValueError, match="at least 2 columns"):
            RAGPipeline._load_data(rag_config.data_path)

    def test_get_answer_by_index_valid(
        self, rag_config, sample_questions, sample_answers, sample_vectors
    ):
        results = [
            RetrievalResult(q, a, 0.1, i)
            for i, (q, a) in enumerate(zip(sample_questions[:3], sample_answers[:3]))
        ]
        pipeline = RAGPipeline.__new__(RAGPipeline)  # Skip __init__
        result = pipeline.get_answer_by_index(1, results)
        assert result.question == sample_questions[0]

    def test_get_answer_by_index_out_of_range(self, rag_config):
        results = [RetrievalResult("q", "a", 0.1, 0)]
        pipeline = RAGPipeline.__new__(RAGPipeline)
        with pytest.raises(IndexError):
            pipeline.get_answer_by_index(5, results)

    @patch("src.rag.pipeline.Embedder")
    def test_initialize_builds_index_when_no_cache(
        self,
        mock_embedder_cls,
        rag_config,
        sample_questions,
        sample_answers,
        sample_vectors,
    ):
        self._make_csv(rag_config.data_path, sample_questions, sample_answers)

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = sample_vectors
        mock_embedder_cls.return_value = mock_embedder

        pipeline = RAGPipeline(rag_config)
        pipeline.initialize()

        assert pipeline._retriever.is_built


# ── Embedder Tests ────────────────────────────────────────────────────────────


class TestEmbedder:

    @patch("src.rag.embedder.SentenceTransformer")
    def test_encode_returns_array(self, mock_st):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((3, 384), dtype=np.float32)
        mock_st.return_value = mock_model

        embedder = Embedder("some-model")
        result = embedder.encode(["q1", "q2", "q3"])
        assert result.shape == (3, 384)

    @patch("src.rag.embedder.SentenceTransformer")
    def test_encode_empty_raises(self, mock_st):
        mock_st.return_value = MagicMock()
        embedder = Embedder("some-model")
        with pytest.raises(ValueError, match="empty list"):
            embedder.encode([])

    @patch("src.rag.embedder.SentenceTransformer")
    def test_encode_single_returns_1d(self, mock_st):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)
        mock_st.return_value = mock_model

        embedder = Embedder("some-model")
        result = embedder.encode_single("test query")
        assert result.ndim == 1
