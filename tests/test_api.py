"""
tests/test_api.py
─────────────────
Integration tests for the Flask API routes.

Tests the full request/response cycle without any real external services.
All speech and LLM calls are mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.api.app import create_app
from src.config import AppConfig, RAGConfig, SpeechConfig
from src.rag.retriever import RetrievalResult


# ── Test Config ───────────────────────────────────────────────────────────────

@pytest.fixture
def test_config(tmp_path) -> AppConfig:
    rag_config = RAGConfig(
        top_k=3,
        embedding_model="paraphrase-MiniLM-L6-v2",
        similarity_metric="l2",
        data_path=str(tmp_path / "qa.csv"),
        faiss_index_path=str(tmp_path / "index.faiss"),
    )
    speech_config = SpeechConfig(
        asr_service_ids={},
        mt_service_id="",
        tts_service_ids={},
        voice_gender="female",
        ulca_api_key="fake",
        ulca_user_id="fake",
        ulca_authorization="fake",
    )
    return AppConfig(
        name="Test App",
        version="0.0.1",
        debug=True,
        host="127.0.0.1",
        port=5003,
        gemini_api_key="fake",
        rag=rag_config,
        speech=speech_config,
        supported_languages=[],
    )


@pytest.fixture
def mock_rag():
    rag = MagicMock()
    rag.query.return_value = [
        RetrievalResult("What is NPV?", "NPV is net present value.", 0.1, 0),
        RetrievalResult("What is IRR?", "IRR is internal rate of return.", 0.3, 1),
        RetrievalResult("What is EMI?", "EMI is equated monthly instalment.", 0.5, 2),
    ]
    rag.get_answer_by_index.return_value = RetrievalResult(
        "What is NPV?", "NPV is net present value.", 0.1, 0
    )
    return rag


@pytest.fixture
def mock_speech():
    speech = MagicMock()
    speech.transcribe.return_value = "What is NPV"
    speech.to_english.return_value = "What is NPV"
    speech.from_english.return_value = "NPV is net present value."
    speech.synthesise.return_value = "base64audiofake=="
    return speech


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.humanise_answer.return_value = (
        "NPV stands for Net Present Value, measuring investment worth."
    )
    return llm


@pytest.fixture
def client(test_config, mock_rag, mock_speech, mock_llm):
    with patch("src.api.app.RAGPipeline") as MockRAG, \
         patch("src.api.app.GeminiClient") as MockLLM, \
         patch("src.api.app.SpeechService") as MockSpeech:

        MockRAG.return_value = mock_rag
        MockLLM.return_value = mock_llm
        MockSpeech.return_value = mock_speech

        app = create_app(test_config)
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c


# ── Health Check ──────────────────────────────────────────────────────────────

class TestHealthCheck:

    def test_health_returns_ok(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"


# ── /chat Endpoint ────────────────────────────────────────────────────────────

class TestChatEndpoint:

    def test_missing_lang_returns_400(self, client):
        response = client.post("/api/v1/chat", json={"audio": "abc=="})
        assert response.status_code == 400
        data = response.get_json()
        assert data["success"] is False
        assert "lang" in data["error"]

    def test_missing_audio_returns_400(self, client):
        response = client.post("/api/v1/chat", json={"lang": "en"})
        assert response.status_code == 400
        assert "audio" in response.get_json()["error"]

    def test_valid_request_returns_options(self, client):
        response = client.post("/api/v1/chat", json={
            "lang": "en",
            "audio": "base64audiofake==",
            "session_id": "test-session-1",
        })
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "options" in data
        assert "options_tts" in data
        assert "session_id" in data
        assert data["num_options"] == 3

    def test_response_contains_asr_output(self, client, mock_speech):
        mock_speech.transcribe.return_value = "What is compound interest"
        response = client.post("/api/v1/chat", json={
            "lang": "hi",
            "audio": "base64audiofake==",
        })
        assert response.status_code == 200
        assert response.get_json()["asr_out"] == "What is compound interest"


# ── /respond Endpoint ─────────────────────────────────────────────────────────

class TestRespondEndpoint:

    def _seed_session(self, client, session_id: str = "test-session"):
        """Call /chat first to populate the session store."""
        client.post("/api/v1/chat", json={
            "lang": "en",
            "audio": "base64audiofake==",
            "session_id": session_id,
        })

    def test_missing_lang_returns_400(self, client):
        response = client.post("/api/v1/respond", json={"audio": "abc=="})
        assert response.status_code == 400

    def test_no_active_session_returns_400(self, client, mock_speech):
        mock_speech.transcribe.return_value = "1"
        response = client.post("/api/v1/respond", json={
            "lang": "en",
            "audio": "base64audiofake==",
            "session_id": "nonexistent-session",
        })
        assert response.status_code == 400
        assert "session" in response.get_json()["error"].lower()

    def test_valid_choice_returns_answer(self, client, mock_speech):
        self._seed_session(client, "respond-test")
        mock_speech.transcribe.return_value = "1"

        response = client.post("/api/v1/respond", json={
            "lang": "en",
            "audio": "base64audiofake==",
            "session_id": "respond-test",
        })
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert "answer" in data
        assert "answer_tts" in data
        assert data["done"] is True

    def test_none_of_above_choice(self, client, mock_speech):
        self._seed_session(client, "none-session")
        mock_speech.transcribe.return_value = "4"  # = len(results) + 1

        response = client.post("/api/v1/respond", json={
            "lang": "en",
            "audio": "base64audiofake==",
            "session_id": "none-session",
        })
        assert response.status_code == 200
        data = response.get_json()
        assert data["success"] is True
        assert data["done"] is True

    def test_word_choice_one_is_accepted(self, client, mock_speech):
        self._seed_session(client, "word-session")
        mock_speech.transcribe.return_value = "one"

        response = client.post("/api/v1/respond", json={
            "lang": "en",
            "audio": "base64audiofake==",
            "session_id": "word-session",
        })
        assert response.status_code == 200
        