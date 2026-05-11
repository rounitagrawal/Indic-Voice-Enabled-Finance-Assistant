"""
tests/test_llm.py
─────────────────
Unit tests for the Gemini LLM client.

All tests mock the external Gemini API — no real network calls or API keys needed.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.llm.gemini_client import GeminiClient


@pytest.fixture
def mock_genai():
    with patch("src.llm.gemini_client.genai") as mock:
        mock_model = MagicMock()
        mock.GenerativeModel.return_value = mock_model
        yield mock, mock_model


class TestGeminiClient:

    def test_returns_humanised_text(self, mock_genai):
        _, mock_model = mock_genai
        mock_model.generate_content.return_value = MagicMock(text="A concise answer.")

        client = GeminiClient(api_key="fake-key")
        result = client.humanise_answer("What is NPV?", "NPV is net present value.")

        assert result == "A concise answer."
        mock_model.generate_content.assert_called_once()

    def test_falls_back_to_raw_answer_on_empty_response(self, mock_genai):
        _, mock_model = mock_genai
        mock_model.generate_content.return_value = MagicMock(text=None)

        client = GeminiClient(api_key="fake-key")
        raw = "NPV is net present value."
        result = client.humanise_answer("What is NPV?", raw)

        assert result == raw  # fallback to raw answer

    def test_falls_back_to_raw_answer_on_api_exception(self, mock_genai):
        _, mock_model = mock_genai
        mock_model.generate_content.side_effect = Exception("API rate limit exceeded")

        client = GeminiClient(api_key="fake-key")
        raw = "NPV is net present value."
        result = client.humanise_answer("What is NPV?", raw)

        assert result == raw  # graceful fallback, no exception raised

    def test_prompt_contains_question_and_answer(self, mock_genai):
        _, mock_model = mock_genai
        mock_model.generate_content.return_value = MagicMock(text="Answer.")

        client = GeminiClient(api_key="fake-key")
        client.humanise_answer("What is SIP?", "SIP is systematic investment plan.")

        call_args = mock_model.generate_content.call_args[0][0]
        assert "What is SIP?" in call_args
        assert "SIP is systematic investment plan." in call_args
