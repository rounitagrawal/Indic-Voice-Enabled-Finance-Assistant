"""
src/llm/gemini_client.py
────────────────────────
Gemini API client for humanising RAG-retrieved answers.

Replaces the original Response.py. Key changes:
- API key loaded from config (environment variable), never hardcoded.
- Model initialised once (not on every call).
- Structured prompt with clear separation of roles.
- Graceful fallback: returns the raw answer if the LLM call fails,
  so the system degrades cleanly rather than throwing a 500.
"""
from __future__ import annotations

import logging

import google.generativeai as genai

logger = logging.getLogger(__name__)

_HUMANISE_PROMPT = """\
You are a friendly and concise financial assistant.

A user asked the following question:
{question}

The knowledge base returned this answer:
{answer}

Rewrite the answer in plain, conversational English. \
Keep it under 70 words. Do not add information not present in the answer. \
Do not use bullet points or lists.\
"""


class GeminiClient:
    """Wrapper around the Gemini generative model."""

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash-latest") -> None:
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name=model_name)
        logger.info("Gemini client initialised with model: %s", model_name)

    def humanise_answer(self, question: str, raw_answer: str) -> str:
        """
        Rewrite a raw knowledge-base answer into a conversational response.

        Args:
            question:    The user's original question (in English).
            raw_answer:  The answer retrieved from the vector store.

        Returns:
            A concise, humanised response string.
            Falls back to raw_answer if the API call fails.
        """
        prompt = _HUMANISE_PROMPT.format(question=question, answer=raw_answer)

        try:
            response = self._model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            logger.warning("Gemini returned an empty response. Falling back to raw answer.")
            return raw_answer
        except Exception as exc:
            logger.error("Gemini API call failed: %s. Falling back to raw answer.", exc)
            return raw_answer
