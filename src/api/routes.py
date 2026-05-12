"""
src/api/routes.py
─────────────────
Flask route definitions for the Indic Finance Assistant API.

Design principles:
- Routes contain zero business logic — they validate input, call services,
  and format responses. That's it.
- All errors return structured JSON with a consistent schema.
- No hardcoded IP addresses, ports, or debug flags anywhere.
- The logic bug in the original /respond endpoint is fixed:
  `restart` is now set correctly based on actual choice value.
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from src.llm.gemini_client import GeminiClient
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import RetrievalResult
from src.speech.speech_service import SpeechService

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

# These are injected at app startup via init_routes()
_rag: RAGPipeline | None = None
_speech: SpeechService | None = None
_llm: GeminiClient | None = None

# Session state: maps session_id → last retrieved results.
# In production, replace this with Redis or a proper session store.
_session_store: dict[str, list[RetrievalResult]] = {}


def init_routes(rag: RAGPipeline, speech: SpeechService, llm: GeminiClient) -> None:
    """Inject service dependencies into the routes module."""
    global _rag, _speech, _llm
    _rag = rag
    _speech = speech
    _llm = llm


def _error_response(message: str, status_code: int = 400) -> tuple:
    """Return a consistent error JSON payload."""
    return jsonify({"success": False, "error": message}), status_code


def _success_response(data: dict) -> tuple:
    """Return a consistent success JSON payload."""
    return jsonify({"success": True, **data}), 200


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Liveness probe endpoint for deployment health checks."""
    return jsonify({"status": "ok", "service": "indic-finance-assistant"}), 200


@api_bp.route("/chat", methods=["POST"])
def chat():
    """
    Step 1: Accept voice input, transcribe it, retrieve top-k candidate answers.

    Request body (JSON):
        lang  (str): Language code ("en", "hi", "ta")
        audio (str): Base64-encoded audio

    Response:
        asr_out        (str): Transcribed text
        options        (str): Numbered list of candidate questions
        options_tts    (str): Base64 audio of options being read out
        session_id     (str): Carry this forward to /respond
    """
    data = request.get_json(silent=True) or {}
    lang = data.get("lang", "").strip()
    audio_b64 = data.get("audio", "").strip()
    session_id = data.get("session_id", "default")

    if not lang:
        return _error_response("Missing required field: 'lang'")
    if not audio_b64:
        return _error_response("Missing required field: 'audio'")

    try:
        # 1. Transcribe audio → text
        user_text = _speech.transcribe(lang, audio_b64)
        logger.info("ASR | lang=%s | text=%s", lang, user_text)

        # 2. Translate to English for retrieval
        user_text_en = _speech.to_english(lang, user_text)

        # 3. Retrieve top-k candidates
        results = _rag.query(user_text_en)
        _session_store[session_id] = results

        # 4. Build numbered options string
        options_lines = [f"{i + 1}. {r.question}" for i, r in enumerate(results)]
        options_lines.append(f"{len(results) + 1}. None of the above")
        options_str = "\n".join(options_lines)

        # 5. Translate options back to user's language and synthesise TTS
        options_in_lang = _speech.from_english(lang, options_str)
        options_tts = _speech.synthesise(lang, options_in_lang)

        return _success_response(
            {
                "asr_out": user_text,
                "options": options_in_lang,
                "options_tts": options_tts,
                "session_id": session_id,
                "num_options": len(results),
            }
        )

    except ValueError as exc:
        logger.warning("Validation error in /chat: %s", exc)
        return _error_response(str(exc))
    except Exception as exc:
        logger.exception("Unexpected error in /chat")
        return _error_response(f"Internal server error: {exc}", 500)


@api_bp.route("/respond", methods=["POST"])
def respond():
    """
    Step 2: Accept the user's choice (by voice), return the full answer.

    Request body (JSON):
        lang       (str): Language code ("en", "hi", "ta")
        audio      (str): Base64-encoded audio of the user saying their choice
        session_id (str): Session ID from /chat

    Response:
        choice_text  (str): What the ASR heard
        answer       (str): The final answer (in user's language)
        answer_tts   (str): Base64 audio of the answer
        done         (bool): True = conversation can restart for a new question
    """
    data = request.get_json(silent=True) or {}
    lang = data.get("lang", "").strip()
    audio_b64 = data.get("audio", "").strip()
    session_id = data.get("session_id", "default")

    if not lang:
        return _error_response("Missing required field: 'lang'")
    if not audio_b64:
        return _error_response("Missing required field: 'audio'")

    results = _session_store.get(session_id)
    if results is None:
        return _error_response("No active session found. Please call /chat first.", 400)

    _word_to_digit = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "ek": 1,
        "do": 2,
        "teen": 3,
        "char": 4,
        "paanch": 5,
    }

    try:
        # 1. Transcribe the user's spoken choice (always in English for digits)
        choice_text = _speech.transcribe("en", audio_b64).strip().lower().strip(".,!?")
        logger.info("Choice ASR output: %s", choice_text)

        # 2. Parse choice to integer
        if choice_text.isdigit():
            choice = int(choice_text)
        else:
            choice = _word_to_digit.get(choice_text)
            if choice is None:
                raise ValueError(
                    f"Could not parse '{choice_text}' as a valid option. "
                    f"Please say a number between 1 and {len(results) + 1}."
                )

        none_of_above = len(results) + 1

        # 3. Handle "None of the above"
        if choice == none_of_above:
            message = "Please rephrase your question and try again."
            message_in_lang = _speech.from_english(lang, message)
            tts_out = _speech.synthesise(lang, message_in_lang)
            return _success_response(
                {
                    "choice_text": choice_text,
                    "answer": message_in_lang,
                    "answer_tts": tts_out,
                    "done": True,
                }
            )

        # 4. Retrieve the selected result
        selected = _rag.get_answer_by_index(choice, results)

        # 5. Humanise the answer via LLM
        final_answer = _llm.humanise_answer(selected.question, selected.answer)

        # 6. Translate and synthesise
        final_answer_in_lang = _speech.from_english(lang, final_answer)
        tts_out = _speech.synthesise(lang, final_answer_in_lang)

        # 7. Clean up session
        del _session_store[session_id]

        return _success_response(
            {
                "choice_text": choice_text,
                "answer": final_answer_in_lang,
                "answer_tts": tts_out,
                "done": True,
            }
        )

    except (ValueError, IndexError) as exc:
        logger.warning("User input error in /respond: %s", exc)
        error_msg = _speech.from_english(lang, str(exc))
        tts_out = _speech.synthesise(lang, error_msg)
        return (
            jsonify(
                {
                    "success": False,
                    "error": error_msg,
                    "answer_tts": tts_out,
                    "done": False,
                }
            ),
            400,
        )

    except Exception as exc:
        logger.exception("Unexpected error in /respond")
        return _error_response(f"Internal server error: {exc}", 500)
