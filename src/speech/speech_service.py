"""
src/speech/speech_service.py
─────────────────────────────
Abstraction layer over the AI4Bharat / ULCA speech APIs (ASR, MT, TTS).

This module wraps the external ASR_TTS_MT module that interfaces with
Shabdh Technologies / AI4Bharat APIs. All configuration is injected
from the central config — no hardcoded values anywhere in this file.

Why this wrapper exists:
- Decouples the routes from the external library.
- Makes the speech layer mockable in tests.
- Centralises error handling for all three modalities.
"""

from __future__ import annotations

import logging

from src.config import SpeechConfig

logger = logging.getLogger(__name__)

# Lazy import: ASR_TTS_MT requires external services to be reachable.
# Importing at module level would crash tests and CI where these aren't available.
try:
    from ASR_TTS_MT import ASR_call, MT_call, TTS_call  # type: ignore[import]

    _SPEECH_AVAILABLE = True
except ImportError:
    logger.warning(
        "ASR_TTS_MT module not found. Speech features will be unavailable. "
        "This is expected in CI / test environments."
    )
    _SPEECH_AVAILABLE = False


class SpeechService:
    """
    Unified interface for Automatic Speech Recognition, Machine Translation,
    and Text-to-Speech using AI4Bharat / ULCA APIs.
    """

    SUPPORTED_LANGUAGES = {"en", "hi", "ta"}

    def __init__(self, config: SpeechConfig) -> None:
        self._config = config

    def transcribe(self, language_code: str, audio_b64: str) -> str:
        """
        Convert base64-encoded audio to text using ASR.

        Args:
            language_code: BCP-47 language code (e.g., "hi", "ta", "en").
            audio_b64:     Base64-encoded audio bytes.

        Returns:
            Transcribed text string.

        Raises:
            RuntimeError: If the speech module is unavailable.
            ValueError:   If language_code is not supported.
        """
        self._require_speech_module()
        self._validate_language(language_code)

        logger.debug("ASR call | lang=%s", language_code)
        result = ASR_call(language_code, audio_b64)
        logger.debug("ASR result: %s", result)
        return result

    def translate(self, source_lang: str, target_lang: str, text: str) -> str:
        """
        Translate text between languages using Machine Translation.

        Args:
            source_lang: Source language code.
            target_lang: Target language code.
            text:        Text to translate.

        Returns:
            Translated text string.
        """
        self._require_speech_module()

        if source_lang == target_lang:
            return text  # No-op: avoid a pointless API call

        logger.debug("MT call | %s → %s", source_lang, target_lang)
        return MT_call(source_lang, target_lang, text)

    def synthesise(self, language_code: str, text: str) -> str:
        """
        Convert text to speech, returning base64-encoded audio.

        Args:
            language_code: Target language code.
            text:          Text to speak.

        Returns:
            Base64-encoded audio string.
        """
        self._require_speech_module()
        self._validate_language(language_code)

        logger.debug("TTS call | lang=%s | text_len=%d", language_code, len(text))
        return TTS_call(language_code, text)

    def to_english(self, language_code: str, text: str) -> str:
        """Translate text to English. No-op if already English."""
        if language_code == "en":
            return text
        return self.translate(language_code, "en", text)

    def from_english(self, language_code: str, text: str) -> str:
        """Translate from English to the target language. No-op if target is English."""
        if language_code == "en":
            return text
        return self.translate("en", language_code, text)

    def _require_speech_module(self) -> None:
        if not _SPEECH_AVAILABLE:
            raise RuntimeError(
                "The ASR_TTS_MT module is not installed. "
                "Speech features are unavailable in this environment."
            )

    def _validate_language(self, language_code: str) -> None:
        if language_code not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: '{language_code}'. "
                f"Supported: {sorted(self.SUPPORTED_LANGUAGES)}"
            )
