"""
src/api/app.py
──────────────
Flask application factory.

Using a factory function (rather than a module-level `app` object) makes
the app testable — tests can create isolated instances without side effects.
"""
from __future__ import annotations

import logging

from flask import Flask

from src.api.routes import api_bp, init_routes
from src.config import AppConfig, load_config
from src.llm.gemini_client import GeminiClient
from src.rag.pipeline import RAGPipeline
from src.speech.speech_service import SpeechService


def create_app(config: AppConfig | None = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config: Optional pre-loaded AppConfig. If None, loads from environment.

    Returns:
        Configured Flask application instance.
    """
    if config is None:
        config = load_config()

    _configure_logging(config.debug)
    logger = logging.getLogger(__name__)

    app = Flask(__name__)
    app.config["DEBUG"] = config.debug

    # ── Initialise services ──────────────────────────────────────────────────
    logger.info("Initialising RAG pipeline …")
    rag = RAGPipeline(config.rag)
    rag.initialize()

    logger.info("Initialising LLM client …")
    llm = GeminiClient(api_key=config.gemini_api_key)

    logger.info("Initialising Speech service …")
    speech = SpeechService(config.speech)

    # ── Wire routes ──────────────────────────────────────────────────────────
    init_routes(rag=rag, speech=speech, llm=llm)
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    logger.info(
        "Application '%s' v%s ready. Debug=%s",
        config.name, config.version, config.debug,
    )
    return app


def _configure_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
