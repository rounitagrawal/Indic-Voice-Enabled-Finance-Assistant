"""
src/config.py
─────────────
Central configuration loader.

Loads from:
  1. configs/config.yaml  (non-secret settings)
  2. Environment variables / .env file  (secrets)

Fails fast with a clear error if required secrets are missing —
better to crash at startup than silently fail mid-request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env file if present (development only; prod uses real env vars)
load_dotenv()

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"


@dataclass(frozen=True)
class SpeechConfig:
    asr_service_ids: dict[str, str]
    mt_service_id: str
    tts_service_ids: dict[str, str]
    voice_gender: str
    ulca_api_key: str
    ulca_user_id: str
    ulca_authorization: str


@dataclass(frozen=True)
class RAGConfig:
    top_k: int
    embedding_model: str
    similarity_metric: str
    data_path: str
    faiss_index_path: str


@dataclass(frozen=True)
class AppConfig:
    name: str
    version: str
    debug: bool
    host: str
    port: int
    gemini_api_key: str
    rag: RAGConfig
    speech: SpeechConfig
    supported_languages: list[dict]


def _require_env(key: str) -> str:
    """Fetch a required environment variable; raise clearly if missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"See .env.example for setup instructions."
        )
    return value


def load_config() -> AppConfig:
    """Load and validate the full application configuration."""
    with open(_CONFIG_PATH, "r") as f:
        raw = yaml.safe_load(f)

    speech_raw = raw["speech"]
    rag_raw = raw["rag"]
    app_raw = raw["app"]

    rag = RAGConfig(
        top_k=int(os.getenv("TOP_K_RETRIEVAL", rag_raw["top_k"])),
        embedding_model=os.getenv("EMBEDDING_MODEL", rag_raw["embedding_model"]),
        similarity_metric=rag_raw["similarity_metric"],
        data_path=os.getenv("DATA_PATH", "data/finance_qa.csv"),
        faiss_index_path=os.getenv("FAISS_INDEX_PATH", "data/faiss_index.bin"),
    )

    speech = SpeechConfig(
        asr_service_ids=speech_raw["asr"],
        mt_service_id=speech_raw["mt"]["service_id"],
        tts_service_ids=speech_raw["tts"],
        voice_gender=speech_raw["tts"]["voice_gender"],
        ulca_api_key=_require_env("ULCA_API_KEY"),
        ulca_user_id=_require_env("ULCA_USER_ID"),
        ulca_authorization=_require_env("ULCA_AUTHORIZATION"),
    )

    is_debug = os.getenv("APP_ENV", "production").lower() == "development"

    return AppConfig(
        name=app_raw["name"],
        version=app_raw["version"],
        debug=is_debug,
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 5003)),
        gemini_api_key=_require_env("GEMINI_API_KEY"),
        rag=rag,
        speech=speech,
        supported_languages=raw["supported_languages"],
    )
