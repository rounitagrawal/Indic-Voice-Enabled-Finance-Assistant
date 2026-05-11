"""
src/speech/ASR_TTS_MT.py
─────────────────────────
AI4Bharat / ULCA speech pipeline functions: ASR, MT, TTS.

All API credentials are loaded from environment variables.
They must NEVER be hardcoded in source code.
Add them to your .env file (see .env.example).
"""
import json
import os

import requests


# ── Credentials (loaded from environment, never hardcoded) ────────────────────
_ULCA_API_KEY = os.environ.get("ULCA_API_KEY", "")
_ULCA_USER_ID = os.environ.get("ULCA_USER_ID", "")
_ULCA_AUTHORIZATION = os.environ.get("ULCA_AUTHORIZATION", "")

_CONFIG_URL = (
    "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
)
_INFERENCE_URL = (
    "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
)
_PIPELINE_ID = "64392f96daac500b55c543cd"


def _config_headers() -> dict:
    """Headers used for pipeline config (model-discovery) calls."""
    return {
        "ulcaApikey": _ULCA_API_KEY,
        "userID": _ULCA_USER_ID,
        "Content-Type": "application/json",
    }


def _inference_headers() -> dict:
    """Headers used for inference (ASR / MT / TTS) calls."""
    return {
        "Authorization": _ULCA_AUTHORIZATION,
        "Content-Type": "application/json",
    }


# ── ASR ───────────────────────────────────────────────────────────────────────

def ASR_config_call(language: str) -> tuple[str, str]:
    """Fetch the ASR service ID and inference key for a given language."""
    payload = json.dumps({
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {
                        "sourceLanguage": language,
                    }
                },
            }
        ],
        "pipelineRequestConfig": {
            "pipelineId": _PIPELINE_ID,
        },
    })

    response = requests.post(_CONFIG_URL, headers=_config_headers(), data=payload)
    response.raise_for_status()
    json_object = response.json()

    service_id = (
        json_object["pipelineResponseConfig"][0]["config"][0]["serviceId"]
    )
    inference_api_key = (
        json_object["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
    )
    return service_id, inference_api_key


def ASR_call(language: str, audio_64: str) -> str:
    """Transcribe base64-encoded audio to text using AI4Bharat ASR."""
    service_id, _ = ASR_config_call(language)

    payload = json.dumps({
        "pipelineTasks": [
            {
                "taskType": "asr",
                "config": {
                    "language": {
                        "sourceLanguage": language,
                    },
                    "serviceId": service_id,
                    "audioFormat": "wav",
                    "samplingRate": 16000,
                },
            }
        ],
        "inputData": {
            "audio": [
                {
                    "audioContent": audio_64,
                }
            ]
        },
    })

    response = requests.post(
        _INFERENCE_URL, headers=_inference_headers(), data=payload
    )
    response.raise_for_status()
    json_obj = response.json()
    out_text = json_obj["pipelineResponse"][0]["output"][0]["source"]
    return out_text


# ── MT ────────────────────────────────────────────────────────────────────────

def MT_config_call(source_language: str, target_language: str) -> str:
    """Fetch the MT inference key for a given language pair."""
    payload = json.dumps({
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": source_language,
                        "targetLanguage": target_language,
                    }
                },
            }
        ],
        "pipelineRequestConfig": {
            "pipelineId": _PIPELINE_ID,
        },
    })

    response = requests.post(_CONFIG_URL, headers=_config_headers(), data=payload)
    response.raise_for_status()
    json_object = response.json()

    inference_api_key = (
        json_object["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
    )
    return inference_api_key


def MT_call(source_lan: str, target_lan: str, input_text: str) -> str:
    """Translate text from source language to target language."""
    service_id = "ai4bharat/indictrans-v2-all-gpu--t4"
    MT_config_call(source_lan, target_lan)  # Validates the language pair

    payload = json.dumps({
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": source_lan,
                        "targetLanguage": target_lan,
                    },
                    "serviceId": service_id,
                },
            }
        ],
        "inputData": {
            "input": [
                {
                    "source": input_text,
                }
            ],
            "audio": [
                {
                    "audioContent": None,
                }
            ],
        },
    })

    response = requests.post(
        _INFERENCE_URL, headers=_inference_headers(), data=payload
    )
    response.raise_for_status()
    json_obj = response.json()
    out_text = json_obj["pipelineResponse"][0]["output"][0]["target"]
    return out_text


# ── TTS ───────────────────────────────────────────────────────────────────────

def TTS_config_call(target_lan: str) -> tuple[str, str]:
    """Fetch the TTS service ID and inference key for a given language."""
    payload = json.dumps({
        "pipelineTasks": [
            {
                "taskType": "tts",
                "config": {
                    "language": {
                        "sourceLanguage": target_lan,
                    }
                },
            }
        ],
        "pipelineRequestConfig": {
            "pipelineId": _PIPELINE_ID,
        },
    })

    response = requests.post(_CONFIG_URL, headers=_config_headers(), data=payload)
    response.raise_for_status()
    json_object = response.json()

    service_id = (
        json_object["pipelineResponseConfig"][0]["config"][0]["serviceId"]
    )
    infer_key = (
        json_object["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
    )
    return service_id, infer_key


def TTS_call(language: str, input_text: str) -> str:
    """Convert text to speech, returning base64-encoded audio."""
    service_id, _ = TTS_config_call(language)

    payload = json.dumps({
        "pipelineTasks": [
            {
                "taskType": "tts",
                "config": {
                    "language": {
                        "sourceLanguage": language,
                    },
                    "serviceId": service_id,
                    "gender": "female",
                },
            }
        ],
        "inputData": {
            "input": [
                {
                    "source": input_text,
                }
            ],
            "audio": [
                {
                    "audioContent": None,
                }
            ],
        },
    })

    response = requests.post(
        _INFERENCE_URL, headers=_inference_headers(), data=payload
    )
    response.raise_for_status()
    json_obj = response.json()
    out_audio = json_obj["pipelineResponse"][0]["audio"][0]["audioContent"]
    return out_audio