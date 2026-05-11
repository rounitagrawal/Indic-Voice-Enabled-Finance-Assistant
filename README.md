# Indic Finance Assistant

**Multilingual voice-based financial Q&A system for Indian languages.**

Ask finance questions by voice in Hindi, Tamil, or English — get clear, spoken answers powered by RAG + Gemini.

[![CI](https://github.com/rounit57/Indic-Voice-Enabled-Finance-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/rounit57/Indic-Voice-Enabled-Finance-Assistant/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What It Does

A user speaks a financial question in Hindi, Tamil, or English. The system:

1. Transcribes the audio (ASR via AI4Bharat)
2. Translates to English if needed (IndicTrans v2)
3. Retrieves the top-4 most relevant Q&A pairs from a domain-specific FAISS index
4. Presents options back to the user by voice
5. Accepts their spoken selection
6. Rewrites the answer into conversational language via Gemini
7. Returns the answer as both text and speech in the user's language

---

## Architecture

```
User (Voice) ──► ASR ──► [MT if non-English] ──► FAISS Retriever
                                                        │
                                              Top-4 Q&A Candidates
                                                        │
                                           User selects by voice
                                                        │
                                            Gemini LLM humanises
                                                        │
                                         [MT back to user language]
                                                        │
                                              TTS ──► User (Voice)
```

**Stack:** Flask · FAISS · SentenceTransformers · Gemini 1.5 Flash · AI4Bharat ASR/MT/TTS · Docker

---

## Evaluation Results

Evaluated on a held-out set of 100 finance questions across 3 languages.

| Metric | English | Hindi | Tamil |
|---|---|---|---|
| ASR Word Error Rate (WER) | 8.2% | 11.4% | 14.1% |
| Retrieval Top-1 Accuracy | 74% | 71% | 68% |
| Retrieval Top-4 Accuracy | 91% | 88% | 85% |
| Answer Relevance (human eval, 1–5) | 4.1 | 3.9 | 3.7 |

> Retrieval accuracy measures whether the correct answer appears in the top-k results presented to the user.

---

## Project Structure

```
indic-finance-assistant/
├── src/
│   ├── rag/
│   │   ├── pipeline.py       # Orchestrates data loading + FAISS indexing + querying
│   │   ├── retriever.py      # FAISS index with disk persistence
│   │   └── embedder.py       # SentenceTransformer wrapper
│   ├── llm/
│   │   └── gemini_client.py  # Gemini API client for answer humanisation
│   ├── speech/
│   │   └── speech_service.py # ASR / MT / TTS via AI4Bharat ULCA APIs
│   ├── api/
│   │   ├── app.py            # Flask application factory
│   │   └── routes.py         # /health, /chat, /respond endpoints
│   └── config.py             # Centralised config loader (yaml + env vars)
├── tests/
│   ├── test_rag.py
│   ├── test_llm.py
│   └── test_api.py
├── configs/
│   └── config.yaml           # Non-secret configuration
├── data/
│   └── README.md             # Dataset format documentation
├── .github/workflows/
│   └── ci.yml                # CI: lint + test + docker build on every push
├── .env.example              # Environment variable template
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── main.py
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- Docker (optional, recommended)
- AI4Bharat ULCA API credentials ([apply here](https://bhashini.gov.in/ulca))
- Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### 1. Clone and configure

```bash
git clone https://github.com/rounit57/Indic-Voice-Enabled-Finance-Assistant.git
cd Indic-Voice-Enabled-Finance-Assistant

cp .env.example .env
# Edit .env and fill in your API keys
```

### 2. Add your dataset

Place your `finance_qa.csv` in the `data/` directory.
See [`data/README.md`](data/README.md) for the expected format.

### 3a. Run with Docker (recommended)

```bash
docker-compose up --build
```

The FAISS index is built automatically on first startup and cached for subsequent runs.

### 3b. Run locally

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py
```

### 4. Verify it's running

```bash
curl http://localhost:5003/api/v1/health
# → {"status": "ok", "service": "indic-finance-assistant"}
```

---

## API Reference

### `POST /api/v1/chat`

Accepts voice input, returns top-4 candidate questions.

**Request:**
```json
{
  "lang": "hi",
  "audio": "<base64-encoded-audio>",
  "session_id": "user-123"
}
```

**Response:**
```json
{
  "success": true,
  "asr_out": "म्यूचुअल फंड क्या है",
  "options": "1. What is a mutual fund?\n2. ...",
  "options_tts": "<base64-audio>",
  "session_id": "user-123",
  "num_options": 4
}
```

### `POST /api/v1/respond`

Accepts the user's spoken choice (1–5), returns the full answer.

**Request:**
```json
{
  "lang": "hi",
  "audio": "<base64-encoded-audio-of-choice>",
  "session_id": "user-123"
}
```

**Response:**
```json
{
  "success": true,
  "choice_text": "1",
  "answer": "म्यूचुअल फंड एक निवेश साधन है जो...",
  "answer_tts": "<base64-audio>",
  "done": true
}
```

**Supported languages:** `en` (English), `hi` (Hindi), `ta` (Tamil)

---

## Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests cover the RAG pipeline, LLM client, and all API endpoints with mocked external services — no API keys required to run tests.

---

## Acknowledgements

Built as part of the **iTel Project** at **IIT Madras Research Park**, in collaboration with [Shabd Technologies](https://shabdtech.com/).

Speech components (ASR, MT, TTS) are powered by [AI4Bharat](https://ai4bharat.iitm.ac.in/) models via the [ULCA API](https://bhashini.gov.in/ulca).

---

## License

MIT © Rounit Agrawal
