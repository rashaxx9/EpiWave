from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"

# Load the project-local .env once at import time so the constants below are ready
# anywhere in the app without extra setup code.
load_dotenv(ENV_PATH)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
CHROMA_DB_PATH = str((PROJECT_ROOT / os.getenv("CHROMA_DB_PATH", "./chroma_db")).resolve())
TOP_K = int(os.getenv("TOP_K", "5"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

# EEG scan directory — users place .edf files here for analysis.
EEG_SCANS_PATH = PROJECT_ROOT / os.getenv("EEG_SCANS_PATH", "./data/eeg_scans")

CHROMA_COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
VERIFY_THRESHOLD = 0.15

if LLM_PROVIDER not in {"ollama", "openai"}:
    raise ValueError("LLM_PROVIDER must be either 'ollama' or 'openai'.")
