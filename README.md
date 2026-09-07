# EpiWave 🧠⚡

> **Educational AI platform for EEG signal analysis and seizure pattern visualization.**

EpiWave is an AI-powered chatbot that lets you upload real EEG recordings (`.edf` files), analyzes them for seizure-related patterns, and answers your questions about the findings — all in a conversational interface.

> ⚠️ **Disclaimer:** EpiWave is strictly for **educational and informational purposes only**. It is not a medical device, does not provide clinical diagnoses, and must not be used to make any medical decisions. Always consult a qualified neurologist for clinical interpretation of EEG data.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Running Tests](#running-tests)
- [Design Decisions](#design-decisions)

---

## Features

- 📂 **EDF file support** — Reads standard `.edf` EEG recordings recursively from a configurable directory (supports CHB-MIT and similar datasets)
- 🔬 **Signal analysis** — Bandpass filtering (0.5–40 Hz), 3σ spike detection, RMS energy-based seizure candidate detection
- 📊 **Chart data generation** — Frontend-agnostic JSON output per channel (signal traces, seizure annotations, band power)
- 🤖 **RAG-powered Q&A** — Analyzed scan findings are vectorized and retrieved contextually to answer natural-language questions
- 🛡️ **Anti-hallucination guardrails** — EpiWave refuses to invent findings, never provides diagnoses, and always cites the source scan
- 💬 **Conversation memory** — Maintains a rolling window of conversation history across turns
- 🔌 **Dual LLM backend** — Works with local [Ollama](https://ollama.com/) (e.g. Llama 3) or OpenAI (e.g. GPT-4o)

---

## Architecture

```
User input
    │
    ▼
chat.py  ──── 'analyze' command ───▶  EEG Pipeline
    │                                   ├── parser.py    (read .edf → raw signal)
    │                                   ├── analyzer.py  (filter, spike detect, seizure detect)
    │                                   ├── visualizer.py (JSON chart data)
    │                                   └── ingest.py    (format text → ChromaDB)
    │
    ▼  (user question)
retrieve.py  ──▶  embed query  ──▶  ChromaDB cosine search
    │
    ▼
verify.py    ──▶  filter by relevance threshold (0.15)
    │
    ▼
respond.py   ──▶  build prompt (3 states: has context / has scans / no scans)
    │
    ▼
llm_wrapper  ──▶  Ollama  or  OpenAI  (streaming)
    │
    ▼
Terminal output
```

---

## Project Structure

```
llm_api/
├── pytest.ini                    ← Test discovery config
└── chatbot/
    ├── .env                      ← Your local config (not committed)
    ├── .env.example              ← Template for .env
    ├── requirements.txt
    ├── chat.py                   ← Main entry point (CLI)
    ├── config.py                 ← Centralised settings from .env
    ├── llm_wrapper.py            ← Ollama / OpenAI abstraction
    │
    ├── eeg/                      ← EEG processing pipeline
    │   ├── parser.py             ← Read .edf files via MNE
    │   ├── analyzer.py           ← Signal filtering + spike/seizure detection
    │   └── visualizer.py        ← Generate JSON chart data
    │
    ├── rag/                      ← Retrieval-Augmented Generation
    │   ├── ingest.py             ← Format analysis → documents for ChromaDB
    │   └── vectorize.py         ← Chunk, embed, upsert, query ChromaDB
    │
    ├── agents/                   ← LLM agent logic
    │   ├── retrieve.py           ← Query ChromaDB
    │   ├── verify.py             ← Filter chunks by relevance threshold
    │   └── respond.py           ← Build prompt + stream LLM response
    │
    ├── data/
    │   └── eeg_scans/            ← Place your .edf files here (subdirs supported)
    │       ├── chb01/
    │       │   └── chb01_03.edf
    │       └── chb02/
    │           └── chb02_16.edf
    │
    ├── chroma_db/                ← Persistent vector store (auto-created)
    │
    └── tests/
        ├── test_eeg_pipeline.py  ← Parser, analyzer, visualizer (20 tests)
        ├── test_rag_pipeline.py  ← Ingest, vectorize, device, ChromaDB (22 tests)
        ├── test_agents.py        ← Verify, retrieve, respond (26 tests)
        ├── test_security.py      ← Path traversal, injection, guardrails (28 tests)
        └── test_memory.py        ← Memory bounds on 8 GB M2 machines (4 tests)
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| [Ollama](https://ollama.com/) | Latest | For local LLM |
| Llama 3 (via Ollama) | — | `ollama pull llama3` |
| OR OpenAI API key | — | Set `LLM_PROVIDER=openai` in `.env` |

> **Memory note:** Llama 3 requires ~4.7 GB. EpiWave forces the embedding model onto CPU to prevent GPU memory conflicts on 8 GB machines.

---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd llm_api

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r chatbot/requirements.txt

# 4. Pull the local LLM (if using Ollama)
ollama pull llama3
```

---

## Configuration

Copy the example env file and fill in your values:

```bash
cp chatbot/.env.example chatbot/.env
```

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `OLLAMA_MODEL` | `llama3` | Any model available in your Ollama install |
| `OPENAI_API_KEY` | *(empty)* | Required only when `LLM_PROVIDER=openai` |
| `OPENAI_MODEL` | `gpt-4o` | OpenAI model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model for RAG |
| `CHROMA_DB_PATH` | `./chroma_db` | Where ChromaDB persists the vector index |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `MAX_HISTORY_TURNS` | `6` | Conversation turns kept in memory |
| `EEG_SCANS_PATH` | `./data/eeg_scans` | Root directory for `.edf` files |

---

## Usage

```bash
cd llm_api
source .venv/bin/activate
python chatbot/chat.py
```

### Commands inside the chatbot

| Command | Description |
|---|---|
| `analyze` | Lists all `.edf` files found, lets you pick one or more to process |
| `quit` | Exit the application |
| *(anything else)* | Ask EpiWave a question |

### Typical workflow

```
You> analyze

Found 78 EEG scan(s):

  [  1] chb01/chb01_01.edf  (40.4 MB)
  [  2] chb01/chb01_02.edf  (40.4 MB)
  ...
  [  3] chb01/chb01_03.edf  (40.4 MB)   ← has .seizures annotation

Enter file numbers to analyze (e.g. '1', '1-3', '1,4,7', or 'all'):
Selection> 3

Processing 1 file(s)...
--- Processing: chb01/chb01_03.edf ---
  Parsed: 23 channels, 3600.0s, 256.0Hz
  Analysis: 31204 spikes, 512 candidate seizure events
  Chart data ready: 512 seizure annotations, 23 spike channels
  Ingested: 1 document(s), 95 chunks into vector store

EEG analysis pipeline complete. You can now ask questions about the scan(s).

You> how many seizure events were detected?
EpiWave> Based on the analysis of chb01_03.edf, 512 candidate seizure events were
detected across 23 channels over the 3600-second recording...
```

### Selecting files

| Input | Meaning |
|---|---|
| `3` | File number 3 only |
| `1-5` | Files 1 through 5 |
| `1,4,7` | Files 1, 4, and 7 |
| `all` | All discovered files (⚠️ can be slow with large datasets) |

---

## Running Tests

```bash
cd llm_api

# Run all 98 tests
pytest

# Run a specific test file
pytest chatbot/tests/test_security.py -v

# Run with memory output visible
pytest chatbot/tests/test_memory.py -v -s
```

### Test coverage summary

| File | Tests | What it covers |
|---|---|---|
| `test_eeg_pipeline.py` | 20 | EDF parser, analyzer output, spike detection, visualizer chart data |
| `test_rag_pipeline.py` | 22 | Ingest formatting, chunking, embedding device (CPU), ChromaDB connection |
| `test_agents.py` | 26 | verify_chunks, retrieve_chunks, build_prompt (3 branches), stream_response |
| `test_security.py` | 28 | Path traversal, prompt injection, API key leakage, guardrails, unicode |
| `test_memory.py` | 4 | RSS bounds — baseline, model load, chunking delta, embedding delta |
| **Total** | **100** | |

---

## Design Decisions

### Why CPU for embeddings?
On Apple M2 machines with 8 GB unified memory, Llama 3 occupies ~4.7 GB of GPU memory. Loading the sentence-transformers model on MPS (Metal) caused `kIOGPUCommandBufferCallbackErrorOutOfMemory` crashes. The embedding model runs fine on CPU (~90 MB) and is not a bottleneck since it only runs during retrieval and ingestion.

### Why a 0.15 relevance threshold?
The `all-MiniLM-L6-v2` model produces cosine similarity scores in the range 0.05–0.60 for domain-specific technical text against conversational queries. A threshold of 0.30 was too aggressive and caused the chatbot to claim "no scans analyzed" even after successful ingestion. 0.15 is a pragmatic floor that passes genuinely relevant chunks while still filtering noise.

### Why three prompt branches?
The LLM needs to know the difference between:
1. **Chunks found** → cite scan findings
2. **Scans exist but query is too vague** → suggest specific questions (don't lie about having no data)
3. **No scans at all** → general EEG education mode

Collapsing 2 and 3 caused the chatbot to say "no scans analyzed" right after the user had just run `analyze`.

### Why not process all files at once?
The CHB-MIT dataset contains 83+ `.edf` files (~3.3 GB). Each file loaded via MNE into RAM is ~400 MB. Processing all sequentially would take 30+ minutes and risk OOM on 8 GB machines. The interactive selector lets you pick exactly what you need.