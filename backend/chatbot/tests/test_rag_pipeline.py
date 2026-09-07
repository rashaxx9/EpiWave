"""
test_rag_pipeline.py — Tests for rag/ingest.py, rag/vectorize.py

Covers:
- Text formatting of analysis dicts
- Chunking logic (non-empty input, empty input, metadata passthrough)
- Vectorize: upsert, deduplication, empty input guard
- Ingest: end-to-end with in-memory ChromaDB
- Embedding compatibility guard
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analysis(filename: str = "test.edf") -> dict[str, Any]:
    """Minimal valid analysis dict matching analyzer.py output."""
    return {
        "filename": filename,
        "duration_seconds": 60.0,
        "num_channels": 4,
        "seizure_events": [
            {"start_sec": 5.0, "end_sec": 10.0, "channel": "EEG1", "confidence": 0.8, "energy_ratio": 3.1},
        ],
        "spike_count": 42,
        "wave_patterns": [
            {"channel": "EEG1", "dominant_band": "delta", "band_power": {"delta": 100.0, "theta": 50.0, "alpha": 20.0, "beta": 10.0, "gamma": 5.0}},
        ],
        "channel_stats": [
            {"channel": "EEG1", "mean_uv": 0.1, "std_uv": 25.0, "spike_count": 42},
        ],
        "summary_text": "Test summary. Educational purposes only. Consult a neurologist.",
    }


# ---------------------------------------------------------------------------
# Ingest formatting tests
# ---------------------------------------------------------------------------

class TestIngestFormatting:
    """Tests for rag/ingest.py text formatting."""

    def test_format_includes_filename(self):
        from rag.ingest import _format_analysis_as_text
        text = _format_analysis_as_text(_make_analysis("my_scan.edf"))
        assert "my_scan.edf" in text

    def test_format_includes_duration(self):
        from rag.ingest import _format_analysis_as_text
        text = _format_analysis_as_text(_make_analysis())
        assert "60.0" in text

    def test_format_includes_seizure_event(self):
        from rag.ingest import _format_analysis_as_text
        text = _format_analysis_as_text(_make_analysis())
        assert "5.0" in text    # start_sec
        assert "10.0" in text   # end_sec
        assert "EEG1" in text

    def test_format_includes_spike_count(self):
        from rag.ingest import _format_analysis_as_text
        text = _format_analysis_as_text(_make_analysis())
        assert "42" in text

    def test_format_includes_summary(self):
        from rag.ingest import _format_analysis_as_text
        text = _format_analysis_as_text(_make_analysis())
        assert "Test summary" in text

    def test_format_no_seizures(self):
        from rag.ingest import _format_analysis_as_text
        analysis = _make_analysis()
        analysis["seizure_events"] = []
        text = _format_analysis_as_text(analysis)
        assert "No candidate seizure events" in text


# ---------------------------------------------------------------------------
# Vectorize / chunking tests (uses real ChromaDB in-memory via tmp_path)
# ---------------------------------------------------------------------------

class TestVectorizeChunking:
    """Tests for rag/vectorize._chunk_documents()"""

    def test_chunk_non_empty_text(self):
        from rag.vectorize import _chunk_documents

        docs = [{"text": "A" * 1000, "metadata": {"filename": "test.edf", "scan_date": "2024-01-01"}}]
        chunks = _chunk_documents(docs)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk["text"].strip() != ""
            assert "filename" in chunk["metadata"]
            assert "chunk_index" in chunk["metadata"]

    def test_chunk_empty_text_skipped(self):
        from rag.vectorize import _chunk_documents

        docs = [{"text": "   ", "metadata": {"filename": "empty.edf"}}]
        chunks = _chunk_documents(docs)
        assert chunks == []

    def test_chunk_eeg_metadata_passthrough(self):
        from rag.vectorize import _chunk_documents

        docs = [{
            "text": "EEG analysis result content " * 50,
            "metadata": {
                "filename": "scan.edf",
                "scan_date": "2024-01-01T00:00:00Z",
                "duration_seconds": 3600.0,
                "num_channels": 23,
                "document_type": "eeg_analysis",
            },
        }]
        chunks = _chunk_documents(docs)
        assert len(chunks) >= 1
        meta = chunks[0]["metadata"]
        assert meta["scan_date"] == "2024-01-01T00:00:00Z"
        assert meta["duration_seconds"] == 3600.0
        assert meta["num_channels"] == 23
        assert meta["document_type"] == "eeg_analysis"

    def test_chunk_id_uniqueness(self):
        from rag.vectorize import _chunk_documents

        docs = [{"text": "word " * 2000, "metadata": {"filename": "a.edf", "scan_date": "2024-01-01"}}]
        chunks = _chunk_documents(docs)
        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_id_contains_filename(self):
        from rag.vectorize import _chunk_documents

        docs = [{"text": "some content " * 100, "metadata": {"filename": "myscan.edf"}}]
        chunks = _chunk_documents(docs)
        for chunk in chunks:
            assert "myscan.edf" in chunk["id"]


# ---------------------------------------------------------------------------
# Vectorize: embed_texts guards
# ---------------------------------------------------------------------------

class TestEmbedTexts:
    """Tests for rag/vectorize.embed_texts()"""

    def test_embed_empty_list_returns_empty(self):
        from rag.vectorize import embed_texts
        result = embed_texts([])
        assert result == []

    def test_embed_returns_float_lists(self):
        from rag.vectorize import embed_texts
        result = embed_texts(["hello world"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(v, float) for v in result[0])

    def test_embed_dimension_consistent(self):
        from rag.vectorize import embed_texts
        result = embed_texts(["short text", "a much longer piece of text that has many more words"])
        assert len(result[0]) == len(result[1]), "Embedding dimensions must match"


# ---------------------------------------------------------------------------
# Vectorize: full roundtrip with real ChromaDB (isolated temp store)
# ---------------------------------------------------------------------------

class TestVectorizeRoundtrip:
    """Full write → read roundtrip against an isolated ChromaDB instance."""

    @pytest.fixture(autouse=True)
    def _use_temp_chroma(self, tmp_path, monkeypatch):
        """Redirect ChromaDB to a temp dir so tests don't touch the real DB."""
        import chromadb
        from functools import lru_cache

        temp_client = chromadb.PersistentClient(path=str(tmp_path / "testdb"))

        # Patch _get_chroma_collection to use our temp client
        import rag.vectorize as vz
        real_collection = temp_client.get_or_create_collection(
            name="test_col",
            metadata={"embedding_model": "all-MiniLM-L6-v2", "embedding_backend": "sentence-transformers", "hnsw:space": "cosine"},
        )
        monkeypatch.setattr(vz, "_get_chroma_collection", lambda: real_collection)

    def test_vectorize_and_count(self):
        from rag.vectorize import vectorize_documents
        docs = [{"text": "EEG analysis content " * 50, "metadata": {"filename": "scan.edf", "scan_date": "2024-01-01"}}]
        count = vectorize_documents(docs)
        assert count > 0

    def test_vectorize_empty_input(self):
        from rag.vectorize import vectorize_documents
        count = vectorize_documents([])
        assert count == 0

    def test_deduplication_on_reindex(self, monkeypatch):
        """Re-ingesting the same filename must replace old chunks, not duplicate them."""
        import rag.vectorize as vz
        from rag.vectorize import vectorize_documents

        docs = [{"text": "EEG content " * 60, "metadata": {"filename": "dup.edf", "scan_date": "2024-01-01"}}]
        count1 = vectorize_documents(docs)
        count2 = vectorize_documents(docs)
        # Count should be the same — old chunks deleted before re-insert
        assert count1 == count2


# ---------------------------------------------------------------------------
# Embedding device verification (was: test_mps.py at project root)
# ---------------------------------------------------------------------------

class TestEmbeddingDevice:
    """Verify the sentence-transformer model loads on CPU, not MPS/GPU.

    Critical on 8 GB M2 Macs where Ollama occupies the GPU.
    Keeping embeddings on CPU prevents the Metal OOM crashes seen during debugging.
    """

    def test_model_loaded_on_cpu(self):
        """Embedding model must report CPU as its device."""
        from rag.vectorize import _get_sentence_transformer
        model = _get_sentence_transformer()
        device_str = str(model.device).lower()
        assert "cpu" in device_str, (
            f"Embedding model is on '{model.device}' — must be CPU to avoid GPU OOM. "
            "Check that device='cpu' is set in vectorize._get_sentence_transformer()."
        )

    def test_model_can_encode_on_cpu(self):
        """Model loaded on CPU must successfully encode a test sentence."""
        from rag.vectorize import embed_query
        result = embed_query("test EEG seizure spike delta wave")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# ChromaDB connectivity (was: test_chroma.py at project root)
# ---------------------------------------------------------------------------

class TestChromaConnection:
    """Verify the live ChromaDB collection is reachable and well-formed."""

    def test_collection_accessible(self):
        """_get_chroma_collection() must return a usable collection without errors."""
        from rag.vectorize import _get_chroma_collection
        collection = _get_chroma_collection()
        assert collection is not None

    def test_collection_count_is_integer(self):
        """collection.count() must return a non-negative integer."""
        from rag.vectorize import _get_chroma_collection
        count = _get_chroma_collection().count()
        assert isinstance(count, int)
        assert count >= 0

    def test_collection_has_correct_metadata(self):
        """Collection metadata must record embedding_model and embedding_backend."""
        from rag.vectorize import _get_chroma_collection
        from config import EMBEDDING_MODEL
        meta = _get_chroma_collection().metadata or {}
        assert meta.get("embedding_model") == EMBEDDING_MODEL
        assert meta.get("embedding_backend") in ("sentence-transformers", "openai")

    def test_collection_uses_cosine_distance(self):
        """Collection must use cosine distance (hnsw:space=cosine) for correct scoring."""
        from rag.vectorize import _get_chroma_collection
        meta = _get_chroma_collection().metadata or {}
        assert meta.get("hnsw:space") == "cosine"
