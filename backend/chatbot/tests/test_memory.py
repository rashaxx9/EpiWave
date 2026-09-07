"""
test_memory.py — Memory usage sanity checks for the EpiWave pipeline.

Verifies that key operations stay within reasonable memory bounds
so the app doesn't silently balloon on 8 GB M2 machines.

Requires: psutil (install via: pip install psutil)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Dependency guard — skip entire module if psutil is not installed
# ---------------------------------------------------------------------------
psutil = pytest.importorskip("psutil", reason="psutil not installed — run: pip install psutil")


def _rss_mb() -> float:
    """Return current process Resident Set Size in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class TestMemoryUsage:
    """Sanity checks that key operations don't leak excessive memory."""

    def test_process_baseline_under_500mb(self):
        """Baseline process memory must be under 500 MB before heavy ops."""
        rss = _rss_mb()
        print(f"\n  Baseline RSS: {rss:.1f} MB")
        assert rss < 500, f"Baseline memory too high: {rss:.1f} MB"

    def test_embedding_model_load_under_1gb(self):
        """Loading the sentence-transformer model must stay under 1 GB total RSS."""
        from rag.vectorize import _get_sentence_transformer

        _get_sentence_transformer()   # warm up
        rss = _rss_mb()
        print(f"\n  RSS after embedding model load: {rss:.1f} MB")
        assert rss < 1024, f"Memory after embedding load too high: {rss:.1f} MB"

    def test_chunking_large_text_no_leak(self):
        """Chunking a large text document must not leave RSS more than 50 MB higher."""
        from rag.vectorize import _chunk_documents

        before = _rss_mb()
        # Simulate a large analysis document (~500 KB of text)
        large_text = "EEG analysis content with lots of detail. " * 10_000
        docs = [{"text": large_text, "metadata": {"filename": "big.edf", "scan_date": "2024-01-01"}}]
        _chunk_documents(docs)
        after = _rss_mb()
        delta = after - before
        print(f"\n  Memory delta after chunking: {delta:.1f} MB")
        assert delta < 50, f"Chunking leaked {delta:.1f} MB — possible memory issue"

    def test_embed_texts_memory_reasonable(self):
        """Embedding a batch of texts must not leak more than 100 MB."""
        from rag.vectorize import embed_texts

        before = _rss_mb()
        texts = ["EEG spike detection analysis result."] * 50
        embed_texts(texts)
        after = _rss_mb()
        delta = after - before
        print(f"\n  Memory delta after embedding 50 texts: {delta:.1f} MB")
        assert delta < 100, f"Embedding batch leaked {delta:.1f} MB"
