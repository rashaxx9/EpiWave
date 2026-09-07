"""
test_agents.py — Tests for agents/verify.py, agents/retrieve.py, agents/respond.py

Covers:
- verify_chunks: threshold filtering, sorting, empty input, edge scores
- retrieve_chunks: empty collection guard, result structure
- build_prompt: all three has_scans/verified_chunks branches
- _format_context: with and without chunks
- _format_history: with and without history
- stream_response: delegates correctly to call_llm
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# verify_chunks tests
# ---------------------------------------------------------------------------

class TestVerifyChunks:
    """Tests for agents/verify.verify_chunks()"""

    def _chunk(self, score: float) -> tuple:
        return ("some text", {"filename": "test.edf"}, score)

    def test_filters_below_threshold(self):
        from agents.verify import verify_chunks

        chunks = [self._chunk(0.05), self._chunk(0.50), self._chunk(0.20)]
        result = verify_chunks(chunks, threshold=0.15)
        scores = [c[2] for c in result]
        assert all(s >= 0.15 for s in scores)

    def test_sorted_descending(self):
        from agents.verify import verify_chunks

        chunks = [self._chunk(0.3), self._chunk(0.9), self._chunk(0.6)]
        result = verify_chunks(chunks, threshold=0.0)
        scores = [c[2] for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input_returns_empty(self):
        from agents.verify import verify_chunks
        assert verify_chunks([]) == []

    def test_all_below_threshold_returns_empty(self):
        from agents.verify import verify_chunks
        chunks = [self._chunk(0.05), self._chunk(0.10)]
        assert verify_chunks(chunks, threshold=0.50) == []

    def test_exact_threshold_included(self):
        from agents.verify import verify_chunks
        chunks = [self._chunk(0.15)]
        result = verify_chunks(chunks, threshold=0.15)
        assert len(result) == 1

    def test_uses_config_threshold_by_default(self):
        """verify_chunks default threshold must come from config.VERIFY_THRESHOLD."""
        from agents.verify import verify_chunks
        from config import VERIFY_THRESHOLD
        # Chunk exactly at threshold must pass
        chunks = [("text", {}, VERIFY_THRESHOLD)]
        result = verify_chunks(chunks)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# retrieve_chunks tests
# ---------------------------------------------------------------------------

class TestRetrieveChunks:
    """Tests for agents/retrieve.retrieve_chunks()"""

    def test_returns_empty_when_collection_empty(self, monkeypatch):
        """retrieve_chunks must return [] immediately when the collection is empty."""
        import agents.retrieve as ret

        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        monkeypatch.setattr(ret, "_get_chroma_collection", lambda: mock_collection)

        result = ret.retrieve_chunks("any query")
        assert result == []
        # Must NOT call embed_query when collection is empty
        mock_collection.query.assert_not_called()

    def test_result_structure(self, monkeypatch):
        """retrieve_chunks must return list of (text, metadata, score) tuples."""
        import agents.retrieve as ret

        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "documents": [["chunk text"]],
            "metadatas": [[{"filename": "scan.edf"}]],
            "distances": [[0.3]],
        }
        monkeypatch.setattr(ret, "_get_chroma_collection", lambda: mock_collection)
        monkeypatch.setattr(ret, "embed_query", lambda q: [0.1] * 384)

        result = ret.retrieve_chunks("seizure events")
        assert len(result) == 1
        text, metadata, score = result[0]
        assert text == "chunk text"
        assert metadata["filename"] == "scan.edf"
        assert 0.0 <= score <= 1.0

    def test_score_is_inverted_distance(self, monkeypatch):
        """Score = 1 - cosine_distance. Distance 0.2 → score 0.8."""
        import agents.retrieve as ret

        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "documents": [["text"]],
            "metadatas": [[{}]],
            "distances": [[0.2]],
        }
        monkeypatch.setattr(ret, "_get_chroma_collection", lambda: mock_collection)
        monkeypatch.setattr(ret, "embed_query", lambda q: [0.0] * 384)

        result = ret.retrieve_chunks("query")
        assert abs(result[0][2] - 0.8) < 1e-6

    def test_score_clamped_to_zero(self, monkeypatch):
        """Distance > 1.0 must not yield negative score."""
        import agents.retrieve as ret

        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "documents": [["text"]],
            "metadatas": [[{}]],
            "distances": [[1.5]],
        }
        monkeypatch.setattr(ret, "_get_chroma_collection", lambda: mock_collection)
        monkeypatch.setattr(ret, "embed_query", lambda q: [0.0] * 384)

        result = ret.retrieve_chunks("query")
        assert result[0][2] == 0.0


# ---------------------------------------------------------------------------
# respond.py — prompt building tests
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    """Tests for agents/respond.build_prompt()"""

    def _chunk(self, text: str = "EEG analysis text", score: float = 0.8) -> tuple:
        return (text, {"filename": "scan.edf", "scan_date": "2024-01-01"}, score)

    # --- Case 1: has verified chunks ---

    def test_with_chunks_cites_filename(self):
        from agents.respond import build_prompt

        prompt = build_prompt([self._chunk()], [], "explain the scan", has_scans=True)
        assert "scan.edf" in prompt

    def test_with_chunks_has_citation_instruction(self):
        from agents.respond import build_prompt

        prompt = build_prompt([self._chunk()], [], "what happened?", has_scans=True)
        assert "Cite the scan filename" in prompt

    def test_with_chunks_includes_context_text(self):
        from agents.respond import build_prompt

        prompt = build_prompt([self._chunk("my eeg data here")], [], "explain", has_scans=True)
        assert "my eeg data here" in prompt

    # --- Case 2: has_scans=True but no verified chunks ---

    def test_has_scans_no_chunks_suggests_specific_questions(self):
        from agents.respond import build_prompt

        prompt = build_prompt([], [], "explain it to me", has_scans=True)
        assert "seizure events" in prompt.lower() or "spike" in prompt.lower()

    def test_has_scans_no_chunks_does_not_say_no_scans(self):
        from agents.respond import build_prompt

        prompt = build_prompt([], [], "tell me about it", has_scans=True)
        assert "No EEG scan has been analyzed yet" not in prompt

    # --- Case 3: has_scans=False ---

    def test_no_scans_acknowledges_no_analysis(self):
        from agents.respond import build_prompt

        prompt = build_prompt([], [], "explain it", has_scans=False)
        assert "No EEG scan has been analyzed yet" in prompt

    def test_no_scans_does_not_mandate_analyze_command_every_time(self):
        from agents.respond import build_prompt

        prompt = build_prompt([], [], "hello", has_scans=False)
        # Instruction must say 'only if user asks' — not unconditionally
        assert "Only mention" in prompt or "only mention" in prompt.lower()

    # --- History formatting ---

    def test_history_appears_in_prompt(self):
        from agents.respond import build_prompt

        history = [("what is EEG?", "EEG is brain activity monitoring.")]
        prompt = build_prompt([], history, "tell me more", has_scans=False)
        assert "what is EEG?" in prompt
        assert "EEG is brain activity monitoring." in prompt

    def test_no_history_shows_placeholder(self):
        from agents.respond import build_prompt

        prompt = build_prompt([], [], "hello", has_scans=False)
        assert "No prior conversation" in prompt

    # --- Context formatting ---

    def test_no_chunks_context_placeholder(self):
        from agents.respond import _format_context

        result = _format_context([])
        assert result  # not empty string
        assert "No" in result or "available" in result

    def test_chunks_formatted_with_relevance_score(self):
        from agents.respond import _format_context

        chunks = [("some content", {"filename": "f.edf", "scan_date": "2024"}, 0.75)]
        result = _format_context(chunks)
        assert "0.75" in result
        assert "some content" in result
        assert "f.edf" in result


# ---------------------------------------------------------------------------
# respond.py — stream_response delegation
# ---------------------------------------------------------------------------

class TestStreamResponse:
    """Tests for agents/respond.stream_response()"""

    def test_delegates_to_call_llm(self):
        from agents.respond import stream_response

        with patch("agents.respond.call_llm", return_value=iter(["Hello"])) as mock_llm:
            tokens = list(stream_response([], [], "hi", has_scans=False))

        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args
        assert call_kwargs.kwargs.get("stream") is True
        assert "Hello" in tokens

    def test_system_prompt_contains_epiwave(self):
        from agents.respond import SYSTEM_PROMPT

        assert "EpiWave" in SYSTEM_PROMPT

    def test_system_prompt_contains_educational_disclaimer(self):
        from agents.respond import SYSTEM_PROMPT

        assert "educational" in SYSTEM_PROMPT.lower()
        assert "neurologist" in SYSTEM_PROMPT.lower()

    def test_system_prompt_forbids_medical_diagnosis(self):
        from agents.respond import SYSTEM_PROMPT

        assert "diagnos" in SYSTEM_PROMPT.lower()
