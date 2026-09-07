"""rag/ingest.py — Ingest EEG analysis results into the RAG vector store.

Replaces the old PDF-based ingestion. Receives a structured analysis dict from
eeg.analyzer, formats it into rich text context, and passes it to vectorize.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rag.vectorize import vectorize_documents


def _format_analysis_as_text(analysis: dict[str, Any]) -> str:
    """Convert the structured analysis dict into a rich text block for embedding.

    The text is written in natural language so the embedding model can capture
    semantic meaning and the LLM can reference it fluently in responses.
    """
    lines: list[str] = []

    # Header
    lines.append(f"=== EEG Analysis Report: {analysis['filename']} ===")
    lines.append(f"Duration: {analysis['duration_seconds']} seconds")
    lines.append(f"Channels: {analysis['num_channels']}")
    lines.append(f"Total spikes detected: {analysis['spike_count']}")
    lines.append("")

    # Seizure events
    seizure_events = analysis.get("seizure_events", [])
    if seizure_events:
        lines.append(f"Candidate seizure events: {len(seizure_events)}")
        for idx, evt in enumerate(seizure_events, start=1):
            lines.append(
                f"  Event {idx}: channel {evt['channel']}, "
                f"from {evt['start_sec']}s to {evt['end_sec']}s, "
                f"confidence={evt['confidence']}, energy_ratio={evt['energy_ratio']}"
            )
        lines.append("")
    else:
        lines.append("No candidate seizure events were detected.")
        lines.append("")

    # Channel statistics
    channel_stats = analysis.get("channel_stats", [])
    if channel_stats:
        lines.append("Per-channel statistics:")
        for cs in channel_stats:
            lines.append(
                f"  {cs['channel']}: mean={cs['mean_uv']}µV, "
                f"std={cs['std_uv']}µV, spikes={cs['spike_count']}"
            )
        lines.append("")

    # Wave patterns
    wave_patterns = analysis.get("wave_patterns", [])
    if wave_patterns:
        lines.append("Dominant wave patterns per channel:")
        for wp in wave_patterns:
            band_details = ", ".join(f"{b}={p}" for b, p in wp["band_power"].items())
            lines.append(f"  {wp['channel']}: dominant={wp['dominant_band']} ({band_details})")
        lines.append("")

    # Summary
    summary = analysis.get("summary_text", "")
    if summary:
        lines.append("Summary:")
        lines.append(summary)

    return "\n".join(lines)


def ingest_analysis(analysis: dict[str, Any]) -> int:
    """Ingest a single EEG analysis result into the vector store.

    Parameters
    ----------
    analysis : dict
        Output from ``eeg.analyzer.analyze_eeg()``.

    Returns
    -------
    int
        Number of chunks stored.
    """
    text = _format_analysis_as_text(analysis)
    scan_date = datetime.now(timezone.utc).isoformat()

    # Build document in the format vectorize.py expects.
    document: dict[str, Any] = {
        "text": text,
        "metadata": {
            "filename": analysis["filename"],
            "scan_date": scan_date,
            "duration_seconds": analysis["duration_seconds"],
            "num_channels": analysis["num_channels"],
            "document_type": "eeg_analysis",
        },
    }

    chunk_count = vectorize_documents([document])
    return chunk_count


def run_ingestion_from_analysis(analysis: dict[str, Any]) -> tuple[int, int]:
    """Convenience wrapper matching the old run_ingestion() signature.

    Returns (document_count, chunk_count).
    """
    chunk_count = ingest_analysis(analysis)
    return 1, chunk_count
