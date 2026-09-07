"""eeg/visualizer.py — Generate chart-ready JSON data from EEG analysis output.

Produces frontend-agnostic data structures compatible with Chart.js, Plotly,
or any charting library. Does NOT render anything — only returns structured data.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Maximum number of data points per channel to keep payloads manageable.
# The raw signal is downsampled to this many points for visualisation.
MAX_DISPLAY_POINTS = 2000


def generate_chart_data(
    parsed_data: dict[str, Any],
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """Build chart-ready JSON-serializable data from parsed EEG + analysis results.

    Parameters
    ----------
    parsed_data : dict
        Output from ``eeg.parser.parse_edf()``.
    analysis : dict
        Output from ``eeg.analyzer.analyze_eeg()``.

    Returns
    -------
    dict with keys:
        - signal_charts: list of per-channel time-series data (downsampled)
        - seizure_annotations: list of time ranges to highlight on the chart
        - spike_markers: per-channel spike timestamps for scatter overlay
        - band_power_chart: per-channel frequency-band power for bar/radar chart
        - metadata: recording summary (filename, duration, num_channels)
    """
    raw_data: np.ndarray = parsed_data["raw_data"]
    sfreq: float = parsed_data["sampling_frequency"]
    ch_names: list[str] = parsed_data["channel_names"]
    duration: float = parsed_data["duration_seconds"]

    # --- 1. Downsampled time-series per channel -------------------------------
    signal_charts = _build_signal_charts(raw_data, sfreq, ch_names)

    # --- 2. Seizure event annotations -----------------------------------------
    seizure_annotations = [
        {
            "channel": evt["channel"],
            "start_sec": evt["start_sec"],
            "end_sec": evt["end_sec"],
            "confidence": evt["confidence"],
            "label": f"Candidate seizure ({evt['confidence']:.0%})",
        }
        for evt in analysis.get("seizure_events", [])
    ]

    # --- 3. Spike markers (timestamps in seconds) -----------------------------
    spike_markers = _build_spike_markers(raw_data, sfreq, ch_names)

    # --- 4. Band-power data for bar / radar charts ----------------------------
    band_power_chart = [
        {
            "channel": wp["channel"],
            "bands": wp["band_power"],
            "dominant": wp["dominant_band"],
        }
        for wp in analysis.get("wave_patterns", [])
    ]

    # --- 5. Metadata -----------------------------------------------------------
    metadata = {
        "filename": analysis["filename"],
        "duration_seconds": round(duration, 2),
        "num_channels": len(ch_names),
        "spike_count": analysis.get("spike_count", 0),
        "seizure_event_count": len(seizure_annotations),
    }

    return {
        "signal_charts": signal_charts,
        "seizure_annotations": seizure_annotations,
        "spike_markers": spike_markers,
        "band_power_chart": band_power_chart,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_signal_charts(
    raw_data: np.ndarray, sfreq: float, ch_names: list[str]
) -> list[dict[str, Any]]:
    """Downsample each channel to MAX_DISPLAY_POINTS and return chart-ready lists."""
    n_samples = raw_data.shape[1]
    # Compute a downsample step so we never exceed MAX_DISPLAY_POINTS per channel.
    step = max(1, n_samples // MAX_DISPLAY_POINTS)

    charts: list[dict[str, Any]] = []
    for ch_idx, ch_name in enumerate(ch_names):
        signal = raw_data[ch_idx, ::step]
        times = np.arange(0, n_samples, step) / sfreq

        charts.append({
            "channel": ch_name,
            # Convert to µV and round for reasonable JSON size
            "amplitude_uv": [round(float(v) * 1e6, 2) for v in signal],
            "time_sec": [round(float(t), 4) for t in times],
        })

    return charts


def _build_spike_markers(
    raw_data: np.ndarray, sfreq: float, ch_names: list[str]
) -> list[dict[str, Any]]:
    """Return per-channel spike timestamps for scatter-plot overlay.

    Uses the same threshold logic as analyzer.py to stay consistent.
    """
    from scipy.signal import find_peaks

    markers: list[dict[str, Any]] = []
    for ch_idx, ch_name in enumerate(ch_names):
        signal = raw_data[ch_idx]
        ch_mean = float(np.mean(signal))
        ch_std = float(np.std(signal))
        # Same 3-sigma threshold as analyzer.py
        threshold = ch_mean + 3.0 * ch_std
        spike_indices, _ = find_peaks(np.abs(signal), height=threshold, distance=int(sfreq * 0.05))

        if len(spike_indices) > 0:
            markers.append({
                "channel": ch_name,
                "spike_times_sec": [round(float(idx / sfreq), 4) for idx in spike_indices],
                "count": len(spike_indices),
            })

    return markers
