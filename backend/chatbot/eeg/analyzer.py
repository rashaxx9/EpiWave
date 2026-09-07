"""eeg/analyzer.py — Detect seizure-related patterns and extract features from parsed EEG data.

DISCLAIMER: This analysis is strictly for **educational and demonstration purposes**.
It uses simplified heuristics and threshold-based detection that are NOT suitable for
clinical diagnosis. Real seizure detection requires validated clinical algorithms,
board-certified neurologist review, and FDA-cleared software.

The approach:
1. Bandpass filter the raw signal (0.5–40 Hz) to isolate the clinically relevant band.
2. Detect spikes as amplitude excursions beyond a threshold (mean + 3×std per channel).
3. Identify sustained high-energy bursts as candidate seizure events.
4. Produce a structured output dict with all findings and a human-readable summary.
"""

from __future__ import annotations

from typing import Any

import mne
import numpy as np
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Educational filter parameters — intentionally conservative
# ---------------------------------------------------------------------------
BANDPASS_LOW = 0.5    # Hz — removes DC drift
BANDPASS_HIGH = 40.0  # Hz — removes muscle/EMG artifact
SPIKE_THRESHOLD_STD = 3.0  # Number of std deviations above mean to flag a spike
SEIZURE_MIN_DURATION = 2.0  # Seconds — minimum burst length to qualify as a candidate event
SEIZURE_ENERGY_FACTOR = 2.5  # Energy must exceed baseline by this factor


def analyze_eeg(parsed_data: dict[str, Any]) -> dict[str, Any]:
    """Run educational seizure-pattern analysis on parsed EEG data.

    Parameters
    ----------
    parsed_data : dict
        Output from ``eeg.parser.parse_edf()``.

    Returns
    -------
    dict with keys:
        - filename, duration_seconds, num_channels (echoed from input)
        - seizure_events: list of dicts with start, end, channel, confidence
        - spike_count: total spikes detected across all channels
        - wave_patterns: list of detected wave-pattern descriptors
        - channel_stats: per-channel summary (mean, std, spike count)
        - summary_text: human-readable paragraph summarising findings
    """
    raw_data: np.ndarray = parsed_data["raw_data"]
    sfreq: float = parsed_data["sampling_frequency"]
    ch_names: list[str] = parsed_data["channel_names"]
    filename: str = parsed_data["filename"]
    duration: float = parsed_data["duration_seconds"]
    info: mne.Info = parsed_data["info"]

    # --- 1. Bandpass filter ---------------------------------------------------
    # MNE's filter operates in-place on a Raw object, so we rebuild one from the
    # numpy array to reuse its optimised FIR implementation.
    raw_obj = mne.io.RawArray(raw_data, info, verbose=False)
    raw_obj.filter(BANDPASS_LOW, BANDPASS_HIGH, verbose=False)
    filtered_data: np.ndarray = raw_obj.get_data()

    # --- 2. Per-channel spike detection ---------------------------------------
    total_spike_count = 0
    channel_stats: list[dict[str, Any]] = []
    all_spike_indices: dict[str, np.ndarray] = {}

    for ch_idx, ch_name in enumerate(ch_names):
        signal = filtered_data[ch_idx]
        ch_mean = float(np.mean(signal))
        ch_std = float(np.std(signal))
        threshold = ch_mean + SPIKE_THRESHOLD_STD * ch_std

        # find_peaks returns indices where the signal exceeds the threshold.
        # distance = sfreq*0.05 prevents counting the same spike twice
        # (minimum 50 ms between peaks).
        spike_indices, _ = find_peaks(np.abs(signal), height=threshold, distance=int(sfreq * 0.05))
        spike_count = len(spike_indices)
        total_spike_count += spike_count
        all_spike_indices[ch_name] = spike_indices

        channel_stats.append({
            "channel": ch_name,
            "mean_uv": round(ch_mean * 1e6, 3),   # Convert V → µV for readability
            "std_uv": round(ch_std * 1e6, 3),
            "spike_count": spike_count,
        })

    # --- 3. Candidate seizure event detection ---------------------------------
    # We look for time windows where the signal energy (RMS) in a sliding window
    # is significantly above the channel baseline.
    seizure_events: list[dict[str, Any]] = []
    window_samples = int(SEIZURE_MIN_DURATION * sfreq)

    for ch_idx, ch_name in enumerate(ch_names):
        signal = filtered_data[ch_idx]
        baseline_rms = float(np.sqrt(np.mean(signal ** 2)))

        if baseline_rms == 0:
            continue

        # Slide a window across the signal to compute local RMS
        n_windows = max(1, len(signal) // window_samples)
        for w_idx in range(n_windows):
            start_sample = w_idx * window_samples
            end_sample = min(start_sample + window_samples, len(signal))
            segment = signal[start_sample:end_sample]

            local_rms = float(np.sqrt(np.mean(segment ** 2)))
            energy_ratio = local_rms / baseline_rms

            if energy_ratio >= SEIZURE_ENERGY_FACTOR:
                start_sec = round(start_sample / sfreq, 2)
                end_sec = round(end_sample / sfreq, 2)
                # Confidence is a simple normalised ratio (capped at 1.0).
                # This is a *demo* metric, not a clinically validated score.
                confidence = round(min(energy_ratio / (SEIZURE_ENERGY_FACTOR * 2), 1.0), 3)

                seizure_events.append({
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "channel": ch_name,
                    "confidence": confidence,
                    "energy_ratio": round(energy_ratio, 3),
                })

    # --- 4. Wave pattern descriptors ------------------------------------------
    wave_patterns = _detect_wave_patterns(filtered_data, sfreq, ch_names)

    # --- 5. Human-readable summary --------------------------------------------
    summary = _build_summary(filename, duration, ch_names, total_spike_count, seizure_events, wave_patterns)

    return {
        "filename": filename,
        "duration_seconds": round(duration, 2),
        "num_channels": len(ch_names),
        "seizure_events": seizure_events,
        "spike_count": total_spike_count,
        "wave_patterns": wave_patterns,
        "channel_stats": channel_stats,
        "summary_text": summary,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_wave_patterns(
    filtered_data: np.ndarray, sfreq: float, ch_names: list[str]
) -> list[dict[str, Any]]:
    """Identify dominant frequency-band activity per channel.

    We compute a simple FFT and check which clinical bands carry the most power:
      - Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), Gamma (30–40 Hz)

    This is purely educational — real clinical analysis uses multitaper or Welch PSD.
    """
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 40.0),
    }
    patterns: list[dict[str, Any]] = []

    for ch_idx, ch_name in enumerate(ch_names):
        signal = filtered_data[ch_idx]
        n = len(signal)
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)

        band_power: dict[str, float] = {}
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            band_power[band_name] = float(np.sum(fft_vals[mask] ** 2))

        dominant_band = max(band_power, key=band_power.get)  # type: ignore[arg-type]
        patterns.append({
            "channel": ch_name,
            "dominant_band": dominant_band,
            "band_power": {k: round(v, 3) for k, v in band_power.items()},
        })

    return patterns


def _build_summary(
    filename: str,
    duration: float,
    ch_names: list[str],
    spike_count: int,
    seizure_events: list[dict[str, Any]],
    wave_patterns: list[dict[str, Any]],
) -> str:
    """Compose a human-readable summary paragraph for the LLM context."""
    lines = [
        f"EEG Scan Analysis Summary for '{filename}'",
        f"Recording duration: {duration:.1f} seconds across {len(ch_names)} channels.",
        f"Total spikes detected: {spike_count}.",
    ]

    if seizure_events:
        lines.append(f"Candidate seizure events detected: {len(seizure_events)}.")
        for idx, evt in enumerate(seizure_events, start=1):
            lines.append(
                f"  Event {idx}: channel {evt['channel']}, "
                f"{evt['start_sec']}s – {evt['end_sec']}s "
                f"(confidence={evt['confidence']}, energy_ratio={evt['energy_ratio']})"
            )
    else:
        lines.append("No candidate seizure events detected in this recording.")

    # Summarise dominant wave patterns across channels
    dominant_counts: dict[str, int] = {}
    for wp in wave_patterns:
        band = wp["dominant_band"]
        dominant_counts[band] = dominant_counts.get(band, 0) + 1
    if dominant_counts:
        band_summary = ", ".join(f"{band} ({count} channels)" for band, count in dominant_counts.items())
        lines.append(f"Dominant wave patterns: {band_summary}.")

    lines.append(
        "\nDISCLAIMER: This analysis is for educational and informational purposes only. "
        "It does not constitute a medical diagnosis. Consult a qualified neurologist for "
        "clinical interpretation."
    )

    return "\n".join(lines)
