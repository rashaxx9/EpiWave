"""
test_eeg_pipeline.py — Tests for eeg/parser.py, eeg/analyzer.py, eeg/visualizer.py

Covers:
- Parser: valid EDF, missing file, wrong extension, corrupt file
- Analyzer: output structure, seizure detection logic, spike counts, wave patterns
- Visualizer: chart data structure, correct downsampling, seizure annotation format
"""

from __future__ import annotations

import sys
import os
import struct
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure the chatbot package root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers — synthetic EDF builder (minimal valid EDF header + data)
# ---------------------------------------------------------------------------

def _write_minimal_edf(path: Path, n_channels: int = 4, duration_sec: int = 10, sfreq: int = 256) -> None:
    """Write a minimal but spec-valid EDF file for testing."""
    n_samples_per_channel = duration_sec * sfreq
    n_records = duration_sec         # 1 record per second
    n_samples_per_record = sfreq     # samples per record per channel

    # ---- Header ----
    header_bytes = 256 + n_channels * 256   # EDF header size

    def field(value: str, width: int) -> bytes:
        return value.ljust(width)[:width].encode("ascii")

    header = b""
    header += field("0", 8)                              # version
    header += field("TEST_PATIENT", 80)                  # patient info
    header += field("TEST_RECORD", 80)                   # recording info
    header += field("05.05.26", 8)                       # startdate
    header += field("00.00.00", 8)                       # starttime
    header += field(str(header_bytes), 8)                # bytes in header
    header += field("", 44)                              # reserved
    header += field(str(n_records), 8)                   # number of data records
    header += field("1", 8)                              # duration of each record (seconds)
    header += field(str(n_channels), 4)                  # number of signals

    # Per-channel fields (each 16 or 80 or 8 chars wide)
    ch_labels = [f"EEG{i+1}" for i in range(n_channels)]
    for label in ch_labels:
        header += field(label, 16)                       # label
    for _ in range(n_channels):
        header += field("AgAgCl", 80)                    # transducer
    for _ in range(n_channels):
        header += field("uV", 8)                         # physical dimension
    for _ in range(n_channels):
        header += field("-3276.8", 8)                    # physical min
    for _ in range(n_channels):
        header += field("3276.7", 8)                     # physical max
    for _ in range(n_channels):
        header += field("-32768", 8)                     # digital min
    for _ in range(n_channels):
        header += field("32767", 8)                      # digital max
    for _ in range(n_channels):
        header += field("", 80)                          # prefiltering
    for _ in range(n_channels):
        header += field(str(n_samples_per_record), 8)   # samples per record
    for _ in range(n_channels):
        header += field("", 32)                          # reserved

    assert len(header) == header_bytes, f"Header size mismatch: {len(header)} != {header_bytes}"

    # ---- Data records ----
    # Each record contains n_channels * n_samples_per_record int16 samples
    data_record = struct.pack(
        f"<{n_channels * n_samples_per_record}h",
        *([0] * (n_channels * n_samples_per_record)),
    )

    with open(path, "wb") as f:
        f.write(header)
        for _ in range(n_records):
            f.write(data_record)


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

class TestEDFParser:
    """Tests for eeg/parser.py"""

    def test_parse_valid_edf(self, tmp_path):
        """Parser must return all required keys for a valid EDF file."""
        from eeg.parser import parse_edf

        edf_path = tmp_path / "test.edf"
        _write_minimal_edf(edf_path, n_channels=4, duration_sec=5, sfreq=256)

        result = parse_edf(edf_path)

        assert result["filename"] == "test.edf"
        assert result["num_channels"] == 4
        assert result["sampling_frequency"] == 256.0
        assert result["duration_seconds"] > 0
        assert isinstance(result["channel_names"], list)
        assert len(result["channel_names"]) == 4
        assert isinstance(result["raw_data"], np.ndarray)
        assert result["raw_data"].shape[0] == 4   # (n_channels, n_samples)

    def test_parse_file_not_found(self):
        """Parser must raise FileNotFoundError for non-existent path."""
        from eeg.parser import parse_edf

        with pytest.raises(FileNotFoundError, match="EDF file not found"):
            parse_edf("/tmp/this_file_does_not_exist.edf")

    def test_parse_wrong_extension(self, tmp_path):
        """Parser must reject non-.edf files with a clear ValueError."""
        from eeg.parser import parse_edf

        bad_file = tmp_path / "scan.csv"
        bad_file.write_text("not an edf")

        with pytest.raises(ValueError, match="Unsupported file format"):
            parse_edf(bad_file)

    def test_parse_corrupt_file(self, tmp_path):
        """Parser must raise RuntimeError for corrupt/invalid EDF content."""
        from eeg.parser import parse_edf

        corrupt = tmp_path / "corrupt.edf"
        corrupt.write_bytes(b"this is not valid edf data at all")

        with pytest.raises(RuntimeError, match="Failed to read EDF file"):
            parse_edf(corrupt)

    def test_list_edf_files_recursive(self, tmp_path):
        """list_edf_files must discover .edf files in subdirectories."""
        from eeg.parser import list_edf_files

        subdir = tmp_path / "patient01"
        subdir.mkdir()
        (subdir / "scan01.edf").write_bytes(b"fake")
        (subdir / "scan02.edf").write_bytes(b"fake")
        (tmp_path / "notes.txt").write_text("ignore me")

        found = list_edf_files(tmp_path)
        assert len(found) == 2
        assert all(f.suffix == ".edf" for f in found)

    def test_list_edf_files_empty_dir(self, tmp_path):
        """list_edf_files must return empty list for empty directory."""
        from eeg.parser import list_edf_files

        found = list_edf_files(tmp_path)
        assert found == []


# ---------------------------------------------------------------------------
# Analyzer tests
# ---------------------------------------------------------------------------

def _make_parsed_data(n_channels: int = 4, duration_sec: int = 10, sfreq: int = 256) -> dict[str, Any]:
    """Build a synthetic parsed_data dict (bypasses parser, no EDF file needed)."""
    import mne
    n_samples = duration_sec * sfreq
    raw_data = np.zeros((n_channels, n_samples), dtype=np.float64)
    ch_names = [f"EEG{i+1}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return {
        "filename": "synthetic.edf",
        "filepath": "/tmp/synthetic.edf",
        "channel_names": ch_names,
        "num_channels": n_channels,
        "sampling_frequency": float(sfreq),
        "duration_seconds": float(duration_sec),
        "raw_data": raw_data,
        "info": info,
    }


class TestEEGAnalyzer:
    """Tests for eeg/analyzer.py"""

    def test_output_keys(self):
        """Analyzer output must contain all required keys."""
        from eeg.analyzer import analyze_eeg

        parsed = _make_parsed_data()
        result = analyze_eeg(parsed)

        required_keys = {
            "filename", "duration_seconds", "num_channels",
            "seizure_events", "spike_count", "wave_patterns",
            "channel_stats", "summary_text",
        }
        assert required_keys.issubset(result.keys())

    def test_filename_echoed(self):
        """Analyzer must echo the input filename unchanged."""
        from eeg.analyzer import analyze_eeg

        parsed = _make_parsed_data()
        result = analyze_eeg(parsed)
        assert result["filename"] == "synthetic.edf"

    def test_zero_signal_no_spikes(self):
        """Flat zero signal must produce zero spikes."""
        from eeg.analyzer import analyze_eeg

        parsed = _make_parsed_data()
        # raw_data is already zeros
        result = analyze_eeg(parsed)
        assert result["spike_count"] == 0

    def test_spike_detection_on_synthetic_spikes(self):
        """Injecting large amplitude spikes must be detected."""
        from eeg.analyzer import analyze_eeg

        parsed = _make_parsed_data(n_channels=2, duration_sec=5, sfreq=256)
        # Inject 3 clear spikes on channel 0 at known positions
        spike_positions = [256, 512, 768]   # sample indices
        for pos in spike_positions:
            parsed["raw_data"][0, pos] = 1e-3   # 1 mV — well above 3σ on zero signal

        result = analyze_eeg(parsed)
        assert result["spike_count"] >= 3

    def test_seizure_events_list(self):
        """Seizure events must be a list of dicts with required keys."""
        from eeg.analyzer import analyze_eeg

        parsed = _make_parsed_data()
        result = analyze_eeg(parsed)

        for evt in result["seizure_events"]:
            assert "start_sec" in evt
            assert "end_sec" in evt
            assert "channel" in evt
            assert "confidence" in evt
            assert 0.0 <= evt["confidence"] <= 1.0
            assert evt["start_sec"] < evt["end_sec"]

    def test_wave_patterns_per_channel(self):
        """Wave patterns must have one entry per channel with dominant_band."""
        from eeg.analyzer import analyze_eeg

        n_ch = 4
        parsed = _make_parsed_data(n_channels=n_ch)
        result = analyze_eeg(parsed)

        assert len(result["wave_patterns"]) == n_ch
        valid_bands = {"delta", "theta", "alpha", "beta", "gamma"}
        for wp in result["wave_patterns"]:
            assert wp["dominant_band"] in valid_bands
            assert "band_power" in wp
            assert set(wp["band_power"].keys()) == valid_bands

    def test_summary_text_contains_disclaimer(self):
        """Summary text must contain the educational disclaimer."""
        from eeg.analyzer import analyze_eeg

        parsed = _make_parsed_data()
        result = analyze_eeg(parsed)
        assert "educational" in result["summary_text"].lower()
        assert "neurologist" in result["summary_text"].lower()

    def test_channel_stats_structure(self):
        """Channel stats must have one entry per channel with correct keys."""
        from eeg.analyzer import analyze_eeg

        n_ch = 3
        parsed = _make_parsed_data(n_channels=n_ch)
        result = analyze_eeg(parsed)

        assert len(result["channel_stats"]) == n_ch
        for cs in result["channel_stats"]:
            assert "channel" in cs
            assert "mean_uv" in cs
            assert "std_uv" in cs
            assert "spike_count" in cs


# ---------------------------------------------------------------------------
# Visualizer tests
# ---------------------------------------------------------------------------

class TestEEGVisualizer:
    """Tests for eeg/visualizer.py"""

    def _make_analysis(self, n_channels: int = 2) -> tuple[dict, dict]:
        from eeg.analyzer import analyze_eeg
        parsed = _make_parsed_data(n_channels=n_channels, duration_sec=5)
        analysis = analyze_eeg(parsed)
        return parsed, analysis

    def test_output_keys(self):
        """Visualizer output must contain all required top-level keys."""
        from eeg.visualizer import generate_chart_data

        parsed, analysis = self._make_analysis()
        result = generate_chart_data(parsed, analysis)

        required = {"signal_charts", "seizure_annotations", "spike_markers", "band_power_chart", "metadata"}
        assert required.issubset(result.keys())

    def test_signal_charts_per_channel(self):
        """signal_charts must contain one entry per channel."""
        from eeg.visualizer import generate_chart_data

        n_ch = 3
        parsed, analysis = self._make_analysis(n_channels=n_ch)
        result = generate_chart_data(parsed, analysis)

        assert len(result["signal_charts"]) == n_ch
        for chart in result["signal_charts"]:
            assert "channel" in chart
            assert "amplitude_uv" in chart
            assert "time_sec" in chart
            assert len(chart["amplitude_uv"]) == len(chart["time_sec"])

    def test_max_display_points_enforced(self):
        """Downsampled signal must not exceed MAX_DISPLAY_POINTS per channel."""
        from eeg.visualizer import generate_chart_data, MAX_DISPLAY_POINTS

        parsed, analysis = self._make_analysis(n_channels=2)
        result = generate_chart_data(parsed, analysis)

        for chart in result["signal_charts"]:
            assert len(chart["amplitude_uv"]) <= MAX_DISPLAY_POINTS

    def test_metadata_fields(self):
        """Metadata block must contain required fields with correct types."""
        from eeg.visualizer import generate_chart_data

        parsed, analysis = self._make_analysis()
        result = generate_chart_data(parsed, analysis)

        meta = result["metadata"]
        assert "filename" in meta
        assert "duration_seconds" in meta
        assert "num_channels" in meta
        assert "spike_count" in meta
        assert isinstance(meta["spike_count"], int)

    def test_seizure_annotation_format(self):
        """Every seizure annotation must have required fields."""
        from eeg.visualizer import generate_chart_data

        parsed, analysis = self._make_analysis()
        result = generate_chart_data(parsed, analysis)

        for ann in result["seizure_annotations"]:
            assert "channel" in ann
            assert "start_sec" in ann
            assert "end_sec" in ann
            assert "confidence" in ann
            assert "label" in ann

    def test_band_power_chart_per_channel(self):
        """band_power_chart must have one entry per channel."""
        from eeg.visualizer import generate_chart_data

        n_ch = 4
        parsed, analysis = self._make_analysis(n_channels=n_ch)
        result = generate_chart_data(parsed, analysis)

        assert len(result["band_power_chart"]) == n_ch
        for bp in result["band_power_chart"]:
            assert "channel" in bp
            assert "dominant" in bp
            assert "bands" in bp
