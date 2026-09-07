"""eeg/parser.py — Read raw .edf files using the MNE library.

Returns a clean dict of raw EEG data ready for analyzer.py.
Handles corrupt or unsupported files gracefully with clear error messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np


def parse_edf(file_path: str | Path) -> dict[str, Any]:
    """Parse a single .edf file and return structured raw EEG data.

    Parameters
    ----------
    file_path : str | Path
        Absolute or relative path to the .edf file.

    Returns
    -------
    dict with keys:
        - filename (str): basename of the file
        - filepath (str): resolved absolute path
        - channel_names (list[str]): names of all EEG channels
        - num_channels (int): number of channels
        - sampling_frequency (float): sampling rate in Hz
        - duration_seconds (float): total recording length in seconds
        - raw_data (np.ndarray): shape (n_channels, n_samples), voltage values
        - info (mne.Info): MNE info object for downstream processing
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"EDF file not found: {path}")
    if path.suffix.lower() not in (".edf", ".edf+"):
        raise ValueError(f"Unsupported file format '{path.suffix}'. Only .edf files are supported.")

    try:
        # verbose=False suppresses MNE's console logging during file read.
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read EDF file '{path.name}'. The file may be corrupt or "
            f"in an unsupported variant. Details: {exc}"
        ) from exc

    # Extract the raw signal matrix — shape: (n_channels, n_samples)
    raw_data: np.ndarray = raw.get_data()

    return {
        "filename": path.name,
        "filepath": str(path),
        "channel_names": raw.ch_names,
        "num_channels": len(raw.ch_names),
        "sampling_frequency": raw.info["sfreq"],
        "duration_seconds": raw.times[-1],
        "raw_data": raw_data,
        "info": raw.info,
    }


def list_edf_files(scan_dir: Path) -> list[Path]:
    """Return a sorted list of .edf files in the given directory (recursive)."""
    scan_dir.mkdir(parents=True, exist_ok=True)
    return sorted(scan_dir.rglob("*.edf"))
