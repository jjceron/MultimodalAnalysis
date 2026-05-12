from __future__ import annotations

import logging
import re
from pathlib import Path

import mne
import numpy as np

from ..config.settings import EEGPreprocessingConfig

logger = logging.getLogger(__name__)


class EEGProcessor:
    """Classic EEG signal preprocessing for multimodal integration."""

    def __init__(self, config: EEGPreprocessingConfig) -> None:
        self.config = config
        # Standard 10-20 channels often found in these datasets
        self.target_channels = [
            "Fp1",
            "Fp2",
            "F3",
            "F4",
            "C3",
            "C4",
            "P3",
            "P4",
            "O1",
            "O2",
            "F7",
            "F8",
            "T3",
            "T4",
            "T5",
            "T6",
            "Fz",
            "Cz",
            "Pz",
        ]

    def _extract_id(self, filename: str) -> int | None:
        """Extract patient ID from filename (e.g., ID0.gdf -> 0)."""
        match = re.search(r"ID(\d+)", filename)
        return int(match.group(1)) if match else None

    def _prepare_raw(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        raw.pick("eeg")
        raw.rename_channels(lambda x: x.replace("EEG-", ""))

        available = set(raw.ch_names)
        selected = [ch for ch in self.target_channels if ch in available]
        if selected:
            raw.pick(selected)

        raw.filter(self.config.lowcut, self.config.highcut, verbose=False)
        raw.notch_filter(self.config.notch, verbose=False)
        if raw.info["sfreq"] != self.config.target_fs:
            raw.resample(self.config.target_fs, verbose=False)

        return raw

    def _crop_mode(self, raw: mne.io.BaseRaw, mode: str) -> mne.io.BaseRaw | None:
        total_dur = raw.times[-1]
        segment = self.config.segment_duration_sec

        if mode == "open":
            tmin = 0.0
            tmax = min(segment, total_dur)
        elif mode == "closed":
            if total_dur <= segment:
                return None
            tmin = segment
            tmax = min(segment * 2, total_dur)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if tmax <= tmin:
            return None

        raw.crop(tmin=tmin, tmax=tmax, include_tmax=False)
        return raw

    def _build_epochs(self, raw: mne.io.BaseRaw) -> np.ndarray:
        sfreq = raw.info["sfreq"]
        tmax = self.config.window_duration - (1.0 / sfreq)
        overlap = max(0.0, self.config.window_duration * self.config.overlap)
        if overlap >= self.config.window_duration:
            overlap = 0.0

        events = mne.make_fixed_length_events(
            raw, duration=self.config.window_duration, overlap=overlap
        )
        epochs = mne.Epochs(
            raw,
            events,
            tmin=0,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
            reject_by_annotation=False,
        )
        return epochs.get_data(copy=True)

    def process_file(self, file_path: Path, mode: str) -> np.ndarray | None:
        """Process a single EEG file and return epochs (E, C, T)."""
        try:
            raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
            raw = self._prepare_raw(raw)
            raw = self._crop_mode(raw, mode)
            if raw is None:
                return None

            return self._build_epochs(raw)

        except Exception as exc:
            logger.error("Error processing %s: %s", file_path.name, exc)
            return None

    def process_all(self, mode: str) -> None:
        """Process all GDF files in raw directory and save by patient."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        files = list(self.config.raw_data_dir.glob("*.gdf"))
        patient_data: dict[tuple[int, str], list[np.ndarray]] = {}

        for f in files:
            p_id = self._extract_id(f.name)
            if p_id is None:
                continue

            data = self.process_file(f, mode)
            if data is None:
                continue

            key = (p_id, mode)
            patient_data.setdefault(key, []).append(data)

        for (p_id, mode_name), data_list in patient_data.items():
            final_data = np.concatenate(data_list, axis=0)
            output_path = self.config.output_dir / f"ID{p_id}_{mode_name}.npy"
            np.save(output_path, final_data)
            logger.info(
                "Saved patient %s data to %s (shape: %s)",
                p_id,
                output_path.name,
                final_data.shape,
            )

    def build_open_closed_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return epochs with labels (0=open, 1=closed) and patient ids."""
        files = list(self.config.raw_data_dir.glob("*.gdf"))
        x_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        id_list: list[np.ndarray] = []

        for f in files:
            p_id = self._extract_id(f.name)
            if p_id is None:
                continue

            for mode, label in ("open", 0), ("closed", 1):
                data = self.process_file(f, mode)
                if data is None:
                    continue

                x_list.append(data)
                y_list.append(np.full(data.shape[0], label, dtype=int))
                id_list.append(np.full(data.shape[0], p_id, dtype=int))

        if not x_list:
            return np.empty((0, 0, 0)), np.empty((0,), dtype=int), np.empty((0,), dtype=int)

        x = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        ids = np.concatenate(id_list, axis=0)
        return x, y, ids
