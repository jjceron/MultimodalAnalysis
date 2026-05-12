from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EEGPreprocessingConfig:
    raw_data_dir: Path
    output_dir: Path
    lowcut: float
    highcut: float
    notch: float
    window_duration: float
    overlap: float
    target_fs: int
    segment_duration_sec: float = 300.0

    @classmethod
    def from_yaml(cls, yaml_path: Path, base_dir: Path | None = None) -> "EEGPreprocessingConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        eeg: dict[str, Any] = cfg["eeg"]
        if base_dir is None:
            base_dir = yaml_path.parent.parent

        def resolve_path(value: str) -> Path:
            path = Path(value)
            if path.is_absolute():
                return path
            return base_dir / path

        return cls(
            raw_data_dir=resolve_path(str(eeg["raw_data_dir"])),
            output_dir=resolve_path(str(eeg["output_dir"])),
            lowcut=float(eeg["lowcut"]),
            highcut=float(eeg["highcut"]),
            notch=float(eeg["notch"]),
            window_duration=float(eeg["window_duration"]),
            overlap=float(eeg["overlap"]),
            target_fs=int(eeg["target_fs"]),
            segment_duration_sec=float(eeg.get("segment_duration_sec", 300.0)),
        )
