from __future__ import annotations

from pathlib import Path
from collections import Counter

import mne
import pandas as pd
import torch
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader

from src.utils.eeg_utils import (
    load_yaml,
    extract_subject_id,
    normalize_condition,
    build_label_table,
    pick_eeg_channels,
    replace_non_finite_raw_values,
    build_epoch_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "preprocessing.yaml"
EYES_SPLIT_SEC = 300.0

class EEGDataset(Dataset):
    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        eeg_raw: str | Path | None = None,
        questionnaire: str | Path | None = None,
        lowcut: float | None = None,
        highcut: float | None = None,
        notch: float | None = None,
        window: float | None = None,
        overlap: float | None = None,
        condition: str | None = None,
    ):
        self.config_path = Path(config_path)

        if not self.config_path.is_absolute():
            self.config_path = PROJECT_ROOT / self.config_path

        cfg = load_yaml(self.config_path)

        eeg_cfg = cfg["eeg"]
        questionnaire_cfg = cfg["questionnaire"]

        self.eeg_raw = (
            Path(eeg_raw)
            if eeg_raw is not None
            else PROJECT_ROOT / eeg_cfg["raw_data_dir"]
        )

        self.questionnaire = (
            Path(questionnaire)
            if questionnaire is not None
            else PROJECT_ROOT / questionnaire_cfg["raw_data_path"]
        )

        self.lowcut = lowcut if lowcut is not None else eeg_cfg.get("lowcut", 1.0)
        self.highcut = highcut if highcut is not None else eeg_cfg.get("highcut", 25.0)
        self.notch = notch if notch is not None else eeg_cfg.get("notch", None)

        self.window = (
            window
            if window is not None
            else eeg_cfg.get("window_duration", eeg_cfg.get("window", 2.0))
        )

        self.overlap = overlap if overlap is not None else eeg_cfg.get("overlap", 0.5)
        self.target_fs = eeg_cfg.get("target_fs", 128)
        self.scale_to_uv = bool(eeg_cfg.get("scale_to_uv", True))
        self.reject_p2p_uv = eeg_cfg.get("reject_p2p_uv", None)

        yaml_condition = eeg_cfg.get("condition", "complete")
        self.condition = normalize_condition(
            condition if condition is not None else yaml_condition
        )

        self.label_table = build_label_table(self.questionnaire)

        self.id_to_label_code = dict(
            zip(
                self.label_table["subject_id"],
                self.label_table["pain_scale_code"],
            )
        )

        self.id_to_label_name = dict(
            zip(
                self.label_table["subject_id"],
                self.label_table["pain_scale_label"],
            )
        )

        self.samples = []
        self._load_subjects()

        if len(self.samples) == 0:
            raise ValueError("No EEG subjects were loaded.")

    def _load_subjects(self):
        gdf_files = sorted(self.eeg_raw.glob("*.gdf"))

        if len(gdf_files) == 0:
            raise FileNotFoundError(f"No .gdf files were found in {self.eeg_raw}")

        for file_path in gdf_files:
            subject_id = extract_subject_id(file_path)

            if subject_id not in self.id_to_label_code:
                continue

            eeg_data, metadata = self._process_gdf(file_path)

            if eeg_data is None:
                continue

            self.samples.append(
                {
                    "subject_id": subject_id,
                    "file": file_path.name,
                    "eeg": torch.tensor(eeg_data, dtype=torch.float32),
                    "label": torch.tensor(
                        self.id_to_label_code[subject_id],
                        dtype=torch.long,
                    ),
                    "label_name": self.id_to_label_name[subject_id],
                    "metadata": metadata,
                }
            )

    def _process_gdf(self, file_path: Path):
        raw = mne.io.read_raw_gdf(
            str(file_path),
            preload=True,
            verbose=False,
        )

        raw = pick_eeg_channels(raw)
        raw = replace_non_finite_raw_values(raw)

        original_sfreq = float(raw.info["sfreq"])
        original_duration_sec = raw.n_times / original_sfreq

        raw.set_eeg_reference("average", verbose=False)

        if self.notch is not None:
            raw.notch_filter([self.notch], verbose=False)

        raw.filter(
            self.lowcut,
            self.highcut,
            fir_design="firwin",
            verbose=False,
        )

        if self.target_fs is not None and float(raw.info["sfreq"]) != float(self.target_fs):
            raw.resample(
                float(self.target_fs),
                npad="auto",
                verbose=False,
            )

        step = self.window * (1 - self.overlap)

        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=self.window,
            overlap=self.window - step,
            preload=True,
            verbose=False,
        )

        eeg_data = epochs.get_data()

        if eeg_data.size == 0:
            print(f"No epochs were generated for {file_path}")
            return None, None

        eeg_data_uv = eeg_data * 1e6

        original_n_epochs = int(eeg_data.shape[0])
        rejected_n_epochs = 0
        kept_n_epochs = original_n_epochs

        if self.reject_p2p_uv is not None:
            p2p_uv = np.ptp(eeg_data_uv, axis=2)
            good_epoch_mask = np.all(p2p_uv <= float(self.reject_p2p_uv), axis=1)

            rejected_n_epochs = int(np.sum(~good_epoch_mask))
            kept_n_epochs = int(np.sum(good_epoch_mask))

            if kept_n_epochs == 0:
                print(
                    f"All epochs were rejected for {file_path} "
                    f"using reject_p2p_uv={self.reject_p2p_uv}"
                )
                return None, None

            epochs = epochs[good_epoch_mask]
            eeg_data = eeg_data[good_epoch_mask]
            eeg_data_uv = eeg_data_uv[good_epoch_mask]

        if self.scale_to_uv:
            eeg_data = eeg_data_uv

        metadata = build_epoch_metadata(
            eeg_data=eeg_data,
            epochs=epochs,
            raw=raw,
            original_sfreq=original_sfreq,
            original_duration_sec=original_duration_sec,
            window=self.window,
            overlap=self.overlap,
            eyes_split_sec=EYES_SPLIT_SEC,
        )

        metadata["artifact_rejection"] = {
            "reject_p2p_uv": self.reject_p2p_uv,
            "original_n_epochs": original_n_epochs,
            "kept_n_epochs": kept_n_epochs,
            "rejected_n_epochs": rejected_n_epochs,
            "rejected_fraction": rejected_n_epochs / max(original_n_epochs, 1),
        }

        return eeg_data, metadata

    def get_eeg(self, idx: int):
        sample = self.samples[idx]
        mask = sample["metadata"]["condition_masks"][self.condition]

        return sample["eeg"][mask]

    def get_subject_ids(self):
        return [sample["subject_id"] for sample in self.samples]

    def get_labels(self):
        return [int(sample["label"].item()) for sample in self.samples]

    def get_summary_dataframe(self):
        rows = []

        for idx, sample in enumerate(self.samples):
            metadata = sample["metadata"]
            selected_eeg = self.get_eeg(idx)
            artifact_info = metadata.get("artifact_rejection", {})

            rows.append(
                {
                    "subject_id": sample["subject_id"],
                    "file": sample["file"],
                    "label_name": sample["label_name"],
                    "label_code": int(sample["label"].item()),
                    "selected_condition": self.condition,
                    "original_sfreq": metadata["original_sfreq"],
                    "sfreq": metadata["sfreq"],
                    "duration_sec": metadata["duration_sec"],
                    "n_channels": metadata["n_channels"],
                    "n_samples": metadata["n_samples"],
                    "complete_n_epochs": metadata["complete_n_epochs"],
                    "open_n_epochs": metadata["open_n_epochs"],
                    "closed_n_epochs": metadata["closed_n_epochs"],
                    "transition_n_epochs": metadata["transition_n_epochs"],
                    "complete_shape": metadata["complete_shape"],
                    "open_shape": metadata["open_shape"],
                    "closed_shape": metadata["closed_shape"],
                    "selected_shape": tuple(selected_eeg.shape),
                    "reject_p2p_uv": artifact_info.get("reject_p2p_uv"),
                    "original_n_epochs_before_rejection": artifact_info.get("original_n_epochs"),
                    "kept_n_epochs_after_rejection": artifact_info.get("kept_n_epochs"),
                    "rejected_n_epochs": artifact_info.get("rejected_n_epochs"),
                    "rejected_fraction": artifact_info.get("rejected_fraction"),
                }
            )

        return pd.DataFrame(rows)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        return (
            sample["subject_id"],
            self.get_eeg(idx),
            sample["label"],
        )

def create_kfold_dataloaders(dataset, k=10, batch_size=32, shuffle=True):
    subjects, labels, eeg_data = [], [], []

    for i in range(len(dataset)):
        name, eeg_tensor, label = dataset[i]
        subjects.append(name)
        labels.append(label.item())
        eeg_data.append(eeg_tensor)

    outer_gkf = StratifiedGroupKFold(
        n_splits=k,
        shuffle=True,
        random_state=3407,
    )

    folds = []

    class FoldDataset(Dataset):
        def __init__(self, eeg_list, label_list, subject_indices):
            self.X, self.y, self.names = [], [], []

            for idx in subject_indices:
                eeg = eeg_list[idx]
                label = label_list[idx]
                name = subjects[idx]

                for epoch in eeg:
                    self.X.append(epoch)
                    self.y.append(label)
                    self.names.append(name)

            self.X = torch.stack(self.X)
            self.y = torch.tensor(self.y, dtype=torch.long)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx], self.names[idx]

    for trainval_idx, test_idx in outer_gkf.split(
        eeg_data,
        labels,
        groups=subjects,
    ):
        inner_gkf = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=3407,
        )

        train_idx, val_idx = next(
            inner_gkf.split(
                [eeg_data[i] for i in trainval_idx],
                [labels[i] for i in trainval_idx],
                groups=[subjects[i] for i in trainval_idx],
            )
        )

        train_subjects = [trainval_idx[i] for i in train_idx]
        val_subjects = [trainval_idx[i] for i in val_idx]

        train_dataset = FoldDataset(eeg_data, labels, train_subjects)
        val_dataset = FoldDataset(eeg_data, labels, val_subjects)
        test_dataset = FoldDataset(eeg_data, labels, test_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        folds.append((train_loader, val_loader, test_loader))

    return folds


if __name__ == "__main__":
    dataset = EEGDataset(condition="closed")
    summary = dataset.get_summary_dataframe()

    print("\nEEG dataset")
    print(f"Selected condition: {dataset.condition}")
    print(f"Subjects: {len(dataset)}")
    print(f"Config path: {dataset.config_path}")

    print("\nSignal configuration")
    print(f"Original fs: {summary['original_sfreq'].unique().tolist()}")
    print(f"Final fs: {summary['sfreq'].unique().tolist()}")
    print(f"Channels: {summary['n_channels'].unique().tolist()}")
    print(f"Samples per epoch: {summary['n_samples'].unique().tolist()}")
    print(f"Window: {dataset.window} s")
    print(f"Overlap: {dataset.overlap}")
    print(f"Lowcut: {dataset.lowcut}")
    print(f"Highcut: {dataset.highcut}")
    print(f"Notch: {dataset.notch}")
    print(f"Scale to uV: {dataset.scale_to_uv}")

    print("\nClass distribution")
    print(summary["label_name"].value_counts())

    print(f"Reject P2P: {dataset.reject_p2p_uv} uV")
    print(f"Rejected epochs total: {int(summary['rejected_n_epochs'].sum())}")
    print(
        "Rejected fraction mean:",
        round(float(summary["rejected_fraction"].mean()), 4),
    )

    print("\nCondition sizes")
    for condition, column in [
        ("complete", "complete_n_epochs"),
        ("open", "open_n_epochs"),
        ("closed", "closed_n_epochs"),
    ]:
        total_epochs = int(summary[column].sum())
        n_channels = int(summary["n_channels"].iloc[0])
        n_samples = int(summary["n_samples"].iloc[0])

        print(f"{condition}: ({total_epochs}, {n_channels}, {n_samples})")

    folds = create_kfold_dataloaders(
        dataset,
        k=5,
        batch_size=32,
        shuffle=True,
    )

    train_loader, val_loader, test_loader = folds[0]

    print("\nFold 1")
    for split_name, loader in {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }.items():
        X = loader.dataset.X
        y = loader.dataset.y
        names = loader.dataset.names

        subject_ids = sorted(set(int(subject) for subject in names))
        epoch_label_counts = Counter(y.tolist())

        subject_to_label = {}

        for subject, label in zip(names, y.tolist()):
            subject_to_label[int(subject)] = int(label)

        subject_label_counts = Counter(subject_to_label.values())

        print(f"{split_name}")
        print(f"  subjects: {subject_ids}")
        print(f"  X shape: {tuple(X.shape)}")
        print(f"  y shape: {tuple(y.shape)}")
        print(f"  epoch label counts: {dict(epoch_label_counts)}")
        print(f"  subject label counts: {dict(subject_label_counts)}")