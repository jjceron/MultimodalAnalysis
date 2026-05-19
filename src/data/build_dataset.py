from __future__ import annotations

from pathlib import Path
from collections import Counter

import mne
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader, Subset

from src.utils.eeg_utils import (
    load_yaml,
    extract_subject_id,
    normalize_condition,
    build_label_table,
    pick_eeg_channels,
    replace_non_finite_raw_values,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "preprocessing.yaml"
EYES_SPLIT_SEC = 300.0


class EEGDataset(Dataset):
    """
    Dataset EEG a nivel de sujeto.

    Cada item es un registro continuo de una condición:

        X_s: (C, T_s)
        y_s: etiqueta del sujeto

    No se crean épocas, ventanas ni trials.
    """

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        eeg_raw: str | Path | None = None,
        questionnaire: str | Path | None = None,
        lowcut: float | None = None,
        highcut: float | None = None,
        notch: float | None = None,
        condition: str | None = None,
        eyes_split_sec: float = EYES_SPLIT_SEC,
        condition_duration_sec: float | None = EYES_SPLIT_SEC,
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

        self.target_fs = eeg_cfg.get("target_fs", 128)
        self.scale_to_uv = bool(eeg_cfg.get("scale_to_uv", True))

        yaml_condition = eeg_cfg.get("condition", "complete")
        self.condition = normalize_condition(
            condition if condition is not None else yaml_condition
        )

        self.eyes_split_sec = float(eyes_split_sec)
        self.condition_duration_sec = condition_duration_sec

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

        sfreq = float(raw.info["sfreq"])

        eeg_data, crop_info = self._get_condition_data(raw)

        if eeg_data.size == 0:
            print(f"No data were extracted for {file_path} with condition={self.condition}")
            return None, None

        if self.scale_to_uv:
            eeg_data = eeg_data * 1e6

        metadata = {
            "original_sfreq": original_sfreq,
            "original_duration_sec": original_duration_sec,
            "sfreq": sfreq,
            "duration_sec": raw.n_times / sfreq,
            "n_channels": int(eeg_data.shape[0]),
            "n_samples": int(eeg_data.shape[1]),
            "selected_condition": self.condition,
            "selected_duration_sec": eeg_data.shape[1] / sfreq,
            "selected_shape": tuple(eeg_data.shape),
            "scale_to_uv": self.scale_to_uv,
            **crop_info,
        }

        return eeg_data, metadata

    def _get_condition_data(self, raw: mne.io.BaseRaw):
        """
        Extrae una condición continua del registro.

        complete:
            toma todo el registro.

        open:
            toma [0 s, 300 s), por defecto.

        closed:
            toma [300 s, 600 s), por defecto.

        Si condition_duration_sec=None:
            open toma [0, eyes_split_sec)
            closed toma [eyes_split_sec, final_del_registro)
        """

        sfreq = float(raw.info["sfreq"])
        n_times = int(raw.n_times)
        total_duration_sec = n_times / sfreq

        if self.condition == "complete":
            start_sec = 0.0
            end_sec = total_duration_sec

        elif self.condition == "open":
            start_sec = 0.0

            if self.condition_duration_sec is None:
                end_sec = self.eyes_split_sec
            else:
                end_sec = start_sec + float(self.condition_duration_sec)

        elif self.condition == "closed":
            start_sec = self.eyes_split_sec

            if self.condition_duration_sec is None:
                end_sec = total_duration_sec
            else:
                end_sec = start_sec + float(self.condition_duration_sec)

        else:
            raise ValueError(
                f"Unknown condition={self.condition}. "
                "Expected 'complete', 'open' or 'closed'."
            )

        start_sample = int(round(start_sec * sfreq))
        end_sample = int(round(end_sec * sfreq))

        start_sample = max(0, min(start_sample, n_times))
        end_sample = max(start_sample, min(end_sample, n_times))

        eeg_data = raw.get_data(
            start=start_sample,
            stop=end_sample,
        )

        crop_info = {
            "condition_start_sec": start_sec,
            "condition_end_sec_requested": end_sec,
            "condition_start_sample": start_sample,
            "condition_end_sample": end_sample,
            "condition_end_sec_real": end_sample / sfreq,
        }

        return eeg_data, crop_info

    def get_eeg(self, idx: int):
        return self.samples[idx]["eeg"]

    def get_subject_ids(self):
        return [sample["subject_id"] for sample in self.samples]

    def get_labels(self):
        return [int(sample["label"].item()) for sample in self.samples]

    def get_summary_dataframe(self):
        rows = []

        for sample in self.samples:
            metadata = sample["metadata"]

            rows.append(
                {
                    "subject_id": sample["subject_id"],
                    "file": sample["file"],
                    "label_name": sample["label_name"],
                    "label_code": int(sample["label"].item()),
                    "selected_condition": metadata["selected_condition"],
                    "original_sfreq": metadata["original_sfreq"],
                    "sfreq": metadata["sfreq"],
                    "original_duration_sec": metadata["original_duration_sec"],
                    "duration_sec": metadata["duration_sec"],
                    "selected_duration_sec": metadata["selected_duration_sec"],
                    "n_channels": metadata["n_channels"],
                    "n_samples": metadata["n_samples"],
                    "selected_shape": metadata["selected_shape"],
                    "condition_start_sec": metadata["condition_start_sec"],
                    "condition_end_sec_requested": metadata["condition_end_sec_requested"],
                    "condition_end_sec_real": metadata["condition_end_sec_real"],
                    "scale_to_uv": metadata["scale_to_uv"],
                }
            )

        return pd.DataFrame(rows)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        return (
            sample["subject_id"],
            sample["eeg"],
            sample["label"],
        )


def create_kfold_dataloaders(
    dataset: EEGDataset,
    k: int = 5,
    batch_size: int = 4,
    shuffle: bool = True,
    split_seed: int = 3407,
    inner_splits: int = 5,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """
    Crea particiones train/val/test a nivel de sujeto.

    Cada muestra del DataLoader corresponde a un sujeto completo:

        batch["X"]:          (B_subjects, C, T_max)
        batch["y"]:          (B_subjects,)
        batch["subject_id"]: lista de IDs de sujetos
        batch["lengths"]:    número real de muestras temporales por sujeto
        batch["mask"]:       máscara temporal válida, útil si hay padding
    """

    subjects = np.array(dataset.get_subject_ids())
    labels = np.array(dataset.get_labels())

    label_counts = Counter(labels.tolist())

    if min(label_counts.values()) < k:
        raise ValueError(
            f"k={k} is too large for the class distribution {dict(label_counts)}. "
            f"Use k <= {min(label_counts.values())}."
        )

    def collate_subjects(batch):
        subject_ids, eeg_list, label_list = zip(*batch)

        lengths = torch.tensor(
            [eeg.shape[-1] for eeg in eeg_list],
            dtype=torch.long,
        )

        n_channels_set = {int(eeg.shape[0]) for eeg in eeg_list}

        if len(n_channels_set) != 1:
            raise ValueError(
                f"All subjects must have the same number of channels. "
                f"Found: {n_channels_set}"
            )

        batch_size_actual = len(eeg_list)
        n_channels = int(eeg_list[0].shape[0])
        max_len = int(lengths.max().item())

        same_length = all(int(eeg.shape[-1]) == max_len for eeg in eeg_list)

        if same_length:
            X = torch.stack(eeg_list, dim=0)
        else:
            X = eeg_list[0].new_zeros(
                batch_size_actual,
                n_channels,
                max_len,
            )

            for i, eeg in enumerate(eeg_list):
                T = eeg.shape[-1]
                X[i, :, :T] = eeg

        mask = torch.zeros(
            batch_size_actual,
            max_len,
            dtype=torch.bool,
        )

        for i, length in enumerate(lengths):
            mask[i, : int(length.item())] = True

        y = torch.stack(
            [
                torch.as_tensor(label, dtype=torch.long).reshape(())
                for label in label_list
            ],
            dim=0,
        )

        return {
            "X": X,
            "y": y,
            "subject_id": list(subject_ids),
            "lengths": lengths,
            "mask": mask,
        }

    outer_gkf = StratifiedGroupKFold(
        n_splits=k,
        shuffle=True,
        random_state=split_seed,
    )

    folds = []

    for trainval_idx, test_idx in outer_gkf.split(
        X=np.zeros(len(labels)),
        y=labels,
        groups=subjects,
    ):
        trainval_idx = np.array(trainval_idx)
        test_idx = np.array(test_idx)

        trainval_labels = labels[trainval_idx]
        trainval_subjects = subjects[trainval_idx]

        trainval_label_counts = Counter(trainval_labels.tolist())
        effective_inner_splits = min(
            inner_splits,
            min(trainval_label_counts.values()),
        )

        if effective_inner_splits < 2:
            raise ValueError(
                "Not enough subjects per class to create an inner "
                "train/validation split."
            )

        inner_gkf = StratifiedGroupKFold(
            n_splits=effective_inner_splits,
            shuffle=True,
            random_state=split_seed,
        )

        train_rel_idx, val_rel_idx = next(
            inner_gkf.split(
                X=np.zeros(len(trainval_idx)),
                y=trainval_labels,
                groups=trainval_subjects,
            )
        )

        train_idx = trainval_idx[train_rel_idx]
        val_idx = trainval_idx[val_rel_idx]

        train_dataset = Subset(dataset, train_idx.tolist())
        val_dataset = Subset(dataset, val_idx.tolist())
        test_dataset = Subset(dataset, test_idx.tolist())

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_subjects,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_subjects,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_subjects,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        folds.append((train_loader, val_loader, test_loader))

    return folds


if __name__ == "__main__":
    dataset = EEGDataset(condition="closed")

    summary = dataset.get_summary_dataframe()

    print("\nEEG continuous-subject dataset")
    print(f"Selected condition: {dataset.condition}")
    print(f"Subjects: {len(dataset)}")
    print(f"Config path: {dataset.config_path}")

    print("\nSignal configuration")
    print(f"Original fs: {summary['original_sfreq'].unique().tolist()}")
    print(f"Final fs: {summary['sfreq'].unique().tolist()}")
    print(f"Channels: {summary['n_channels'].unique().tolist()}")
    print(f"Samples per subject: {summary['n_samples'].unique().tolist()}")
    print(f"Selected duration: {summary['selected_duration_sec'].unique().tolist()}")
    print(f"Lowcut: {dataset.lowcut}")
    print(f"Highcut: {dataset.highcut}")
    print(f"Notch: {dataset.notch}")
    print(f"Scale to uV: {dataset.scale_to_uv}")

    print("\nClass distribution")
    print(summary["label_name"].value_counts())

    print("\nDataset shapes by subject")
    print(summary[["subject_id", "label_name", "selected_shape"]].head())

    folds = create_kfold_dataloaders(
        dataset,
        k=5,
        batch_size=8,
        shuffle=True,
    )

    train_loader, val_loader, test_loader = folds[0]

    print("\nFold 1")

    for split_name, loader in {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }.items():
        subset_indices = loader.dataset.indices

        split_subjects = [
            int(dataset.samples[i]["subject_id"])
            for i in subset_indices
        ]

        split_labels = [
            int(dataset.samples[i]["label"].item())
            for i in subset_indices
        ]

        print(f"\n{split_name}")
        print("  Split-level information")
        print(f"    total subjects in split: {len(split_subjects)}")
        print(f"    subject ids: {sorted(split_subjects)}")
        print(f"    subject label counts: {dict(Counter(split_labels))}")

        print("  DataLoader / mini-batch information")
        print(f"    configured mini-batch size: {loader.batch_size}")
        print(f"    number of mini-batches: {len(loader)}")

        mini_batch_subject_counts = []

        first_batch = None

        for batch_idx, batch in enumerate(loader):
            mini_batch_subject_counts.append(len(batch["subject_id"]))

            if batch_idx == 0:
                first_batch = batch

        print(f"    subjects per mini-batch: {mini_batch_subject_counts}")

        print("  First mini-batch example")
        print(f"    X shape: {tuple(first_batch['X'].shape)}")
        print(f"    y shape: {tuple(first_batch['y'].shape)}")
        print(f"    subject ids in this mini-batch: {first_batch['subject_id']}")
        print(f"    valid lengths: {first_batch['lengths'].tolist()}")