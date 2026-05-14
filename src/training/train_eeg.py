from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import argparse
import json
import random
import sys
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

from src.data.build_eeg import EEGDataset, create_kfold_dataloaders
from src.models.eegformer import EEGFormer
from src.models.eegnet import EEGNet
from src.utils.plotting import plot_fold_training_history


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_environment(device: torch.device) -> None:
    print("\nEnvironment")
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Torch CUDA version: {torch.version.cuda}")

    if device.type == "cuda":
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU")


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def to_subject_list(names) -> list[int]:
    if torch.is_tensor(names):
        return [int(x) for x in names.detach().cpu().tolist()]

    output = []
    for name in names:
        if torch.is_tensor(name):
            output.append(int(name.item()))
        else:
            output.append(int(name))
    return output


def compute_class_weights(loader, num_classes: int, device: torch.device):
    subject_to_label = {}

    y_epoch = loader.dataset.y.detach().cpu().numpy().tolist()
    names = loader.dataset.names

    for subject_id, label in zip(names, y_epoch):
        subject_to_label[int(subject_id)] = int(label)

    y = np.array(list(subject_to_label.values()), dtype=int)

    counts = np.bincount(y, minlength=num_classes)
    total = counts.sum()
    safe_counts = np.maximum(counts, 1)

    weights = total / (num_classes * safe_counts)
    weights = weights / weights.mean()

    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    return weights_tensor, counts.tolist(), weights.tolist()


def build_model(
    n_channels: int,
    n_samples: int,
    num_classes: int,
    args,
) -> torch.nn.Module:
    if args.model == "eegnet":
        return EEGNet(
            n_channels=n_channels,
            n_samples=n_samples,
            num_classes=num_classes,
            F1=args.F1,
            D=args.D,
            F2=args.F2,
            kern_length=args.kern_length,
            pool1=args.pool1,
            pool2=args.pool2,
            dropout_eeg=args.dropout_eeg,
            dropout_classifier=args.dropout_classifier,
        )

    if args.model == "eegformer":
        return EEGFormer(
            n_channels=n_channels,
            n_samples=n_samples,
            num_classes=num_classes,
            F1=args.F1,
            D=args.D,
            F2=args.F2,
            kern_length=args.kern_length,
            pool1=args.pool1,
            pool2=args.pool2,
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            num_layers=args.num_layers,
            dropout_eeg=args.dropout_eeg,
            dropout_transformer=args.dropout_transformer,
            dropout_classifier=args.dropout_classifier,
        )

    raise ValueError(f"Unknown model: {args.model}")


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for X, y, _ in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def evaluate_epoch_level(
    model,
    loader,
    criterion,
    device,
    num_classes: int,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(X)
            loss = criterion(logits, y)
            preds = torch.argmax(logits, dim=1)

            batch_size = X.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            y_true.extend(y.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        ),
    }


def evaluate_subject_level(
    model,
    loader,
    device,
    num_classes: int,
    aggregation: str = "majority_vote",
) -> dict:
    model.eval()

    if aggregation not in {"majority_vote", "mean_prob"}:
        raise ValueError("aggregation must be one of: majority_vote, mean_prob.")

    subject_probs = defaultdict(list)
    subject_preds = defaultdict(list)
    subject_true = {}

    with torch.no_grad():
        for X, y, names in loader:
            X = X.to(device, non_blocking=True)

            logits = model(X)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_list = y.detach().cpu().numpy().tolist()
            name_list = to_subject_list(names)

            for subject_id, true_label, pred_label, prob_vector in zip(
                name_list,
                y_list,
                preds,
                probs,
            ):
                subject_probs[subject_id].append(prob_vector)
                subject_preds[subject_id].append(int(pred_label))
                subject_true[subject_id] = int(true_label)

    subjects = []
    y_true = []
    y_pred = []
    y_prob = []
    subject_details = []

    for subject_id in sorted(subject_true.keys()):
        mean_prob = np.mean(subject_probs[subject_id], axis=0)
        votes = np.bincount(subject_preds[subject_id], minlength=num_classes)

        if aggregation == "majority_vote":
            final_pred = int(np.argmax(votes))
        else:
            final_pred = int(np.argmax(mean_prob))

        subjects.append(subject_id)
        y_true.append(subject_true[subject_id])
        y_pred.append(final_pred)
        y_prob.append(mean_prob)

        subject_details.append(
            {
                "subject_id": int(subject_id),
                "true_label": int(subject_true[subject_id]),
                "pred_label": int(final_pred),
                "votes": votes.astype(int).tolist(),
                "mean_prob": mean_prob.tolist(),
                "n_epochs": int(np.sum(votes)),
            }
        )

    y_prob = np.asarray(y_prob)

    metrics = {
        "aggregation": aggregation,
        "subjects": subjects,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob.tolist(),
        "subject_details": subject_details,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            average="macro",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
        ).tolist(),
    }

    if len(np.unique(y_true)) == num_classes:
        try:
            metrics["macro_auc_ovr"] = roc_auc_score(
                y_true,
                y_prob,
                labels=list(range(num_classes)),
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            metrics["macro_auc_ovr"] = np.nan
    else:
        metrics["macro_auc_ovr"] = np.nan

    return metrics


def get_split_subjects(loader) -> list[int]:
    return sorted(set(int(subject) for subject in loader.dataset.names))


def save_json(path: Path, data: dict) -> None:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return value

    clean_data = {key: convert(value) for key, value in data.items()}

    with path.open("w", encoding="utf-8") as file:
        json.dump(clean_data, file, indent=4)


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    fold_idx: int,
    epoch: int,
    best_epoch: int,
    best_val_bacc: float,
    history: list[dict],
    args,
    n_channels: int,
    n_samples: int,
    num_classes: int,
    train_subjects: list[int],
    val_subjects: list[int],
    test_subjects: list[int],
    class_weights: list[float] | None,
    train_label_counts: list[int],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "fold": fold_idx,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_bacc": best_val_bacc,
            "history": history,
            "args": vars(args),
            "condition": args.condition,
            "aggregation": args.aggregation,
            "class_weights": class_weights,
            "train_label_counts": train_label_counts,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "num_classes": num_classes,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--condition", type=str, default="complete")
    parser.add_argument(
        "--model",
        type=str,
        default="eegformer",
        choices=["eegformer", "eegnet"],
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="majority_vote",
        choices=["majority_vote", "mean_prob"],
    )
    

    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--save_root", type=str, default="models")
    parser.add_argument("--experiment_name", type=str, default="eeg_baseline")

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start_fold", type=int, default=1)
    parser.add_argument("--no_class_weights", action="store_true")

    parser.add_argument("--F1", type=int, default=8)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--F2", type=int, default=16)
    parser.add_argument("--kern_length", type=int, default=64)
    parser.add_argument("--pool1", type=int, default=4)
    parser.add_argument("--pool2", type=int, default=4)

    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)

    parser.add_argument("--dropout_eeg", type=float, default=0.4)
    parser.add_argument("--dropout_transformer", type=float, default=0.3)
    parser.add_argument("--dropout_classifier", type=float, default=0.5)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)

    device = get_device()
    print_environment(device)

    save_dir = PROJECT_ROOT / args.save_root / args.experiment_name / args.model / args.condition / args.aggregation
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = PROJECT_ROOT / "configs" / "preprocessing.yaml"
    shutil.copy2(config_path, save_dir / "preprocessing.yaml")

    dataset = EEGDataset(
        config_path=config_path,
        condition=args.condition,
    )
    summary = dataset.get_summary_dataframe()

    n_channels = int(summary["n_channels"].iloc[0])
    n_samples = int(summary["n_samples"].iloc[0])
    num_classes = 3
    total_epochs = int(sum(shape[0] for shape in summary["selected_shape"].tolist()))

    print("\nDataset")
    print(f"Model: {args.model}")
    print(f"Condition: {args.condition}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Subjects: {len(dataset)}")
    print(f"Class distribution: {dict(summary['label_name'].value_counts())}")
    print(f"Selected EEG shape: ({total_epochs}, {n_channels}, {n_samples})")
    print(f"Config path: {dataset.config_path}")
    print(f"Scale to uV: {dataset.scale_to_uv}")
    print(f"Reject P2P uV: {dataset.reject_p2p_uv}")

    folds = create_kfold_dataloaders(
        dataset,
        k=args.k,
        batch_size=args.batch_size,
        shuffle=True,
    )

    all_fold_results = []
    global_cm = np.zeros((num_classes, num_classes), dtype=int)

    for fold_idx, (train_loader, val_loader, test_loader) in enumerate(folds, start=1):
        if fold_idx < args.start_fold:
            continue

        print(f"\nFold {fold_idx}/{args.k}")

        train_subjects = get_split_subjects(train_loader)
        val_subjects = get_split_subjects(val_loader)
        test_subjects = get_split_subjects(test_loader)

        print(f"Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"Val subjects ({len(val_subjects)}): {val_subjects}")
        print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
        print(f"Train X: {tuple(train_loader.dataset.X.shape)}")
        print(f"Val X: {tuple(val_loader.dataset.X.shape)}")
        print(f"Test X: {tuple(test_loader.dataset.X.shape)}")

        model = build_model(
            n_channels=n_channels,
            n_samples=n_samples,
            num_classes=num_classes,
            args=args,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        if args.no_class_weights:
            class_weights_tensor = None
            train_label_counts = np.bincount(
                train_loader.dataset.y.detach().cpu().numpy(),
                minlength=num_classes,
            ).tolist()
            class_weights_list = None
            print("Class weights: disabled")
        else:
            class_weights_tensor, train_label_counts, class_weights_list = compute_class_weights(
                loader=train_loader,
                num_classes=num_classes,
                device=device,
            )
            print(f"Train label counts: {train_label_counts}")
            print(f"Class weights: {[round(w, 4) for w in class_weights_list]}")

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

        best_model_path = save_dir / f"fold_{fold_idx:02d}_best.pt"
        last_model_path = save_dir / f"fold_{fold_idx:02d}_last.pt"
        history_csv_path = save_dir / f"fold_{fold_idx:02d}_history.csv"
        history_plot_path = save_dir / f"fold_{fold_idx:02d}_training_plot.png"

        start_epoch = 1
        best_epoch = 0
        best_val_bacc = -np.inf
        history = []

        if args.resume and last_model_path.exists():
            checkpoint = load_checkpoint(last_model_path, device)

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            start_epoch = int(checkpoint["epoch"]) + 1
            best_epoch = int(checkpoint.get("best_epoch", 0))
            best_val_bacc = float(checkpoint.get("best_val_bacc", -np.inf))
            history = checkpoint.get("history", [])

            print(
                f"Resuming fold {fold_idx} from epoch {start_epoch}. "
                f"Best epoch: {best_epoch}, best val BAcc: {best_val_bacc:.4f}"
            )

        elif args.resume:
            print(f"No checkpoint found for fold {fold_idx}. Starting from scratch.")

        if start_epoch <= args.epochs:
            for epoch in range(start_epoch, args.epochs + 1):
                train_loss = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                )

                val_epoch_metrics = evaluate_epoch_level(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    num_classes=num_classes,
                )

                val_subject_metrics = evaluate_subject_level(
                    model=model,
                    loader=val_loader,
                    device=device,
                    num_classes=num_classes,
                    aggregation=args.aggregation,
                )

                current_val_bacc = val_subject_metrics["balanced_accuracy"]

                row = {
                    "fold": fold_idx,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_epoch_loss": val_epoch_metrics["loss"],
                    "val_epoch_balanced_accuracy": val_epoch_metrics["balanced_accuracy"],
                    "val_subject_accuracy": val_subject_metrics["accuracy"],
                    "val_subject_balanced_accuracy": val_subject_metrics["balanced_accuracy"],
                    "val_subject_macro_f1": val_subject_metrics["macro_f1"],
                    "aggregation": args.aggregation,
                }

                history.append(row)

                print(
                    f"Epoch {epoch:03d}/{args.epochs} | "
                    f"TrainLoss={train_loss:.4f} | "
                    f"ValLoss={val_epoch_metrics['loss']:.4f} | "
                    f"ValSubjBAcc={val_subject_metrics['balanced_accuracy']:.4f} | "
                    f"ValSubjF1={val_subject_metrics['macro_f1']:.4f}"
                )

                if current_val_bacc > best_val_bacc:
                    best_val_bacc = current_val_bacc
                    best_epoch = epoch

                    save_checkpoint(
                        path=best_model_path,
                        model=model,
                        optimizer=optimizer,
                        fold_idx=fold_idx,
                        epoch=epoch,
                        best_epoch=best_epoch,
                        best_val_bacc=best_val_bacc,
                        history=history,
                        args=args,
                        n_channels=n_channels,
                        n_samples=n_samples,
                        num_classes=num_classes,
                        train_subjects=train_subjects,
                        val_subjects=val_subjects,
                        test_subjects=test_subjects,
                        class_weights=class_weights_list,
                        train_label_counts=train_label_counts,
                    )

                save_checkpoint(
                    path=last_model_path,
                    model=model,
                    optimizer=optimizer,
                    fold_idx=fold_idx,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_val_bacc=best_val_bacc,
                    history=history,
                    args=args,
                    n_channels=n_channels,
                    n_samples=n_samples,
                    num_classes=num_classes,
                    train_subjects=train_subjects,
                    val_subjects=val_subjects,
                    test_subjects=test_subjects,
                    class_weights=class_weights_list,
                    train_label_counts=train_label_counts,
                )

                pd.DataFrame(history).to_csv(history_csv_path, index=False)

        else:
            print(f"Fold {fold_idx} already reached epoch {start_epoch - 1}.")

        if history_csv_path.exists():
            plot_fold_training_history(
                history_csv=history_csv_path,
                output_path=history_plot_path,
            )
            print(f"Saved training plot: {history_plot_path}")

        if not best_model_path.exists():
            best_model_path = last_model_path

        checkpoint = load_checkpoint(best_model_path, device)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_subject_metrics = evaluate_subject_level(
            model=model,
            loader=test_loader,
            device=device,
            num_classes=num_classes,
            aggregation=args.aggregation,
        )

        cm = np.asarray(test_subject_metrics["confusion_matrix"])
        global_cm += cm

        fold_result = {
            "fold": fold_idx,
            "best_epoch": best_epoch,
            "best_val_subject_balanced_accuracy": best_val_bacc,
            "aggregation": args.aggregation,
            "train_label_counts": train_label_counts,
            "class_weights": class_weights_list,
            "test_subject_accuracy": test_subject_metrics["accuracy"],
            "test_subject_balanced_accuracy": test_subject_metrics["balanced_accuracy"],
            "test_subject_macro_precision": test_subject_metrics["macro_precision"],
            "test_subject_macro_recall": test_subject_metrics["macro_recall"],
            "test_subject_macro_f1": test_subject_metrics["macro_f1"],
            "test_subject_macro_auc_ovr": test_subject_metrics["macro_auc_ovr"],
            "test_subjects": test_subject_metrics["subjects"],
            "test_y_true": test_subject_metrics["y_true"],
            "test_y_pred": test_subject_metrics["y_pred"],
            "test_subject_details": test_subject_metrics["subject_details"],
            "test_confusion_matrix": test_subject_metrics["confusion_matrix"],
            "model_path": str(best_model_path),
            "history_csv": str(history_csv_path),
            "training_plot": str(history_plot_path),
        }

        all_fold_results.append(fold_result)

        print(
            f"Fold {fold_idx} test | "
            f"Acc={fold_result['test_subject_accuracy']:.4f} | "
            f"BAcc={fold_result['test_subject_balanced_accuracy']:.4f} | "
            f"MacroF1={fold_result['test_subject_macro_f1']:.4f}"
        )

        print("Test subject vote counts")
        for item in test_subject_metrics["subject_details"]:
            print(
                f"  ID{item['subject_id']} | "
                f"true={item['true_label']} | "
                f"pred={item['pred_label']} | "
                f"votes={item['votes']}"
            )

        save_json(save_dir / f"fold_{fold_idx:02d}_test_metrics.json", fold_result)

    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv(save_dir / "results_by_fold.csv", index=False)

    metric_cols = [
        "test_subject_accuracy",
        "test_subject_balanced_accuracy",
        "test_subject_macro_precision",
        "test_subject_macro_recall",
        "test_subject_macro_f1",
        "test_subject_macro_auc_ovr",
    ]

    summary_stats = results_df[metric_cols].agg(["mean", "std"])
    summary_stats.to_csv(save_dir / "summary_metrics.csv")

    save_json(
        save_dir / "global_summary.json",
        {
            "model": args.model,
            "condition": args.condition,
            "aggregation": args.aggregation,
            "num_folds": args.k,
            "config_path": str(config_path),
            "scale_to_uv": dataset.scale_to_uv,
            "reject_p2p_uv": dataset.reject_p2p_uv,
            "global_confusion_matrix": global_cm.tolist(),
            "save_dir": str(save_dir),
        },
    )

    print("\nFinal subject-level results")
    print(summary_stats)

    print("\nGlobal confusion matrix")
    print(global_cm)

    print(f"\nSaved models and metrics in: {save_dir}")