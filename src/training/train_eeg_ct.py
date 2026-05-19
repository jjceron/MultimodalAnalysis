from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

from src.data.build_eeg_ct import EEGDataset, create_kfold_dataloaders
from src.models.eegnet_baseline import EEGNetBaseline


PRESET_CONFIGS = {
    "diagnostic": [
        {"name": "mean_lr1e-3", "agg": "mean", "lr": 1e-3},
        {"name": "mean_lr1e-4", "agg": "mean", "lr": 1e-4},
        {"name": "max_lr1e-4", "agg": "max", "lr": 1e-4},
        {"name": "logsumexp_lr1e-4", "agg": "logsumexp", "lr": 1e-4},
    ],

    "pooling_diagnostic": [
        {
            "name": "topk05_lr1e-4",
            "agg": "topk_mean",
            "topk_ratio": 0.05,
            "lr": 1e-4,
        },
        {
            "name": "topk10_lr1e-4",
            "agg": "topk_mean",
            "topk_ratio": 0.10,
            "lr": 1e-4,
        },
        {
            "name": "topk20_lr1e-4",
            "agg": "topk_mean",
            "topk_ratio": 0.20,
            "lr": 1e-4,
        },
        {
            "name": "q90_lr1e-4",
            "agg": "quantile",
            "quantile_q": 0.90,
            "lr": 1e-4,
        },
        {
            "name": "q95_lr1e-4",
            "agg": "quantile",
            "quantile_q": 0.95,
            "lr": 1e-4,
        },
        {
            "name": "meanmax050_lr1e-4",
            "agg": "meanmax",
            "meanmax_alpha": 0.50,
            "lr": 1e-4,
        },
        {
            "name": "meanmax075_lr1e-4",
            "agg": "meanmax",
            "meanmax_alpha": 0.75,
            "lr": 1e-4,
        },
        {
            "name": "lse_tau05_lr1e-4",
            "agg": "logsumexp",
            "lse_tau": 0.50,
            "lr": 1e-4,
        },
        {
            "name": "lse_tau20_lr1e-4",
            "agg": "logsumexp",
            "lse_tau": 2.00,
            "lr": 1e-4,
        },
        {
            "name": "probmean_lr1e-4",
            "agg": "prob_mean",
            "lr": 1e-4,
        },
    ],
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def standardize_eeg(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
):
    """
    Estandarización por sujeto y canal.

    x:
        (B, C, T)

    mask:
        (B, T)

    return:
        (B, C, T)
    """

    if mask is None:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(eps)

        return (x - mean) / std

    mask_f = mask.float().unsqueeze(1)
    denom = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)

    mean = (x * mask_f).sum(dim=-1, keepdim=True) / denom

    var = (((x - mean) * mask_f) ** 2).sum(dim=-1, keepdim=True) / denom
    std = torch.sqrt(var + eps)

    x_norm = (x - mean) / std

    x_norm = x_norm.masked_fill(~mask.unsqueeze(1), 0.0)

    return x_norm


def get_class_weights(
    dataset: EEGDataset,
    train_loader,
    n_classes: int,
    device: torch.device,
):
    train_indices = train_loader.dataset.indices

    train_labels = [
        int(dataset.samples[i]["label"].item())
        for i in train_indices
    ]

    counts = Counter(train_labels)
    total = sum(counts.values())

    weights = []

    for class_idx in range(n_classes):
        class_count = counts.get(class_idx, 0)

        if class_count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (n_classes * class_count))

    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_metrics(
    targets,
    preds,
    total_loss: float,
    n_classes: int,
):
    n = max(len(targets), 1)

    labels = list(range(n_classes))

    return {
        "loss": total_loss / n,
        "acc": accuracy_score(targets, preds),
        "balanced_acc": balanced_accuracy_score(targets, preds),
        "macro_f1": f1_score(
            targets,
            preds,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            targets,
            preds,
            labels=labels,
        ).tolist(),
        "target_counts": dict(Counter(targets)),
        "pred_counts": dict(Counter(preds)),
    }


def print_metrics(split_name: str, metrics: dict):
    print(
        f"{split_name:>5} | "
        f"loss={metrics['loss']:.4f} | "
        f"acc={metrics['acc']:.4f} | "
        f"bacc={metrics['balanced_acc']:.4f} | "
        f"macro_f1={metrics['macro_f1']:.4f} | "
        f"pred_counts={metrics['pred_counts']}"
    )


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    n_classes: int,
    standardize: bool = True,
    grad_clip: float | None = 1.0,
):
    model.train()

    total_loss = 0.0
    all_targets = []
    all_preds = []

    for batch in loader:
        x = batch["X"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)

        if standardize:
            x = standardize_eeg(x, mask=mask)

        logits_subject, _ = model(x, mask=mask)

        loss = criterion(logits_subject, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size

        preds = logits_subject.argmax(dim=1)

        all_targets.extend(y.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    return compute_metrics(
        targets=all_targets,
        preds=all_preds,
        total_loss=total_loss,
        n_classes=n_classes,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    n_classes: int,
    standardize: bool = True,
    collect_predictions: bool = False,
    split_name: str | None = None,
    fold_id: int | None = None,
    config_name: str | None = None,
):
    model.eval()

    total_loss = 0.0
    all_targets = []
    all_preds = []
    prediction_rows = []

    for batch in loader:
        x = batch["X"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)

        if standardize:
            x = standardize_eeg(x, mask=mask)

        logits_subject, logits_time = model(x, mask=mask)

        loss = criterion(logits_subject, y)

        probs = torch.softmax(logits_subject, dim=1)
        preds = logits_subject.argmax(dim=1)

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size

        y_cpu = y.detach().cpu().tolist()
        pred_cpu = preds.detach().cpu().tolist()
        probs_cpu = probs.detach().cpu().numpy()

        all_targets.extend(y_cpu)
        all_preds.extend(pred_cpu)

        if collect_predictions:
            for i, subject_id in enumerate(batch["subject_id"]):
                row = {
                    "config": config_name,
                    "fold": fold_id,
                    "split": split_name,
                    "subject_id": int(subject_id),
                    "y_true": int(y_cpu[i]),
                    "y_pred": int(pred_cpu[i]),
                    "correct": int(y_cpu[i] == pred_cpu[i]),
                    "length": int(batch["lengths"][i].item()),
                }

                for class_idx in range(n_classes):
                    row[f"prob_{class_idx}"] = float(probs_cpu[i, class_idx])

                prediction_rows.append(row)

    metrics = compute_metrics(
        targets=all_targets,
        preds=all_preds,
        total_loss=total_loss,
        n_classes=n_classes,
    )

    return metrics, prediction_rows


def inspect_first_batch(
    model: nn.Module,
    loader,
    device: torch.device,
    standardize: bool = True,
):
    batch = next(iter(loader))

    x = batch["X"].to(device)
    mask = batch["mask"].to(device)

    if standardize:
        x = standardize_eeg(x, mask=mask)

    model.eval()

    with torch.no_grad():
        logits_subject, logits_time = model(x, mask=mask)

    print("\nShape sanity check")
    print(f"  input X:        {tuple(x.shape)}")
    print(f"  mask:           {tuple(mask.shape)}")
    print(f"  logits_time:    {tuple(logits_time.shape)}")
    print(f"  logits_subject: {tuple(logits_subject.shape)}")
    print(f"  subject ids:    {batch['subject_id']}")
    print(f"  lengths:        {batch['lengths'].tolist()}")


def get_split_subjects_and_labels(dataset: EEGDataset, loader):
    indices = loader.dataset.indices

    subjects = [
        int(dataset.samples[i]["subject_id"])
        for i in indices
    ]

    labels = [
        int(dataset.samples[i]["label"].item())
        for i in indices
    ]

    return subjects, labels


def run_one_config_one_fold(
    dataset: EEGDataset,
    train_loader,
    val_loader,
    test_loader,
    config: dict,
    args,
    device: torch.device,
    fold_idx: int,
):
    config_name = config["name"]
    agg = config["agg"]
    lr = float(config["lr"])

    run_name = f"{config_name}_fold_{fold_idx + 1}"

    print("\n" + "=" * 80)
    print(f"Run: {run_name}")
    print("=" * 80)

    labels = dataset.get_labels()
    n_classes = int(max(labels)) + 1

    train_subjects, train_labels = get_split_subjects_and_labels(dataset, train_loader)
    val_subjects, val_labels = get_split_subjects_and_labels(dataset, val_loader)
    test_subjects, test_labels = get_split_subjects_and_labels(dataset, test_loader)

    print("\nSplit information")
    print(f"  train subjects: {len(train_subjects)} | labels: {dict(Counter(train_labels))}")
    print(f"  val subjects:   {len(val_subjects)} | labels: {dict(Counter(val_labels))}")
    print(f"  test subjects:  {len(test_subjects)} | labels: {dict(Counter(test_labels))}")
    print(f"  train batches:  {len(train_loader)}")
    print(f"  val batches:    {len(val_loader)}")
    print(f"  test batches:   {len(test_loader)}")

    model = EEGNetBaseline(
        n_channels=args.n_channels,
        n_classes=n_classes,
        dropout=args.dropout,
        agg=agg,
        topk_ratio=config.get("topk_ratio", 0.10),
        quantile_q=config.get("quantile_q", 0.95),
        lse_tau=config.get("lse_tau", 1.0),
        meanmax_alpha=config.get("meanmax_alpha", 0.5),
    ).to(device)

    if args.no_class_weights:
        class_weights = None
    else:
        class_weights = get_class_weights(
            dataset=dataset,
            train_loader=train_loader,
            n_classes=n_classes,
            device=device,
        )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay,
    )

    standardize = not args.no_standardize

    print("\nModel/training configuration")
    print(f"  agg:             {agg}")
    print(f"  lr:              {lr}")
    print(f"  weight_decay:    {args.weight_decay}")
    print(f"  dropout:         {args.dropout}")
    print(f"  standardize:     {standardize}")
    print(f"  class_weights:   {None if class_weights is None else class_weights.detach().cpu().tolist()}")
    print(f"  patience:        {args.patience}")

    if args.inspect_shapes:
        inspect_first_batch(
            model=model,
            loader=train_loader,
            device=device,
            standardize=standardize,
        )

    best_state_dict = None
    best_epoch = 0
    best_val_bacc = -1.0
    best_val_loss = float("inf")
    patience_counter = 0

    history_rows = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            n_classes=n_classes,
            standardize=standardize,
            grad_clip=args.grad_clip,
        )

        val_metrics, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            n_classes=n_classes,
            standardize=standardize,
        )

        history_rows.append(
            {
                "config": config_name,
                "fold": fold_idx + 1,
                "epoch": epoch,
                "agg": agg,
                "lr": lr,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "train_balanced_acc": train_metrics["balanced_acc"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_balanced_acc": val_metrics["balanced_acc"],
                "val_macro_f1": val_metrics["macro_f1"],
                "train_pred_counts": json.dumps(train_metrics["pred_counts"]),
                "val_pred_counts": json.dumps(val_metrics["pred_counts"]),
            }
        )

        print(f"\nEpoch {epoch:03d}/{args.epochs}")
        print_metrics("train", train_metrics)
        print_metrics("val", val_metrics)

        improved = (
            val_metrics["balanced_acc"] > best_val_bacc
            or (
                val_metrics["balanced_acc"] == best_val_bacc
                and val_metrics["loss"] < best_val_loss
            )
        )

        if improved:
            best_val_bacc = val_metrics["balanced_acc"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch

            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            patience_counter = 0

            print(
                f"  new best | "
                f"epoch={best_epoch} | "
                f"val_bacc={best_val_bacc:.4f} | "
                f"val_loss={best_val_loss:.4f}"
            )
        else:
            patience_counter += 1

        if args.patience is not None and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    if best_state_dict is None:
        raise RuntimeError("No best model state was stored.")

    model.load_state_dict(best_state_dict)

    train_best_metrics, train_pred_rows = evaluate(
        model=model,
        loader=train_loader,
        criterion=criterion,
        device=device,
        n_classes=n_classes,
        standardize=standardize,
        collect_predictions=True,
        split_name="train",
        fold_id=fold_idx + 1,
        config_name=config_name,
    )

    val_best_metrics, val_pred_rows = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        n_classes=n_classes,
        standardize=standardize,
        collect_predictions=True,
        split_name="val",
        fold_id=fold_idx + 1,
        config_name=config_name,
    )

    test_metrics, test_pred_rows = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        n_classes=n_classes,
        standardize=standardize,
        collect_predictions=True,
        split_name="test",
        fold_id=fold_idx + 1,
        config_name=config_name,
    )

    print("\nBest model evaluation")
    print(f"  best epoch: {best_epoch}")
    print_metrics("train", train_best_metrics)
    print_metrics("val", val_best_metrics)
    print_metrics("test", test_metrics)

    print("\nTest confusion matrix")
    print(np.array(test_metrics["confusion_matrix"]))

    print("\nTest predictions")
    test_pred_df = pd.DataFrame(test_pred_rows)
    print(
        test_pred_df[
            ["subject_id", "y_true", "y_pred", "correct", "prob_0", "prob_1", "prob_2"]
        ].to_string(index=False)
    )

    run_dir = Path(args.save_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history_rows)
    predictions_df = pd.DataFrame(train_pred_rows + val_pred_rows + test_pred_rows)

    history_path = run_dir / "history.csv"
    predictions_path = run_dir / "predictions.csv"
    checkpoint_path = run_dir / "best_model.pt"

    history_df.to_csv(history_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    torch.save(
        {
            "model_state_dict": best_state_dict,
            "config": config,
            "args": vars(args),
            "fold": fold_idx + 1,
            "best_epoch": best_epoch,
            "best_val_bacc": best_val_bacc,
            "best_val_loss": best_val_loss,
            "n_classes": n_classes,
        },
        checkpoint_path,
    )

    summary_row = {
        "config": config_name,
        "fold": fold_idx + 1,
        "agg": agg,
        "lr": lr,
        "best_epoch": best_epoch,
        "best_val_bacc": best_val_bacc,
        "best_val_loss": best_val_loss,
        "train_bacc": train_best_metrics["balanced_acc"],
        "train_acc": train_best_metrics["acc"],
        "train_macro_f1": train_best_metrics["macro_f1"],
        "val_bacc": val_best_metrics["balanced_acc"],
        "val_acc": val_best_metrics["acc"],
        "val_macro_f1": val_best_metrics["macro_f1"],
        "test_bacc": test_metrics["balanced_acc"],
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_loss": test_metrics["loss"],
        "train_pred_counts": json.dumps(train_best_metrics["pred_counts"]),
        "val_pred_counts": json.dumps(val_best_metrics["pred_counts"]),
        "test_pred_counts": json.dumps(test_metrics["pred_counts"]),
        "test_confusion_matrix": json.dumps(test_metrics["confusion_matrix"]),
        "history_path": str(history_path),
        "predictions_path": str(predictions_path),
        "checkpoint_path": str(checkpoint_path),
    }

    return summary_row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--condition", type=str, default="closed")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])

    parser.add_argument("--preset", type=str, default="diagnostic")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--n-channels", type=int, default=24)

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--inspect-shapes", action="store_true")

    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/eeg_ct/eegnet_experiments",
    )

    args = parser.parse_args()

    if args.preset not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown preset '{args.preset}'. "
            f"Available presets: {list(PRESET_CONFIGS.keys())}"
        )

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")

    dataset = EEGDataset(condition=args.condition)

    labels = dataset.get_labels()
    n_classes = int(max(labels)) + 1

    print("\nDataset")
    print(f"  condition: {args.condition}")
    print(f"  subjects: {len(dataset)}")
    print(f"  labels: {dict(Counter(labels))}")
    print(f"  n_classes: {n_classes}")

    folds = create_kfold_dataloaders(
        dataset,
        k=args.k,
        batch_size=args.batch_size,
        shuffle=True,
        random_state=args.seed,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    for fold_idx in args.folds:
        if fold_idx < 0 or fold_idx >= len(folds):
            raise ValueError(
                f"Fold index must be in [0, {len(folds) - 1}], got {fold_idx}"
            )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    configs = PRESET_CONFIGS[args.preset]

    all_summary_rows = []

    for config_idx, config in enumerate(configs):
        for fold_idx in args.folds:
            seed_for_run = args.seed + 1000 * config_idx + fold_idx
            set_seed(seed_for_run)

            train_loader, val_loader, test_loader = folds[fold_idx]

            summary_row = run_one_config_one_fold(
                dataset=dataset,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=config,
                args=args,
                device=device,
                fold_idx=fold_idx,
            )

            all_summary_rows.append(summary_row)

            summary_df = pd.DataFrame(all_summary_rows)
            summary_df.to_csv(save_dir / "summary_partial.csv", index=False)

    summary_df = pd.DataFrame(all_summary_rows)

    summary_path = save_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("Final summary")
    print("=" * 80)

    cols = [
        "config",
        "fold",
        "best_epoch",
        "val_bacc",
        "test_bacc",
        "test_acc",
        "test_macro_f1",
        "test_pred_counts",
    ]

    print(summary_df[cols].to_string(index=False))

    grouped = (
        summary_df
        .groupby("config")
        .agg(
            mean_test_bacc=("test_bacc", "mean"),
            std_test_bacc=("test_bacc", "std"),
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            mean_test_macro_f1=("test_macro_f1", "mean"),
            std_test_macro_f1=("test_macro_f1", "std"),
        )
        .reset_index()
    )

    grouped_path = save_dir / "summary_by_config.csv"
    grouped.to_csv(grouped_path, index=False)

    print("\nSummary by config")
    print(grouped.to_string(index=False))

    print("\nSaved files")
    print(f"  summary:           {summary_path}")
    print(f"  summary_by_config: {grouped_path}")


if __name__ == "__main__":
    main()