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
    confusion_matrix,
    f1_score,
    recall_score,
)

from src.data.build_eeg_ct import EEGDataset, create_kfold_dataloaders
from src.models.eegnet_baseline import EEGNetBaseline


FIXED_ARCH = {
    "name": "eegnet_small_t600",
    "F1": 8,
    "D": 2,
    "F2": 16,
    "temporal_kernel": 63,
    "separable_kernel": 15,
    "pool1": 8,
    "pool2": 8,
    "dropout": 0.5,
    "lr": 1e-4,
    "weight_decay": 1e-4,
}


POOLER_CONFIGS = {
    "meanmax050_ref": {
        "name": "meanmax050_ref",
        "agg": "meanmax",
        "meanmax_alpha": 0.5,
        "attn_hidden": 16,
    },
    "learned_meanmax": {
        "name": "learned_meanmax",
        "agg": "learned_meanmax",
        "meanmax_alpha": 0.5,
        "attn_hidden": 16,
    },
    "attn_logits": {
        "name": "attn_logits",
        "agg": "attn_logits",
        "meanmax_alpha": 0.5,
        "attn_hidden": 16,
    },
    "gated_attn_logits": {
        "name": "gated_attn_logits",
        "agg": "gated_attn_logits",
        "meanmax_alpha": 0.5,
        "attn_hidden": 16,
    },
    "class_attn_logits": {
        "name": "class_attn_logits",
        "agg": "class_attn_logits",
        "meanmax_alpha": 0.5,
        "attn_hidden": 16,
    },
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
    if mask is None:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return (x - mean) / std

    mask_f = mask.float().unsqueeze(1)
    denom = mask_f.sum(dim=-1, keepdim=True).clamp_min(1.0)

    mean = (x * mask_f).sum(dim=-1, keepdim=True) / denom

    var = (((x - mean) * mask_f) ** 2).sum(dim=-1, keepdim=True) / denom
    std = torch.sqrt(var + eps)

    x_norm = (x - mean) / std
    x_norm = x_norm.masked_fill(~mask.unsqueeze(1), 0.0)

    return x_norm


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
    n_classes: int,
    total_loss: float | None = None,
):
    labels = list(range(n_classes))
    n = max(len(targets), 1)

    metrics = {
        "acc": accuracy_score(targets, preds),
        "balanced_acc": recall_score(
            targets,
            preds,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
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
        "target_counts": dict(Counter(int(v) for v in targets)),
        "pred_counts": dict(Counter(int(v) for v in preds)),
    }

    if total_loss is not None:
        metrics["loss"] = total_loss / n

    return metrics


def compute_majority_baseline(
    train_labels,
    test_labels,
    n_classes: int,
):
    counts = Counter(int(v) for v in train_labels)

    majority_class = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0][0]

    preds = [majority_class for _ in test_labels]

    metrics = compute_metrics(
        targets=test_labels,
        preds=preds,
        n_classes=n_classes,
        total_loss=None,
    )

    metrics["majority_class"] = majority_class

    return metrics


def print_metrics(split_name: str, metrics: dict):
    loss_text = ""

    if "loss" in metrics:
        loss_text = f"loss={metrics['loss']:.4f} | "

    print(
        f"{split_name:>8} | "
        f"{loss_text}"
        f"acc={metrics['acc']:.4f} | "
        f"bacc={metrics['balanced_acc']:.4f} | "
        f"macro_f1={metrics['macro_f1']:.4f} | "
        f"pred_counts={metrics['pred_counts']}"
    )


def build_model(
    pooler_config: dict,
    args,
    n_classes: int,
    device: torch.device,
):
    model = EEGNetBaseline(
        n_channels=args.n_channels,
        n_classes=n_classes,
        F1=FIXED_ARCH["F1"],
        D=FIXED_ARCH["D"],
        F2=FIXED_ARCH["F2"],
        temporal_kernel=FIXED_ARCH["temporal_kernel"],
        separable_kernel=FIXED_ARCH["separable_kernel"],
        pool1=FIXED_ARCH["pool1"],
        pool2=FIXED_ARCH["pool2"],
        dropout=args.dropout,
        agg=pooler_config["agg"],
        meanmax_alpha=pooler_config["meanmax_alpha"],
        attn_hidden=args.attn_hidden,
    )

    return model.to(device)


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    n_classes: int,
    standardize: bool,
    grad_clip: float | None,
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
        n_classes=n_classes,
        total_loss=total_loss,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    n_classes: int,
    standardize: bool,
    collect_predictions: bool = False,
    split_name: str | None = None,
    split_seed: int | None = None,
    init_seed: int | None = None,
    fold_id: int | None = None,
    pooler_name: str | None = None,
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
        confidence = probs.max(dim=1).values
        preds = logits_subject.argmax(dim=1)

        batch_size = y.shape[0]
        total_loss += loss.item() * batch_size

        y_cpu = y.detach().cpu().tolist()
        pred_cpu = preds.detach().cpu().tolist()
        probs_cpu = probs.detach().cpu().numpy()
        confidence_cpu = confidence.detach().cpu().tolist()

        all_targets.extend(y_cpu)
        all_preds.extend(pred_cpu)

        if collect_predictions:
            for i, subject_id in enumerate(batch["subject_id"]):
                row = {
                    "pooler": pooler_name,
                    "split_seed": split_seed,
                    "init_seed": init_seed,
                    "fold": fold_id,
                    "split": split_name,
                    "subject_id": int(subject_id),
                    "y_true": int(y_cpu[i]),
                    "y_pred": int(pred_cpu[i]),
                    "correct": int(y_cpu[i] == pred_cpu[i]),
                    "confidence": float(confidence_cpu[i]),
                    "length": int(batch["lengths"][i].item()),
                }

                for class_idx in range(n_classes):
                    row[f"prob_{class_idx}"] = float(probs_cpu[i, class_idx])

                prediction_rows.append(row)

    metrics = compute_metrics(
        targets=all_targets,
        preds=all_preds,
        n_classes=n_classes,
        total_loss=total_loss,
    )

    return metrics, prediction_rows


def inspect_first_batch(
    model: nn.Module,
    loader,
    device: torch.device,
    standardize: bool,
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


def run_one_training(
    dataset: EEGDataset,
    train_loader,
    val_loader,
    test_loader,
    args,
    pooler_config: dict,
    device: torch.device,
    n_classes: int,
    split_seed: int,
    init_seed: int,
    fold_idx: int,
):
    set_seed(init_seed)

    fold_id = fold_idx + 1
    pooler_name = pooler_config["name"]

    run_name = (
        f"{pooler_name}"
        f"_splitseed_{split_seed}"
        f"_initseed_{init_seed}"
        f"_fold_{fold_id}"
    )

    print("\n" + "=" * 96)
    print(f"Run: {run_name}")
    print("=" * 96)

    train_subjects, train_labels = get_split_subjects_and_labels(
        dataset,
        train_loader,
    )

    val_subjects, val_labels = get_split_subjects_and_labels(
        dataset,
        val_loader,
    )

    test_subjects, test_labels = get_split_subjects_and_labels(
        dataset,
        test_loader,
    )

    print("\nSplit information")
    print(f"  pooler:           {pooler_name}")
    print(f"  split_seed:       {split_seed}")
    print(f"  init_seed:        {init_seed}")
    print(f"  fold:             {fold_id}")
    print(f"  train subjects:   {len(train_subjects)} | labels: {dict(Counter(train_labels))}")
    print(f"  val subjects:     {len(val_subjects)} | labels: {dict(Counter(val_labels))}")
    print(f"  test subjects:    {len(test_subjects)} | labels: {dict(Counter(test_labels))}")
    print(f"  train batches:    {len(train_loader)}")
    print(f"  val batches:      {len(val_loader)}")
    print(f"  test batches:     {len(test_loader)}")

    standardize = not args.no_standardize

    model = build_model(
        pooler_config=pooler_config,
        args=args,
        n_classes=n_classes,
        device=device,
    )

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
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    majority_metrics = compute_majority_baseline(
        train_labels=train_labels,
        test_labels=test_labels,
        n_classes=n_classes,
    )

    chance_bacc = 1.0 / n_classes

    print("\nModel configuration")
    print(f"  architecture:     {FIXED_ARCH['name']}")
    print(f"  F1:               {FIXED_ARCH['F1']}")
    print(f"  D:                {FIXED_ARCH['D']}")
    print(f"  F2:               {FIXED_ARCH['F2']}")
    print(f"  pool1:            {FIXED_ARCH['pool1']}")
    print(f"  pool2:            {FIXED_ARCH['pool2']}")
    print(f"  total_pool:       {FIXED_ARCH['pool1'] * FIXED_ARCH['pool2']}")
    print(f"  expected T':      approx 38400 / {FIXED_ARCH['pool1'] * FIXED_ARCH['pool2']}")
    print(f"  aggregation:      {pooler_config['agg']}")
    print(f"  attn_hidden:      {args.attn_hidden}")

    print("\nTraining configuration")
    print(f"  epochs:           {args.epochs}")
    print(f"  patience:         {args.patience}")
    print(f"  lr:               {args.lr}")
    print(f"  weight_decay:     {args.weight_decay}")
    print(f"  dropout:          {args.dropout}")
    print(f"  standardize:      {standardize}")
    print(f"  grad_clip:        {args.grad_clip}")
    print(
        f"  class_weights:    "
        f"{None if class_weights is None else class_weights.detach().cpu().tolist()}"
    )

    print("\nReference baselines on this test split")
    print(f"  chance bacc:      {chance_bacc:.4f}")
    print(
        f"  majority class:   {majority_metrics['majority_class']} | "
        f"acc={majority_metrics['acc']:.4f} | "
        f"bacc={majority_metrics['balanced_acc']:.4f} | "
        f"macro_f1={majority_metrics['macro_f1']:.4f}"
    )

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
                "pooler": pooler_name,
                "split_seed": split_seed,
                "init_seed": init_seed,
                "fold": fold_id,
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "train_balanced_acc": train_metrics["balanced_acc"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_pred_counts": json.dumps(train_metrics["pred_counts"]),
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_balanced_acc": val_metrics["balanced_acc"],
                "val_macro_f1": val_metrics["macro_f1"],
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
        split_seed=split_seed,
        init_seed=init_seed,
        fold_id=fold_id,
        pooler_name=pooler_name,
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
        split_seed=split_seed,
        init_seed=init_seed,
        fold_id=fold_id,
        pooler_name=pooler_name,
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
        split_seed=split_seed,
        init_seed=init_seed,
        fold_id=fold_id,
        pooler_name=pooler_name,
    )

    print("\nBest model evaluation")
    print(f"  best epoch: {best_epoch}")
    print_metrics("train", train_best_metrics)
    print_metrics("val", val_best_metrics)
    print_metrics("test", test_metrics)

    print("\nTest confusion matrix")
    print(np.array(test_metrics["confusion_matrix"]))

    run_dir = (
        Path(args.save_dir)
        / pooler_name
        / f"splitseed_{split_seed}"
        / f"initseed_{init_seed}"
        / f"fold_{fold_id}"
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history_rows)
    predictions_df = pd.DataFrame(
        train_pred_rows + val_pred_rows + test_pred_rows
    )

    history_path = run_dir / "history.csv"
    predictions_path = run_dir / "predictions.csv"
    checkpoint_path = run_dir / "best_model.pt"

    history_df.to_csv(history_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    if args.save_checkpoints:
        torch.save(
            {
                "model_state_dict": best_state_dict,
                "arch": copy.deepcopy(FIXED_ARCH),
                "pooler_config": copy.deepcopy(pooler_config),
                "args": vars(args),
                "split_seed": split_seed,
                "init_seed": init_seed,
                "fold": fold_id,
                "best_epoch": best_epoch,
                "best_val_bacc": best_val_bacc,
                "best_val_loss": best_val_loss,
                "n_classes": n_classes,
            },
            checkpoint_path,
        )
    else:
        checkpoint_path = ""

    test_bacc = test_metrics["balanced_acc"]
    test_acc = test_metrics["acc"]
    test_macro_f1 = test_metrics["macro_f1"]

    summary_row = {
        "pooler": pooler_name,
        "agg": pooler_config["agg"],
        "split_seed": split_seed,
        "init_seed": init_seed,
        "fold": fold_id,
        "best_epoch": best_epoch,
        "best_val_bacc": best_val_bacc,
        "best_val_loss": best_val_loss,
        "F1": FIXED_ARCH["F1"],
        "D": FIXED_ARCH["D"],
        "F2": FIXED_ARCH["F2"],
        "pool1": FIXED_ARCH["pool1"],
        "pool2": FIXED_ARCH["pool2"],
        "total_pool": FIXED_ARCH["pool1"] * FIXED_ARCH["pool2"],
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "attn_hidden": args.attn_hidden,
        "chance_bacc": chance_bacc,
        "majority_class": majority_metrics["majority_class"],
        "majority_test_bacc": majority_metrics["balanced_acc"],
        "majority_test_acc": majority_metrics["acc"],
        "majority_test_macro_f1": majority_metrics["macro_f1"],
        "train_bacc": train_best_metrics["balanced_acc"],
        "train_acc": train_best_metrics["acc"],
        "train_macro_f1": train_best_metrics["macro_f1"],
        "val_bacc": val_best_metrics["balanced_acc"],
        "val_acc": val_best_metrics["acc"],
        "val_macro_f1": val_best_metrics["macro_f1"],
        "test_bacc": test_bacc,
        "test_acc": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_loss": test_metrics["loss"],
        "test_bacc_minus_chance": test_bacc - chance_bacc,
        "test_bacc_minus_majority": test_bacc - majority_metrics["balanced_acc"],
        "test_gt_chance": int(test_bacc > chance_bacc),
        "test_gt_majority_bacc": int(
            test_bacc > majority_metrics["balanced_acc"]
        ),
        "train_pred_counts": json.dumps(train_best_metrics["pred_counts"]),
        "val_pred_counts": json.dumps(val_best_metrics["pred_counts"]),
        "test_pred_counts": json.dumps(test_metrics["pred_counts"]),
        "test_target_counts": json.dumps(test_metrics["target_counts"]),
        "test_confusion_matrix": json.dumps(test_metrics["confusion_matrix"]),
        "history_path": str(history_path),
        "predictions_path": str(predictions_path),
        "checkpoint_path": str(checkpoint_path),
    }

    return summary_row


def summarize_results(
    summary_df: pd.DataFrame,
    save_dir: Path,
):
    summary_path = save_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    grouped_pooler = (
        summary_df
        .groupby("pooler")
        .agg(
            n_runs=("test_bacc", "count"),
            mean_test_bacc=("test_bacc", "mean"),
            std_test_bacc=("test_bacc", "std"),
            median_test_bacc=("test_bacc", "median"),
            min_test_bacc=("test_bacc", "min"),
            max_test_bacc=("test_bacc", "max"),
            mean_test_acc=("test_acc", "mean"),
            std_test_acc=("test_acc", "std"),
            mean_test_macro_f1=("test_macro_f1", "mean"),
            std_test_macro_f1=("test_macro_f1", "std"),
            mean_val_bacc=("val_bacc", "mean"),
            std_val_bacc=("val_bacc", "std"),
            mean_best_epoch=("best_epoch", "mean"),
            frac_test_bacc_gt_chance=("test_gt_chance", "mean"),
            frac_test_bacc_gt_majority=("test_gt_majority_bacc", "mean"),
        )
        .reset_index()
        .sort_values("mean_test_bacc", ascending=False)
    )

    grouped_pooler_path = save_dir / "summary_by_pooler.csv"
    grouped_pooler.to_csv(grouped_pooler_path, index=False)

    grouped_pooler_seed = (
        summary_df
        .groupby(["pooler", "split_seed"])
        .agg(
            n_runs=("test_bacc", "count"),
            mean_test_bacc=("test_bacc", "mean"),
            std_test_bacc=("test_bacc", "std"),
            mean_test_acc=("test_acc", "mean"),
            mean_test_macro_f1=("test_macro_f1", "mean"),
            frac_test_bacc_gt_chance=("test_gt_chance", "mean"),
        )
        .reset_index()
    )

    grouped_pooler_seed_path = save_dir / "summary_by_pooler_split_seed.csv"
    grouped_pooler_seed.to_csv(grouped_pooler_seed_path, index=False)

    grouped_pooler_init = (
        summary_df
        .groupby(["pooler", "init_seed"])
        .agg(
            n_runs=("test_bacc", "count"),
            mean_test_bacc=("test_bacc", "mean"),
            std_test_bacc=("test_bacc", "std"),
            mean_test_acc=("test_acc", "mean"),
            mean_test_macro_f1=("test_macro_f1", "mean"),
            frac_test_bacc_gt_chance=("test_gt_chance", "mean"),
        )
        .reset_index()
    )

    grouped_pooler_init_path = save_dir / "summary_by_pooler_init_seed.csv"
    grouped_pooler_init.to_csv(grouped_pooler_init_path, index=False)

    print("\n" + "=" * 96)
    print("Final summary")
    print("=" * 96)

    cols = [
        "pooler",
        "split_seed",
        "init_seed",
        "fold",
        "best_epoch",
        "val_bacc",
        "test_bacc",
        "test_acc",
        "test_macro_f1",
        "test_pred_counts",
    ]

    print(summary_df[cols].to_string(index=False))

    print("\nSummary by pooler")
    print(grouped_pooler.to_string(index=False))

    print("\nSaved files")
    print(f"  summary:                       {summary_path}")
    print(f"  summary_by_pooler:             {grouped_pooler_path}")
    print(f"  summary_by_pooler_split_seed:  {grouped_pooler_seed_path}")
    print(f"  summary_by_pooler_init_seed:   {grouped_pooler_init_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--condition", type=str, default="closed")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])

    parser.add_argument(
        "--poolers",
        type=str,
        nargs="+",
        default=[
            "meanmax050_ref",
            "learned_meanmax",
            "attn_logits",
            "gated_attn_logits",
            "class_attn_logits",
        ],
        choices=sorted(POOLER_CONFIGS.keys()),
    )

    parser.add_argument(
        "--split-seeds",
        type=int,
        nargs="+",
        default=[3407, 1234, 2025],
    )

    parser.add_argument(
        "--init-seeds",
        type=int,
        nargs="+",
        default=[3407, 1234],
    )

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--lr", type=float, default=FIXED_ARCH["lr"])
    parser.add_argument("--weight-decay", type=float, default=FIXED_ARCH["weight_decay"])
    parser.add_argument("--dropout", type=float, default=FIXED_ARCH["dropout"])

    parser.add_argument("--attn-hidden", type=int, default=16)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--n-channels", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--inspect-shapes", action="store_true")
    parser.add_argument("--save-checkpoints", action="store_true")

    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs/eeg_ct/eegnet_trainable_pooling",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")

    dataset = EEGDataset(condition=args.condition)

    labels = dataset.get_labels()
    n_classes = int(max(labels)) + 1

    print("\nDataset")
    print(f"  condition: {args.condition}")
    print(f"  subjects:  {len(dataset)}")
    print(f"  labels:    {dict(Counter(labels))}")
    print(f"  n_classes: {n_classes}")

    print("\nFixed architecture")
    print(f"  name:       {FIXED_ARCH['name']}")
    print(f"  F1:         {FIXED_ARCH['F1']}")
    print(f"  D:          {FIXED_ARCH['D']}")
    print(f"  F2:         {FIXED_ARCH['F2']}")
    print(f"  pool1:      {FIXED_ARCH['pool1']}")
    print(f"  pool2:      {FIXED_ARCH['pool2']}")
    print(f"  total_pool: {FIXED_ARCH['pool1'] * FIXED_ARCH['pool2']}")
    print(f"  T':         approx 600 for T=38400")

    print("\nTrainable pooling experiment")
    print(f"  poolers:      {args.poolers}")
    print(f"  split_seeds:  {args.split_seeds}")
    print(f"  init_seeds:   {args.init_seeds}")
    print(f"  folds:        {args.folds}")
    print(
        f"  total runs:   "
        f"{len(args.poolers) * len(args.split_seeds) * len(args.init_seeds) * len(args.folds)}"
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []

    for split_seed in args.split_seeds:
        print("\n" + "#" * 96)
        print(f"Creating folds with split_seed={split_seed}")
        print("#" * 96)

        set_seed(split_seed)

        folds = create_kfold_dataloaders(
            dataset,
            k=args.k,
            batch_size=args.batch_size,
            shuffle=True,
            random_state=split_seed,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        for fold_idx in args.folds:
            if fold_idx < 0 or fold_idx >= len(folds):
                raise ValueError(
                    f"Fold index must be in [0, {len(folds) - 1}], got {fold_idx}"
                )

        for pooler_name in args.poolers:
            pooler_config = POOLER_CONFIGS[pooler_name]

            for fold_idx in args.folds:
                train_loader, val_loader, test_loader = folds[fold_idx]

                for init_seed in args.init_seeds:
                    summary_row = run_one_training(
                        dataset=dataset,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        args=args,
                        pooler_config=pooler_config,
                        device=device,
                        n_classes=n_classes,
                        split_seed=split_seed,
                        init_seed=init_seed,
                        fold_idx=fold_idx,
                    )

                    all_summary_rows.append(summary_row)

                    partial_df = pd.DataFrame(all_summary_rows)
                    partial_df.to_csv(save_dir / "summary_partial.csv", index=False)

    summary_df = pd.DataFrame(all_summary_rows)

    summarize_results(
        summary_df=summary_df,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()