"""Single-split and k-fold training runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from milk10k_effb2_metadata.data import (
    fit_metadata_spec,
    kfold_splits,
    lesion_split,
    make_loaders,
    metadata_vector,
)
from milk10k_effb2_metadata.engine import train_phase
from milk10k_effb2_metadata.losses import build_loss
from milk10k_effb2_metadata.metrics import apply_class_bias, compute_metrics, optimize_class_bias, predict, save_predictions
from milk10k_effb2_metadata.model_setup import build_model, load_resume_checkpoint
from milk10k_effb2_metadata.training_utils import json_safe, save_kfold_summary, save_run_config


def build_tail_tracking_config(
    train_df: pd.DataFrame,
    class_names: list[str],
    label_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.loss != "ldam" or args.tail_num_classes <= 0:
        return None

    counts_series = train_df["label"].value_counts().reindex(class_names, fill_value=0)
    train_class_counts = {label: int(counts_series[label]) for label in class_names}
    tail_class_names = sorted(class_names, key=lambda label: (train_class_counts[label], label))[
        : min(args.tail_num_classes, len(class_names))
    ]
    return {
        "tail_class_names": tail_class_names,
        "tail_class_indices": [label_to_idx[label] for label in tail_class_names],
        "train_class_counts": train_class_counts,
    }


def run_training_split(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_names: list[str],
    label_to_idx: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    clinical_backbone_backend: str,
    dermoscopic_backbone_backend: str,
    output_dir: Path,
    fold: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = output_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)

    metadata_spec = fit_metadata_spec(train_df)
    metadata_dim = len(metadata_vector(train_df.iloc[0], metadata_spec))
    save_run_config(
        output_dir,
        args,
        class_names,
        metadata_spec,
        train_df,
        val_df,
        clinical_backbone_backend,
        dermoscopic_backbone_backend,
        fold,
    )

    model = build_model(
        class_names,
        metadata_dim,
        args,
        device,
        clinical_backbone_backend,
        dermoscopic_backbone_backend,
    )
    resume_epoch, resume_best_val_f1, resume_phase = load_resume_checkpoint(args.resume_checkpoint, model, device)
    train_loader, val_loader = make_loaders(train_df, val_df, label_to_idx, metadata_spec, args)
    criterion = build_loss(train_df, label_to_idx, args, device)
    tail_config = build_tail_tracking_config(train_df, class_names, label_to_idx, args)

    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Paired lesions: train={len(train_df)}, val={len(val_df)}, total={len(df)}")
    print(f"Metadata input dim: {metadata_dim}")
    print(f"MONET columns: {len(metadata_spec.get('monet_columns', []))}")
    print(
        f"Metadata mode: disable_metadata={args.disable_metadata}, "
        f"freeze_metadata_head={args.freeze_metadata_head}, metadata_lr={args.metadata_lr}, "
        f"metadata_fusion={args.metadata_fusion}, image_fusion={getattr(args, 'image_fusion', 'concat')}, "
        f"gate_hidden_dim={args.metadata_gate_hidden_dim}"
    )
    print(f"Loss: {args.loss}, class_weight={args.class_weight}, weighted_sampler={args.weighted_sampler}")
    if getattr(args, "image_fusion", "concat") == "moe" and args.logit_fusion_mode == "fixed":
        print("Note: --image-fusion moe already mixes expert logits; --logit-fusion-mode fixed adds extra branch logits.")
    if args.loss == "ce_f1":
        print(f"Soft-F1 class controls: ignore={args.f1_ignore_classes}, weights={args.f1_class_weight}")
    if args.loss == "ldam" and args.class_weight:
        print("Note: --class-weight is ignored for --loss ldam because LDAM+DRW uses effective-number alpha.")
    if tail_config is not None:
        tail_counts = {label: tail_config["train_class_counts"][label] for label in tail_config["tail_class_names"]}
        print(f"LDAM tail tracking: tail_num_classes={args.tail_num_classes}, tail_counts={tail_counts}")

    history: list[dict[str, Any]] = []
    history_path = output_dir / "history.csv"
    if args.resume_checkpoint is not None and history_path.exists():
        history = pd.read_csv(history_path).to_dict("records")
    best_start = resume_best_val_f1 if args.resume_checkpoint is not None else float("-inf")
    best_tail_start = float("-inf")
    tail_best_path = output_dir / "tail_best.pt"
    if args.resume_checkpoint is not None and tail_best_path.exists():
        tail_checkpoint = torch.load(tail_best_path, map_location=device, weights_only=False)
        best_tail_start = float(tail_checkpoint.get("best_val_tail_recall_macro", float("-inf")))
    skip_freeze_until = resume_epoch if resume_phase == "freeze" else 1
    if resume_phase == "finetune":
        skip_freeze_until = args.freeze_epochs + 1
    skip_finetune_until = resume_epoch if resume_phase == "finetune" else 1
    epoch, best_val_f1, best_val_tail_recall = train_phase(
        "freeze",
        args.freeze_epochs,
        1,
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        args,
        class_names,
        label_to_idx,
        metadata_spec,
        output_dir,
        history,
        best_start,
        skip_freeze_until,
        **(tail_config or {}),
        best_val_tail_recall=best_tail_start,
    )
    epoch, best_val_f1, best_val_tail_recall = train_phase(
        "finetune",
        args.finetune_epochs,
        epoch,
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        args,
        class_names,
        label_to_idx,
        metadata_spec,
        output_dir,
        history,
        best_val_f1,
        skip_finetune_until,
        **(tail_config or {}),
        best_val_tail_recall=best_val_tail_recall,
    )

    best_path = output_dir / "best.pt"
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
    y_true, y_prob = predict(model, val_loader, device)
    metrics, per_class_df, cm = compute_metrics(y_true, y_prob, class_names)
    metrics = {
        "best_selection_metric": float(best_val_f1),
        "selection_metric_name": args.selection_metric,
        "best_val_f1_macro": float(best_val_f1) if args.selection_metric == "f1_macro" else None,
        **metrics,
    }
    if tail_config is not None:
        metrics["best_val_tail_recall_macro"] = float(best_val_tail_recall)
        metrics["tail_class_names"] = tail_config["tail_class_names"]
    if args.calibrate_bias:
        class_bias, calibrated_score = optimize_class_bias(
            y_true,
            y_prob,
            class_names,
            metric_name=args.calibration_metric,
            max_bias=args.calibration_max_bias,
            step=args.calibration_step,
            passes=args.calibration_passes,
        )
        calibrated_prob = apply_class_bias(y_prob, class_bias)
        calibrated_metrics, calibrated_per_class_df, calibrated_cm = compute_metrics(y_true, calibrated_prob, class_names)
        calibration_payload = {
            "metric": args.calibration_metric,
            "optimized_score": float(calibrated_score),
            "class_names": class_names,
            "class_bias": [float(item) for item in class_bias.tolist()],
            "metrics": calibrated_metrics,
        }
        with open(output_dir / "calibration.json", "w", encoding="utf-8") as f:
            json.dump(json_safe(calibration_payload), f, indent=2)
        calibrated_per_class_df.to_csv(output_dir / "per_class_metrics_calibrated.csv", index=False)
        pd.DataFrame(calibrated_cm, index=class_names, columns=class_names).to_csv(
            output_dir / "confusion_matrix_calibrated.csv"
        )
        metrics["calibrated"] = calibrated_metrics
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(metrics), f, indent=2)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / "confusion_matrix.csv")
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)
    save_predictions(val_df, y_true, y_prob, class_names, output_dir)
    print(
        f"Done: best_val_f1_macro={best_val_f1:.4f}, "
        f"val_acc={metrics['accuracy']:.4f}, balanced_acc={metrics['balanced_accuracy']:.4f}, "
        f"f1_macro={metrics['f1_macro']:.4f}, top3={metrics['top3_accuracy']:.4f}, "
        f"auc_macro={metrics['roc_auc_macro_ovr']}"
    )
    return metrics


def train_single_run(
    df: pd.DataFrame,
    class_names: list[str],
    label_to_idx: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    clinical_backbone_backend: str,
    dermoscopic_backbone_backend: str,
) -> dict[str, Any]:
    if args.synthetic_train_only:
        synthetic_mask = df["lesion_id"].astype(str).str.contains("__sdpair_", regex=False)
        real_df = df[~synthetic_mask].copy()
        synthetic_df = df[synthetic_mask].copy()
        train_df, val_df = lesion_split(real_df, args.val_size, args.seed)
        train_df = pd.concat([train_df, synthetic_df], ignore_index=True, sort=False)
        print(
            f"Synthetic train-only split: real_train={len(train_df) - len(synthetic_df)}, "
            f"synthetic_train={len(synthetic_df)}, val_real={len(val_df)}"
        )
    else:
        train_df, val_df = lesion_split(df, args.val_size, args.seed)
    return run_training_split(
        df,
        train_df,
        val_df,
        class_names,
        label_to_idx,
        args,
        device,
        clinical_backbone_backend,
        dermoscopic_backbone_backend,
        args.output_dir,
    )


def train_kfold(
    df: pd.DataFrame,
    class_names: list[str],
    label_to_idx: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    clinical_backbone_backend: str,
    dermoscopic_backbone_backend: str,
) -> list[dict[str, Any]]:
    fold_metrics = []
    for fold_idx, (train_df, val_df) in enumerate(kfold_splits(df, args.k_folds, args.seed)):
        print(f"\nK-fold {fold_idx + 1}/{args.k_folds}")
        metrics = run_training_split(
            df,
            train_df,
            val_df,
            class_names,
            label_to_idx,
            args,
            device,
            clinical_backbone_backend,
            dermoscopic_backbone_backend,
            args.output_dir / f"fold_{fold_idx:02d}",
            fold_idx,
        )
        fold_metrics.append({"fold": fold_idx, **metrics})
    save_kfold_summary(fold_metrics, args.output_dir)
    return fold_metrics
