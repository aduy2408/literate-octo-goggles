#!/usr/bin/env python3
"""
Train PyTorch/timm transfer-learning baselines on MILK10k.

Expected files in --data-dir:
  MILK10k_Training_Input/<lesion_id>/<isic_id>.jpg
  MILK10k_Training_GroundTruth.csv
  MILK10k_Training_Metadata.csv

Examples:
  python train_milk10k_baselines.py --models resnet50,densenet121 --epochs 15
  python train_milk10k_baselines.py --models efficientnet_family,densenet_family
  python train_milk10k_baselines.py --models all --image-type dermoscopic
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from timm.data import create_transform, resolve_data_config
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import (
    Milk10kDataset,
    PairedMilk10kDataset,
    lesion_level_train_val_split,
    load_dataframe,
    resolve_data_dir,
    set_seed,
    to_paired_lesion_dataframe,
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    timm_name: str
    image_size: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "efficientnet_b0": ModelSpec("efficientnet_b0", "efficientnet_b0", 224),
    "efficientnet_b1": ModelSpec("efficientnet_b1", "efficientnet_b1", 240),
    "efficientnet_b2": ModelSpec("efficientnet_b2", "efficientnet_b2", 260),
    "efficientnet_b3": ModelSpec("efficientnet_b3", "efficientnet_b3", 300),
    "densenet121": ModelSpec("densenet121", "densenet121", 224),
    "densenet169": ModelSpec("densenet169", "densenet169", 224),
    "densenet201": ModelSpec("densenet201", "densenet201", 224),
    "convnext_tiny": ModelSpec("convnext_tiny", "convnext_tiny", 224),
    "convnext_small": ModelSpec("convnext_small", "convnext_small", 224),
    "convnext_base": ModelSpec("convnext_base", "convnext_base", 224),
    "mobilenetv2": ModelSpec("mobilenetv2", "mobilenetv2_100", 224),
    "resnet50": ModelSpec("resnet50", "resnet50", 224),
    "vgg16": ModelSpec("vgg16", "vgg16", 224),
    "xception": ModelSpec("xception", "legacy_xception", 299),
    "inceptionv3": ModelSpec("inceptionv3", "inception_v3", 299),
    "inceptionresnetv2": ModelSpec("inceptionresnetv2", "inception_resnet_v2", 299),
}

MODEL_ALIASES: dict[str, list[str]] = {
    "efficientnet_family": [
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
    ],
    "densenet_family": ["densenet121", "densenet169", "densenet201"],
    "convnext_family": ["convnext_tiny", "convnext_small", "convnext_base"],
    "all": [
        "efficientnet_b0",
        "efficientnet_b1",
        "efficientnet_b2",
        "efficientnet_b3",
        "densenet121",
        "densenet169",
        "densenet201",
        "convnext_tiny",
        "convnext_small",
        "convnext_base",
        "mobilenetv2",
        "resnet50",
        "vgg16",
        "xception",
        "inceptionv3",
        "inceptionresnetv2",
    ],
}

HEAD_PARAM_MARKERS = ("classifier", "fc", "head", "last_linear")
AUTO_BATCH_SIZES = {
    "mobilenetv2": 16,
    "resnet50": 16,
    "vgg16": 8,
    "densenet121": 16,
    "densenet169": 16,
    "densenet201": 8,
    "efficientnet_b0": 16,
    "efficientnet_b1": 16,
    "efficientnet_b2": 16,
    "efficientnet_b3": 8,
    "convnext_tiny": 16,
    "convnext_small": 8,
    "convnext_base": 4,
    "xception": 8,
    "inceptionv3": 8,
    "inceptionresnetv2": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MILK10k PyTorch/timm image-classification baselines.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Folder containing MILK10k CSVs and MILK10k_Training_Input. Auto-detected if omitted.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("baseline_runs_torch"), help="Where outputs are saved.")
    parser.add_argument(
        "--models",
        default="resnet50",
        help="Comma-separated model names or aliases: all, efficientnet_family, densenet_family, convnext_family.",
    )
    parser.add_argument(
        "--image-type",
        choices=["all", "dermoscopic", "clinical_close_up"],
        default="all",
        help="Use all images or only one MILK10k acquisition type.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Batch size integer, or 'auto' for per-model T4-friendly defaults.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.20, help="Validation split ratio. Default is 0.20 for 8:2 train/val.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None, help="Override model default input size.")
    parser.add_argument("--no-pretrained", action="store_true", help="Train from random init instead of ImageNet weights.")
    parser.add_argument("--class-weight", action="store_true", help="Use balanced class weights.")
    parser.add_argument("--train-mode", choices=["head", "full"], default="head", help="Train classifier head only or all layers.")
    parser.add_argument("--fine-tune-epochs", type=int, default=0, help="Extra epochs training full model after head training.")
    parser.add_argument("--patience", type=int, default=5, help="Early-stopping patience on val_loss.")
    parser.add_argument(
        "--fine-tune-patience",
        type=int,
        default=None,
        help="Early-stopping patience for fine-tuning. Defaults to --patience.",
    )
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--resume", action="store_true", help="Resume from best .pt checkpoint if present.")
    return parser.parse_args()


def expand_models(raw: str) -> list[str]:
    expanded: list[str] = []
    for item in [x.strip().lower() for x in raw.split(",") if x.strip()]:
        for name in MODEL_ALIASES.get(item, [item]):
            if name not in MODEL_SPECS:
                valid = sorted(set(MODEL_SPECS) | set(MODEL_ALIASES))
                raise ValueError(f"Unknown model '{name}'. Valid names/aliases: {', '.join(valid)}")
            if name not in expanded:
                expanded.append(name)
    return expanded


def initial_batch_size(model_name: str, args: argparse.Namespace) -> int:
    if str(args.batch_size).lower() == "auto":
        batch_size = AUTO_BATCH_SIZES.get(model_name, 8)
        if args.image_type == "all":
            batch_size = max(1, batch_size // 2)
        return batch_size
    batch_size = int(args.batch_size)
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    return batch_size


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or ("cuda" in text and "out of memory" in text)


def clear_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_model(spec: ModelSpec, num_classes: int, pretrained: bool) -> nn.Module:
    return timm.create_model(spec.timm_name, pretrained=pretrained, num_classes=num_classes)


def configure_trainable(model: nn.Module, train_mode: str) -> None:
    if train_mode == "full":
        for param in model.parameters():
            param.requires_grad = True
        return

    for param in model.parameters():
        param.requires_grad = False

    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if isinstance(classifier, nn.Module):
        for param in classifier.parameters():
            param.requires_grad = True

    for name, param in model.named_parameters():
        if any(marker in name for marker in HEAD_PARAM_MARKERS):
            param.requires_grad = True

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if trainable == 0:
        raise RuntimeError("No trainable parameters found for head training. Retry with --train-mode full.")


def make_transforms(model: nn.Module, image_size: int):
    data_config = resolve_data_config({}, model=model)
    data_config["input_size"] = (3, image_size, image_size)
    train_transform = create_transform(**data_config, is_training=True)
    eval_transform = create_transform(**data_config, is_training=False)
    return train_transform, eval_transform


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_to_idx: dict[str, int],
    model: nn.Module,
    image_size: int,
    batch_size: int,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    train_transform, eval_transform = make_transforms(model, image_size)
    dataset_cls = PairedMilk10kDataset if {"clinical_path", "dermoscopic_path"}.issubset(train_df.columns) else Milk10kDataset
    train_ds = dataset_cls(train_df, label_to_idx, train_transform)
    val_ds = dataset_cls(val_df, label_to_idx, eval_transform)

    common = dict(batch_size=batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


def build_loss(train_df: pd.DataFrame, label_to_idx: dict[str, int], args: argparse.Namespace, device: torch.device) -> nn.Module:
    if not args.class_weight:
        return nn.CrossEntropyLoss()
    y = np.array([label_to_idx[label] for label in train_df["label"]])
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(label_to_idx)), y=y)
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def forward_batch(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 5:
        batch_size, num_views, channels, height, width = images.shape
        logits = model(images.reshape(batch_size * num_views, channels, height, width))
        return logits.view(batch_size, num_views, -1).mean(dim=1)
    return model(images)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    top3_correct = 0
    total = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast("cuda", enabled=use_amp):
                logits = forward_batch(model, images)
                loss = criterion(logits, labels)

            if training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        topk = min(3, logits.size(1))
        top3_correct += logits.topk(topk, dim=1).indices.eq(labels[:, None]).any(dim=1).sum().item()
        total += batch_size

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "top3_accuracy": top3_correct / max(total, 1),
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all = []
    probs_all = []
    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        logits = forward_batch(model, images)
        probs = torch.softmax(logits, dim=1)
        labels_all.append(labels.numpy())
        probs_all.append(probs.cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(probs_all)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    class_names: list[str],
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "class_names": class_names,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None, device: torch.device) -> tuple[int, float]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return int(checkpoint.get("epoch", 0)) + 1, float(checkpoint.get("best_val_loss", float("inf")))


def safe_roc_auc(y_true_bin: np.ndarray, y_prob: np.ndarray, average: str | None) -> float | None:
    try:
        return float(roc_auc_score(y_true_bin, y_prob, average=average, multi_class="ovr"))
    except ValueError:
        return None


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> tuple[dict, pd.DataFrame, np.ndarray]:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    y_true_bin = label_binarize(y_true, classes=labels)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    total = cm.sum()
    per_class_rows = []
    for idx, class_name in enumerate(class_names):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(total - tp - fn - fp)
        support = int(support_per_class[idx])

        class_accuracy = tp / support if support else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        try:
            auc_ovr = float(roc_auc_score(y_true_bin[:, idx], y_prob[:, idx]))
        except ValueError:
            auc_ovr = None

        per_class_rows.append(
            {
                "class": class_name,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "accuracy": class_accuracy,
                "precision": float(precision_per_class[idx]),
                "recall_sensitivity": float(recall_per_class[idx]),
                "specificity": specificity,
                "f1": float(f1_per_class[idx]),
                "auc_ovr": auc_ovr,
            }
        )

    top2_idx = np.argsort(y_prob, axis=1)[:, -min(2, len(class_names)) :]
    top3_idx = np.argsort(y_prob, axis=1)[:, -min(3, len(class_names)) :]
    top2_accuracy = float(np.mean((top2_idx == y_true[:, None]).any(axis=1)))
    top3_accuracy = float(np.mean((top3_idx == y_true[:, None]).any(axis=1)))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "top2_accuracy": top2_accuracy,
        "top3_accuracy": top3_accuracy,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "roc_auc_macro_ovr": safe_roc_auc(y_true_bin, y_prob, "macro"),
        "roc_auc_weighted_ovr": safe_roc_auc(y_true_bin, y_prob, "weighted"),
        "roc_auc_micro_ovr": safe_roc_auc(y_true_bin, y_prob, "micro"),
        "specificity_macro": float(np.mean([row["specificity"] for row in per_class_rows])),
        "per_class": per_class_rows,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "class_names": class_names,
    }

    return metrics, pd.DataFrame(per_class_rows), cm


def evaluate_and_save(
    model: nn.Module,
    eval_loader: DataLoader,
    eval_df: pd.DataFrame,
    class_names: list[str],
    run_dir: Path,
    model_name: str,
    device: torch.device,
    criterion: nn.Module,
    split_name: str = "val",
) -> dict:
    eval_stats = run_epoch(model, eval_loader, criterion, device)
    y_true, y_prob = predict(model, eval_loader, device)
    y_pred = y_prob.argmax(axis=1)

    cls_metrics, per_class_df, cm = compute_classification_metrics(y_true, y_pred, y_prob, class_names)
    metrics = {
        f"{split_name}_loss": float(eval_stats["loss"]),
        f"{split_name}_accuracy": cls_metrics["accuracy"],
        f"{split_name}_balanced_accuracy": cls_metrics["balanced_accuracy"],
        f"{split_name}_top2_accuracy": cls_metrics["top2_accuracy"],
        f"{split_name}_top3_accuracy": cls_metrics["top3_accuracy"],
        **cls_metrics,
    }

    with open(run_dir / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(run_dir / f"{model_name}_confusion_matrix.csv")
    per_class_df.to_csv(run_dir / f"{model_name}_per_class_metrics.csv", index=False)
    prediction_df = pd.DataFrame(
        {
            "lesion_id": eval_df["lesion_id"].tolist() if "lesion_id" in eval_df.columns else list(range(len(eval_df))),
            "y_true": y_true,
            "y_pred": y_pred,
            "label_true": [class_names[i] for i in y_true],
            "label_pred": [class_names[i] for i in y_pred],
            "confidence": y_prob.max(axis=1),
        }
    )
    for column in ["path", "clinical_path", "dermoscopic_path"]:
        if column in eval_df.columns:
            prediction_df[column] = eval_df[column].tolist()
    probability_df = pd.DataFrame(y_prob, columns=[f"prob_{name}" for name in class_names])
    pd.concat([prediction_df, probability_df], axis=1).to_csv(
        run_dir / f"{model_name}_{split_name}_predictions.csv",
        index=False,
    )
    probability_df.to_csv(run_dir / f"{model_name}_{split_name}_probabilities.csv", index=False)
    return metrics


def train_one_model(
    model_name: str,
    df_splits: tuple[pd.DataFrame, pd.DataFrame],
    class_names: list[str],
    label_to_idx: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    batch_size: int,
) -> dict:
    spec = MODEL_SPECS[model_name]
    image_size = args.image_size or spec.image_size
    run_dir = args.output_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / f"{model_name}_best.pt"

    print(f"\n[{model_name}] timm={spec.timm_name}, image_size={image_size}, batch_size={batch_size}, device={device}")
    model = build_model(spec, len(class_names), pretrained=not args.no_pretrained).to(device)
    configure_trainable(model, args.train_mode)

    train_df, val_df = df_splits
    train_loader, val_loader = make_loaders(
        train_df,
        val_df,
        label_to_idx,
        model,
        image_size,
        batch_size,
        args,
    )
    criterion = build_loss(train_df, label_to_idx, args, device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume and checkpoint_path.exists():
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, device)
        print(f"[{model_name}] resumed from {checkpoint_path} at epoch {start_epoch}")

    history = []
    patience_count = 0
    use_amp = args.amp and device.type == "cuda"

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        val_stats = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_stats["loss"])

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / f"{model_name}_history.csv", index=False)

        print(
            f"[{model_name}] epoch {epoch:03d}: "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
            f"val_acc={val_stats['accuracy']:.4f}"
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            patience_count = 0
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_loss, class_names, args)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"[{model_name}] early stopping at epoch {epoch}")
                break

    if args.fine_tune_epochs > 0:
        if checkpoint_path.exists():
            load_checkpoint(checkpoint_path, model, None, device)
        print(f"[{model_name}] fine-tuning full model for {args.fine_tune_epochs} epochs")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2)
        fine_tune_patience = args.fine_tune_patience if args.fine_tune_patience is not None else args.patience
        fine_tune_patience_count = 0

        for step in range(1, args.fine_tune_epochs + 1):
            epoch = args.epochs + step
            train_stats = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
            val_stats = run_epoch(model, val_loader, criterion, device)
            scheduler.step(val_stats["loss"])
            row = {
                "epoch": epoch,
                "phase": "fine_tune",
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in val_stats.items()},
            }
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / f"{model_name}_history.csv", index=False)
            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                fine_tune_patience_count = 0
                save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_loss, class_names, args)
            else:
                fine_tune_patience_count += 1
                if fine_tune_patience_count >= fine_tune_patience:
                    print(f"[{model_name}] fine-tune early stopping at epoch {epoch}")
                    break

    load_checkpoint(checkpoint_path, model, None, device)
    metrics = evaluate_and_save(model, val_loader, val_df, class_names, run_dir, model_name, device, criterion)
    print(
        f"[{model_name}] val_acc={metrics['val_accuracy']:.4f}, "
        f"top3={metrics['val_top3_accuracy']:.4f}, auc={metrics['roc_auc_macro_ovr']}"
    )
    return metrics


def train_one_model_with_oom_retry(
    model_name: str,
    df_splits: tuple[pd.DataFrame, pd.DataFrame],
    class_names: list[str],
    label_to_idx: dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    batch_size = initial_batch_size(model_name, args)
    attempted = []

    while batch_size >= 1:
        attempted.append(batch_size)
        try:
            metrics = train_one_model(model_name, df_splits, class_names, label_to_idx, args, device, batch_size)
            metrics["batch_size"] = batch_size
            metrics["batch_size_attempts"] = attempted
            return metrics
        except Exception as exc:
            if not is_cuda_oom(exc) or batch_size == 1:
                raise
            print(f"[{model_name}] CUDA OOM at batch_size={batch_size}; retrying with batch_size={batch_size // 2}")
            del exc
            clear_cuda_memory()
            batch_size //= 2

    raise RuntimeError(f"{model_name} failed before training with attempted batch sizes: {attempted}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = resolve_data_dir(args.data_dir)

    models = expand_models(args.models)
    print(f"Data dir: {data_dir}")
    df = load_dataframe(data_dir, args.image_type)
    if args.image_type == "all":
        df = to_paired_lesion_dataframe(df)
    class_names = sorted(df["label"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    train_df, val_df = lesion_level_train_val_split(df, args.val_size, args.seed)

    print("Models:", ", ".join(models))
    print("Classes:", class_names)
    unit_name = "lesions" if args.image_type == "all" else "images"
    print(f"{unit_name.title()}: train={len(train_df)}, val={len(val_df)}, total={len(df)}")
    print("Train label counts:")
    print(train_df["label"].value_counts().sort_index().to_string())

    split_dir = args.output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)

    summary = {}
    for model_name in models:
        try:
            summary[model_name] = train_one_model_with_oom_retry(
                model_name,
                (train_df, val_df),
                class_names,
                label_to_idx,
                args,
                device,
            )
        except Exception as exc:
            print(f"[{model_name}] FAILED: {exc}")
            summary[model_name] = {"error": str(exc)}
        with open(args.output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
