#!/usr/bin/env python3
"""
Train a dual-encoder MILK10k classifier using paired clinical and dermoscopic images.

This script keeps the original baseline script intact and reuses its data
loading, split, loss, checkpoint, and metrics helpers. The model path here is
explicitly dual-input:

  clinical image -> clinical encoder ----\
                                       fusion -> classifier
  dermoscopic image -> dermoscopic encoder -/
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
from timm.data import create_transform, resolve_data_config
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import (
    PairedMilk10kDataset,
    lesion_level_train_val_split,
    load_dataframe,
    resolve_data_dir,
    set_seed,
    to_paired_lesion_dataframe,
)
from milk10k_dual_encoder.INDIVIDUAL_SCRIPTS.train_milk10k_baselines import (
    build_loss,
    clear_cuda_memory,
    compute_classification_metrics,
    is_cuda_oom,
    load_checkpoint,
    save_checkpoint,
)


@dataclass(frozen=True)
class DualModelSpec:
    key: str
    timm_name: str
    image_size: int


MODEL_SPECS: dict[str, DualModelSpec] = {
    "resnet50": DualModelSpec("resnet50", "resnet50", 224),
    "densenet121": DualModelSpec("densenet121", "densenet121", 224),
    "efficientnet_b0": DualModelSpec("efficientnet_b0", "efficientnet_b0", 224),
    "efficientnet_b3": DualModelSpec("efficientnet_b3", "efficientnet_b3", 300),
    "convnext_tiny": DualModelSpec("convnext_tiny", "convnext_tiny", 224),
    "convnext_small": DualModelSpec("convnext_small", "convnext_small", 224),
}


class DualEncoderClassifier(nn.Module):
    def __init__(
        self,
        timm_name: str,
        num_classes: int,
        pretrained: bool = True,
        shared_encoder: bool = False,
        fusion: str = "concat",
        dropout: float = 0.3,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.fusion = fusion
        self.clinical_encoder = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        if shared_encoder:
            self.dermoscopic_encoder = self.clinical_encoder
        else:
            self.dermoscopic_encoder = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")

        feature_dim = int(self.clinical_encoder.num_features)
        fused_dim = {
            "concat": feature_dim * 2,
            "concat_diff": feature_dim * 3,
            "concat_diff_product": feature_dim * 4,
        }[fusion]

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def fuse_features(self, clinical_features: torch.Tensor, dermoscopic_features: torch.Tensor) -> torch.Tensor:
        if self.fusion == "concat":
            return torch.cat([clinical_features, dermoscopic_features], dim=1)
        if self.fusion == "concat_diff":
            return torch.cat(
                [clinical_features, dermoscopic_features, torch.abs(clinical_features - dermoscopic_features)],
                dim=1,
            )
        if self.fusion == "concat_diff_product":
            return torch.cat(
                [
                    clinical_features,
                    dermoscopic_features,
                    torch.abs(clinical_features - dermoscopic_features),
                    clinical_features * dermoscopic_features,
                ],
                dim=1,
            )
        raise ValueError(f"Unknown fusion mode: {self.fusion}")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        clinical = images[:, 0]
        dermoscopic = images[:, 1]
        clinical_features = self.clinical_encoder(clinical)
        dermoscopic_features = self.dermoscopic_encoder(dermoscopic)
        fused = self.fuse_features(clinical_features, dermoscopic_features)
        return self.classifier(fused)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MILK10k dual-encoder paired-image classifier.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("dual_encoder_runs_torch"))
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="convnext_tiny")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=None, help="Optional lower LR for encoders. Defaults to --lr.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--fusion", choices=["concat", "concat_diff", "concat_diff_product"], default="concat_diff_product")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--shared-encoder", action="store_true", help="Use one shared encoder for both modalities.")
    parser.add_argument("--freeze-encoders", action="store_true", help="Train only the fusion classifier.")
    parser.add_argument("--unfreeze-epoch", type=int, default=0, help="Unfreeze encoders at this epoch, 0 disables.")
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def make_paired_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_to_idx: dict[str, int],
    model: nn.Module,
    image_size: int,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    data_config = resolve_data_config({}, model=model.clinical_encoder)
    data_config["input_size"] = (3, image_size, image_size)
    train_transform = create_transform(**data_config, is_training=True)
    eval_transform = create_transform(**data_config, is_training=False)

    train_ds = PairedMilk10kDataset(train_df, label_to_idx, train_transform)
    val_ds = PairedMilk10kDataset(val_df, label_to_idx, eval_transform)
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return DataLoader(train_ds, shuffle=True, **common), DataLoader(val_ds, shuffle=False, **common)


def set_encoder_trainable(model: DualEncoderClassifier, trainable: bool) -> None:
    for param in model.clinical_encoder.parameters():
        param.requires_grad = trainable
    for param in model.dermoscopic_encoder.parameters():
        param.requires_grad = trainable


def build_optimizer(model: DualEncoderClassifier, args: argparse.Namespace) -> torch.optim.Optimizer:
    encoder_lr = args.encoder_lr if args.encoder_lr is not None else args.lr
    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    encoder_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("classifier.")
    ]
    param_groups = [{"params": classifier_params, "lr": args.lr}]
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": encoder_lr})
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


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
            with autocast(enabled=use_amp):
                logits = model(images)
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
        logits = model(images)
        labels_all.append(labels.numpy())
        probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(probs_all)


def evaluate_dual_and_save(
    model: nn.Module,
    eval_loader: DataLoader,
    eval_df: pd.DataFrame,
    class_names: list[str],
    run_dir: Path,
    model_name: str,
    device: torch.device,
    criterion: nn.Module,
) -> dict:
    eval_stats = run_epoch(model, eval_loader, criterion, device)
    y_true, y_prob = predict(model, eval_loader, device)
    y_pred = y_prob.argmax(axis=1)
    cls_metrics, per_class_df, cm = compute_classification_metrics(y_true, y_pred, y_prob, class_names)
    metrics = {
        "val_loss": float(eval_stats["loss"]),
        "val_accuracy": cls_metrics["accuracy"],
        "val_balanced_accuracy": cls_metrics["balanced_accuracy"],
        "val_top2_accuracy": cls_metrics["top2_accuracy"],
        "val_top3_accuracy": cls_metrics["top3_accuracy"],
        **cls_metrics,
    }

    with open(run_dir / f"{model_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(run_dir / f"{model_name}_confusion_matrix.csv")
    per_class_df.to_csv(run_dir / f"{model_name}_per_class_metrics.csv", index=False)
    prediction_df = pd.DataFrame(
        {
            "lesion_id": eval_df["lesion_id"].tolist(),
            "clinical_path": eval_df["clinical_path"].tolist(),
            "dermoscopic_path": eval_df["dermoscopic_path"].tolist(),
            "y_true": y_true,
            "y_pred": y_pred,
            "label_true": [class_names[i] for i in y_true],
            "label_pred": [class_names[i] for i in y_pred],
            "confidence": y_prob.max(axis=1),
        }
    )
    probability_df = pd.DataFrame(y_prob, columns=[f"prob_{name}" for name in class_names])
    pd.concat([prediction_df, probability_df], axis=1).to_csv(run_dir / f"{model_name}_val_predictions.csv", index=False)
    return metrics


def train_once(args: argparse.Namespace, device: torch.device) -> dict:
    data_dir = resolve_data_dir(args.data_dir)
    spec = MODEL_SPECS[args.model]
    image_size = args.image_size or spec.image_size
    run_name = f"dual_{args.model}_{args.fusion}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / f"{run_name}_best.pt"

    df = load_dataframe(data_dir, "all")
    paired_df = to_paired_lesion_dataframe(df)
    class_names = sorted(paired_df["label"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    train_df, val_df = lesion_level_train_val_split(paired_df, args.val_size, args.seed)

    split_dir = run_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)

    model = DualEncoderClassifier(
        timm_name=spec.timm_name,
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
        shared_encoder=args.shared_encoder,
        fusion=args.fusion,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
    ).to(device)
    if args.freeze_encoders:
        set_encoder_trainable(model, False)

    train_loader, val_loader = make_paired_loaders(train_df, val_df, label_to_idx, model, image_size, args)
    criterion = build_loss(train_df, label_to_idx, args, device)
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    use_amp = args.amp and device.type == "cuda"

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume and checkpoint_path.exists():
        start_epoch, best_val_loss = load_checkpoint(checkpoint_path, model, optimizer, device)
        print(f"Resumed from {checkpoint_path} at epoch {start_epoch}")

    print(f"Data dir: {data_dir}")
    print(f"Model: {run_name}, timm={spec.timm_name}, image_size={image_size}, device={device}")
    print(f"Paired lesions: train={len(train_df)}, val={len(val_df)}, total={len(paired_df)}")
    print("Classes:", class_names)
    print("Train label counts:")
    print(train_df["label"].value_counts().sort_index().to_string())

    history = []
    patience_count = 0
    unfroze = not args.freeze_encoders
    for epoch in range(start_epoch, args.epochs + 1):
        if args.unfreeze_epoch and not unfroze and epoch >= args.unfreeze_epoch:
            print(f"Unfreezing encoders at epoch {epoch}")
            set_encoder_trainable(model, True)
            optimizer = build_optimizer(model, args)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=2)
            unfroze = True

        train_stats = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        val_stats = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_stats["loss"])

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / f"{run_name}_history.csv", index=False)
        print(
            f"epoch {epoch:03d}: train_loss={train_stats['loss']:.4f} "
            f"val_loss={val_stats['loss']:.4f} val_acc={val_stats['accuracy']:.4f}"
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            patience_count = 0
            save_checkpoint(checkpoint_path, model, optimizer, epoch, best_val_loss, class_names, args)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    load_checkpoint(checkpoint_path, model, None, device)
    metrics = evaluate_dual_and_save(model, val_loader, val_df, class_names, run_dir, run_name, device, criterion)
    with open(args.output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump({run_name: metrics}, f, indent=2)
    print(
        f"{run_name}: val_acc={metrics['val_accuracy']:.4f}, "
        f"balanced_acc={metrics['val_balanced_accuracy']:.4f}, auc={metrics['roc_auc_macro_ovr']}"
    )
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    attempted = []
    while args.batch_size >= 1:
        attempted.append(args.batch_size)
        try:
            train_once(args, device)
            return
        except Exception as exc:
            if not is_cuda_oom(exc) or args.batch_size == 1:
                raise
            print(f"CUDA OOM at batch_size={args.batch_size}; retrying with batch_size={args.batch_size // 2}")
            del exc
            clear_cuda_memory()
            args.batch_size //= 2

    raise RuntimeError(f"Training failed before completion with attempted batch sizes: {attempted}")


if __name__ == "__main__":
    main()
