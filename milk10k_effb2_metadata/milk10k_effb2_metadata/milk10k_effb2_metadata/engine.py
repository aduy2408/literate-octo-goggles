"""Epoch and phase execution for metadata model training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from milk10k_effb2_metadata.metrics import macro_dice_from_confusion_matrix, move_batch
from milk10k_effb2_metadata.model_setup import build_optimizer
from milk10k_effb2_metadata.models import DualEffB2MetadataClassifier, set_encoder_trainable
from milk10k_effb2_metadata.training_utils import json_safe


def metric_name(label: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in label).strip("_")


def run_epoch(
    model: DualEffB2MetadataClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
    tail_class_indices: list[int] | None = None,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    top3_correct = 0
    total = 0
    preds_all = []
    labels_all = []

    for batch in tqdm(loader, leave=False):
        clinical, dermoscopic, metadata, labels = move_batch(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast("cuda", enabled=use_amp):
                logits = model(clinical, dermoscopic, metadata)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        topk = min(3, logits.size(1))
        top3_correct += logits.topk(topk, dim=1).indices.eq(labels[:, None]).any(dim=1).sum().item()
        total += batch_size
        preds_all.append(logits.argmax(dim=1).detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    y_pred = np.concatenate(preds_all) if preds_all else np.array([])
    y_true = np.concatenate(labels_all) if labels_all else np.array([])

    stats = {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if total else 0.0,
        "f1_macro": float(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]) if total else 0.0,
        "top3_accuracy": top3_correct / max(total, 1),
    }
    if total and class_names:
        labels = list(range(len(class_names)))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        stats["dice_macro"] = macro_dice_from_confusion_matrix(cm)
        for idx, class_name in enumerate(class_names):
            name = metric_name(class_name)
            row_total = int(cm[idx, :].sum())
            stats[f"support_{name}"] = float(support[idx])
            stats[f"precision_{name}"] = float(precision[idx])
            stats[f"recall_{name}"] = float(recall[idx])
            stats[f"f1_{name}"] = float(f1[idx])
            stats[f"correct_{name}"] = float(cm[idx, idx])
            for pred_idx, pred_name in enumerate(class_names):
                if pred_idx == idx:
                    continue
                count = int(cm[idx, pred_idx])
                if count <= 0:
                    continue
                pred_metric = metric_name(pred_name)
                stats[f"conf_{name}_to_{pred_metric}_count"] = float(count)
                stats[f"conf_{name}_to_{pred_metric}_rate"] = count / row_total if row_total else 0.0
    if tail_class_indices:
        recalls = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=tail_class_indices,
            average=None,
            zero_division=0,
        )[1]
        stats["tail_recall_macro"] = float(np.mean(recalls)) if len(recalls) else 0.0
    return stats


def format_class_diagnostics(stats: dict[str, float], class_name: str, class_names: list[str]) -> str:
    name = metric_name(class_name)
    support = int(stats.get(f"support_{name}", 0.0))
    correct = int(stats.get(f"correct_{name}", 0.0))
    recall = stats.get(f"recall_{name}", 0.0)
    precision = stats.get(f"precision_{name}", 0.0)
    f1 = stats.get(f"f1_{name}", 0.0)
    wrongs = []
    for pred_name in class_names:
        if pred_name == class_name:
            continue
        pred_metric = metric_name(pred_name)
        count = int(stats.get(f"conf_{name}_to_{pred_metric}_count", 0.0))
        if count > 0:
            rate = stats.get(f"conf_{name}_to_{pred_metric}_rate", 0.0)
            wrongs.append((count, pred_name, rate))
    wrongs.sort(reverse=True)
    wrong_text = ", ".join(f"{pred}={count} ({rate:.0%})" for count, pred, rate in wrongs[:3]) or "none"
    return (
        f"{class_name}: n={support} correct={correct} recall={recall:.3f} "
        f"precision={precision:.3f} f1={f1:.3f} wrong_to=[{wrong_text}]"
    )


def save_checkpoint(
    path: Path,
    model: DualEffB2MetadataClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    phase: str,
    best_val_f1: float,
    class_names: list[str],
    label_to_idx: dict[str, int],
    metadata_spec: dict[str, Any],
    args: argparse.Namespace,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "phase": phase,
        "model_state": model.state_dict(),
        "model_type": model.__class__.__name__,
        "optimizer_state": optimizer.state_dict(),
        "best_val_f1_macro": best_val_f1,
        "best_selection_metric": best_val_f1,
        "selection_metric_name": args.selection_metric,
        "class_names": class_names,
        "label_to_idx": label_to_idx,
        "metadata_spec": metadata_spec,
        "args": json_safe(vars(args)),
    }
    if extra:
        payload.update(json_safe(extra))
    torch.save(payload, path)


def train_phase(
    phase: str,
    num_epochs: int,
    start_epoch: int,
    model: DualEffB2MetadataClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    class_names: list[str],
    label_to_idx: dict[str, int],
    metadata_spec: dict[str, Any],
    output_dir: Path,
    history: list[dict[str, Any]],
    best_val_f1: float,
    skip_until_epoch: int = 1,
    tail_class_indices: list[int] | None = None,
    tail_class_names: list[str] | None = None,
    train_class_counts: dict[str, int] | None = None,
    best_val_tail_recall: float = float("-inf"),
) -> tuple[int, float, float]:
    if num_epochs <= 0:
        return start_epoch, best_val_f1, best_val_tail_recall

    encoders_trainable = phase == "finetune"
    set_encoder_trainable(model, encoders_trainable)
    optimizer = build_optimizer(model, args, encoders_trainable)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=2)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    use_amp = args.amp and device.type == "cuda"
    patience_count = 0

    print(f"\nPhase: {phase}, epochs={num_epochs}, encoders_trainable={encoders_trainable}")
    for local_epoch in range(1, num_epochs + 1):
        epoch = start_epoch + local_epoch - 1
        if epoch < skip_until_epoch:
            print(f"Skipping already completed {phase} epoch {epoch:03d}")
            continue
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)
        train_stats = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            use_amp,
            tail_class_indices,
            class_names,
        )
        val_stats = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            tail_class_indices=tail_class_indices,
            class_names=class_names,
        )
        selection_metric = args.selection_metric
        scheduler.step(val_stats[selection_metric])
        row = {
            "phase": phase,
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_stats.items()},
            **{f"val_{key}": value for key, value in val_stats.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        print(
            f"{phase} epoch {epoch:03d}: "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
            f"train_bal_acc={train_stats['balanced_accuracy']:.4f} train_f1={train_stats['f1_macro']:.4f} "
            f"val_acc={val_stats['accuracy']:.4f} val_bal_acc={val_stats['balanced_accuracy']:.4f} "
            f"val_f1={val_stats['f1_macro']:.4f} val_dice={val_stats.get('dice_macro', 0.0):.4f} "
            f"val_top3={val_stats['top3_accuracy']:.4f}"
        )
        if tail_class_indices:
            print(
                f"LDAM tail: classes={tail_class_names} "
                f"train_tail_recall={train_stats['tail_recall_macro']:.4f} "
                f"val_tail_recall={val_stats['tail_recall_macro']:.4f}"
            )
            for class_name in tail_class_names or []:
                print(f"  train {format_class_diagnostics(train_stats, class_name, class_names)}")
                print(f"  val   {format_class_diagnostics(val_stats, class_name, class_names)}")

        if val_stats[selection_metric] > best_val_f1:
            best_val_f1 = val_stats[selection_metric]
            patience_count = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                phase,
                best_val_f1,
                class_names,
                label_to_idx,
                metadata_spec,
                args,
            )
            print(
                f"Saved best checkpoint: phase={phase} epoch={epoch:03d} "
                f"best_{selection_metric}={best_val_f1:.4f} path={output_dir / 'best.pt'}"
            )
        else:
            patience_count += 1

        if tail_class_indices and val_stats["tail_recall_macro"] > best_val_tail_recall:
            best_val_tail_recall = val_stats["tail_recall_macro"]
            save_checkpoint(
                output_dir / "tail_best.pt",
                model,
                optimizer,
                epoch,
                phase,
                best_val_f1,
                class_names,
                label_to_idx,
                metadata_spec,
                args,
                {
                    "best_val_tail_recall_macro": best_val_tail_recall,
                    "tail_class_names": tail_class_names or [],
                    "tail_class_indices": tail_class_indices,
                    "train_class_counts": train_class_counts or {},
                    "selection_metric": "val_tail_recall_macro",
                },
            )
            print(
                f"Saved tail checkpoint: phase={phase} epoch={epoch:03d} "
                f"best_val_tail_recall_macro={best_val_tail_recall:.4f} path={output_dir / 'tail_best.pt'}"
            )

        if patience_count >= args.patience:
            print(f"Early stopping {phase} at epoch {epoch}")
            break

    return epoch + 1, best_val_f1, best_val_tail_recall
