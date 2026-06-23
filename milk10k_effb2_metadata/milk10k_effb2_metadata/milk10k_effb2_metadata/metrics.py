"""Prediction and classification metric helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from milk10k_effb2_metadata.models import DualEffB2MetadataClassifier


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clinical = batch["clinical"].to(device, non_blocking=True)
    dermoscopic = batch["dermoscopic"].to(device, non_blocking=True)
    metadata = batch["metadata"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return clinical, dermoscopic, metadata, labels


@torch.no_grad()
def predict(model: DualEffB2MetadataClassifier, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all = []
    probs_all = []
    for batch in tqdm(loader, leave=False):
        clinical, dermoscopic, metadata, labels = move_batch(batch, device)
        logits = model(clinical, dermoscopic, metadata)
        labels_all.append(labels.cpu().numpy())
        probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(labels_all), np.concatenate(probs_all)


def macro_dice_from_confusion_matrix(cm: np.ndarray) -> float:
    per_class = []
    for idx in range(cm.shape[0]):
        tp = float(cm[idx, idx])
        fn = float(cm[idx, :].sum() - tp)
        fp = float(cm[:, idx].sum() - tp)
        denom = 2.0 * tp + fp + fn
        per_class.append(0.0 if denom <= 0.0 else (2.0 * tp) / denom)
    return float(np.mean(per_class)) if per_class else 0.0


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    y_pred = y_prob.argmax(axis=1)
    labels = list(range(len(class_names)))
    y_true_bin = label_binarize(y_true, classes=labels)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

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
        try:
            auc_ovr = float(roc_auc_score(y_true_bin[:, idx], y_prob[:, idx]))
        except ValueError:
            auc_ovr = None
        per_class_rows.append(
            {
                "class": class_name,
                "support": int(support_per_class[idx]),
                "precision": float(precision_per_class[idx]),
                "recall_sensitivity": float(recall_per_class[idx]),
                "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
                "f1": float(f1_per_class[idx]),
                "auc_ovr": auc_ovr,
            }
        )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "top2_accuracy": float(np.mean((np.argsort(y_prob, axis=1)[:, -min(2, len(class_names)) :] == y_true[:, None]).any(axis=1))),
        "top3_accuracy": float(np.mean((np.argsort(y_prob, axis=1)[:, -min(3, len(class_names)) :] == y_true[:, None]).any(axis=1))),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "dice_macro": macro_dice_from_confusion_matrix(cm),
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


def safe_roc_auc(y_true_bin: np.ndarray, y_prob: np.ndarray, average: str | None) -> float | None:
    try:
        return float(roc_auc_score(y_true_bin, y_prob, average=average, multi_class="ovr"))
    except ValueError:
        return None


def save_predictions(
    val_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    y_pred = y_prob.argmax(axis=1)
    prediction_df = pd.DataFrame(
        {
            "lesion_id": val_df["lesion_id"].tolist(),
            "clinical_path": val_df["clinical_path"].tolist(),
            "dermoscopic_path": val_df["dermoscopic_path"].tolist(),
            "y_true": y_true,
            "y_pred": y_pred,
            "label_true": [class_names[idx] for idx in y_true],
            "label_pred": [class_names[idx] for idx in y_pred],
            "confidence": y_prob.max(axis=1),
        }
    )
    probability_df = pd.DataFrame(y_prob, columns=[f"prob_{name}" for name in class_names])
    pd.concat([prediction_df, probability_df], axis=1).to_csv(output_dir / "val_predictions.csv", index=False)


def apply_class_bias(y_prob: np.ndarray, bias: np.ndarray) -> np.ndarray:
    log_prob = np.log(np.clip(y_prob, 1e-12, 1.0))
    adjusted = log_prob + bias[None, :]
    adjusted -= adjusted.max(axis=1, keepdims=True)
    exp_scores = np.exp(adjusted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def metric_value_from_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    metric_name: str,
) -> float:
    metrics, _, _ = compute_metrics(y_true, y_prob, class_names)
    return float(metrics[metric_name])


def optimize_class_bias(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    metric_name: str = "dice_macro",
    max_bias: float = 1.5,
    step: float = 0.25,
    passes: int = 3,
) -> tuple[np.ndarray, float]:
    bias = np.zeros(len(class_names), dtype=np.float32)
    best_score = metric_value_from_probabilities(y_true, y_prob, class_names, metric_name)
    current_step = step

    for _ in range(max(1, passes)):
        deltas = np.arange(-max_bias, max_bias + current_step * 0.5, current_step, dtype=np.float32)
        improved = False
        for class_idx in range(len(class_names)):
            best_class_bias = float(bias[class_idx])
            best_class_score = best_score
            for delta in deltas:
                trial_bias = bias.copy()
                trial_bias[class_idx] = best_class_bias + float(delta)
                trial_prob = apply_class_bias(y_prob, trial_bias)
                score = metric_value_from_probabilities(y_true, trial_prob, class_names, metric_name)
                if score > best_class_score + 1e-12:
                    best_class_score = score
                    best_class_bias = float(trial_bias[class_idx])
            if best_class_score > best_score + 1e-12:
                bias[class_idx] = best_class_bias
                best_score = best_class_score
                improved = True
        current_step = max(current_step / 2.0, 0.01)
        if not improved:
            break

    return bias, best_score
