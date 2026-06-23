"""Losses, prediction metrics, calibration, and serialization helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_auc(y_true_bin, y_prob, average):
    try:
        return float(roc_auc_score(y_true_bin, y_prob, average=average, multi_class="ovr"))
    except ValueError:
        return None


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]):
    y_pred = y_prob.argmax(axis=1)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    total = int(cm.sum())
    rows = []
    for idx, name in enumerate(class_names):
        tp = int(cm[idx, idx]); fn = int(cm[idx].sum() - tp); fp = int(cm[:, idx].sum() - tp); tn = total - tp - fn - fp
        binary = (y_true == idx).astype(np.int64)
        try:
            auc = float(roc_auc_score(binary, y_prob[:, idx]))
        except ValueError:
            auc = None
        rows.append({
            "class": name, "support": int(support[idx]), "precision": float(precision[idx]),
            "recall_sensitivity": float(recall[idx]), "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "f1": float(f1[idx]), "auc_ovr": auc,
        })
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    y_true_bin = label_binarize(y_true, classes=labels)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "top2_accuracy": float(np.mean((np.argsort(y_prob, axis=1)[:, -min(2, len(labels)):] == y_true[:, None]).any(axis=1))),
        "top3_accuracy": float(np.mean((np.argsort(y_prob, axis=1)[:, -min(3, len(labels)):] == y_true[:, None]).any(axis=1))),
        "precision_macro": float(macro[0]), "recall_macro": float(macro[1]), "f1_macro": float(macro[2]),
        "dice_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]), "recall_weighted": float(weighted[1]), "f1_weighted": float(weighted[2]),
        "roc_auc_macro_ovr": safe_auc(y_true_bin, y_prob, "macro"),
        "roc_auc_weighted_ovr": safe_auc(y_true_bin, y_prob, "weighted"),
        "specificity_macro": float(np.mean([row["specificity"] for row in rows])),
        "per_class": rows,
        "classification_report": classification_report(y_true, y_pred, labels=labels, target_names=class_names, zero_division=0, output_dict=True),
    }
    return metrics, pd.DataFrame(rows), cm


def apply_class_bias(y_prob: np.ndarray, bias: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(y_prob, 1e-12, 1.0)) + bias[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def optimize_class_bias(y_true, y_prob, class_names, max_bias=1.5, step=0.25, passes=3, metric_name="f1_macro"):
    bias = np.zeros(len(class_names), dtype=np.float32)
    best = compute_metrics(y_true, y_prob, class_names)[0][metric_name]
    candidates = np.arange(-max_bias, max_bias + step / 2, step)
    for _ in range(passes):
        improved = False
        for index in range(len(class_names)):
            local_best, local_value = best, bias[index]
            for value in candidates:
                trial = bias.copy(); trial[index] = value
                score = compute_metrics(y_true, apply_class_bias(y_prob, trial), class_names)[0][metric_name]
                if score > local_best:
                    local_best, local_value = score, float(value)
            if local_best > best:
                bias[index], best, improved = local_value, local_best, True
        if not improved:
            break
    return bias, float(best)


def json_safe(value: Any):
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return value


def save_evaluation(output_dir: Path, y_true, y_prob, df, class_names, prefix=""):
    metrics, per_class, cm = compute_metrics(y_true, y_prob, class_names)
    stem = f"{prefix}_" if prefix else ""
    (output_dir / f"{stem}metrics.json").write_text(json.dumps(json_safe(metrics), indent=2), encoding="utf-8")
    per_class.to_csv(output_dir / f"{stem}per_class_metrics.csv", index=False)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / f"{stem}confusion_matrix.csv")
    predictions = pd.DataFrame(y_prob, columns=class_names)
    predictions.insert(0, "true_label", [class_names[i] for i in y_true])
    predictions.insert(0, "lesion_id", df["lesion_id"].tolist())
    predictions.to_csv(output_dir / f"{stem}val_predictions.csv", index=False)
    return metrics
