"""Metrics and reporting helpers for collapse-focused experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from milk10k_new_collapse_research.compat import ensure_legacy_package_path

ensure_legacy_package_path()
from milk10k_effb2_metadata.metrics import compute_metrics, save_predictions
from milk10k_new_collapse_research.config import BASELINE_F1_MACRO, TAIL_GATE_CLASSES


def write_standard_outputs(
    output_dir: Path,
    val_df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, per_class_df, cm = compute_metrics(y_true, y_prob, class_names)
    if extra:
        metrics.update(extra)
    (output_dir / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2), encoding="utf-8")
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(output_dir / "confusion_matrix.csv")
    save_predictions(val_df, y_true, y_prob, class_names, output_dir)
    (output_dir / "collapse_report.md").write_text(
        build_collapse_report(metrics, cm, class_names),
        encoding="utf-8",
    )
    return metrics


def build_collapse_report(metrics: dict[str, Any], cm: np.ndarray, class_names: list[str]) -> str:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    lines = [
        "# Collapse Report",
        "",
        f"- f1_macro: `{metrics.get('f1_macro', 0.0):.4f}`",
        f"- baseline_f1_macro: `{BASELINE_F1_MACRO:.4f}`",
        f"- balanced_accuracy: `{metrics.get('balanced_accuracy', 0.0):.4f}`",
        f"- accuracy: `{metrics.get('accuracy', 0.0):.4f}`",
        "",
        "## Tail Gates",
        "",
        "| Class | Support | Correct | Recall | F1 | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    per_class = {row["class"]: row for row in metrics.get("per_class", [])}
    for label in TAIL_GATE_CLASSES:
        if label not in class_to_idx:
            continue
        idx = class_to_idx[label]
        row = per_class.get(label, {})
        support = int(cm[idx].sum())
        correct = int(cm[idx, idx])
        recall = float(row.get("recall_sensitivity", 0.0))
        f1 = float(row.get("f1", 0.0))
        gate = tail_gate_status(label, correct, f1)
        lines.append(f"| `{label}` | {support} | {correct} | {recall:.4f} | {f1:.4f} | {gate} |")

    lines.extend(["", "## Key Confusions", ""])
    for src, dst in [("MEL", "NV"), ("NV", "MEL"), ("INF", "BCC")]:
        if src in class_to_idx and dst in class_to_idx:
            lines.append(f"- `{src} -> {dst}`: `{int(cm[class_to_idx[src], class_to_idx[dst]])}`")
    for src in ["AKIEC", "BCC", "BKL", "SCCKA"]:
        if src not in class_to_idx:
            continue
        idx = class_to_idx[src]
        cluster = [label for label in ["AKIEC", "BCC", "BKL", "SCCKA"] if label in class_to_idx and label != src]
        total = sum(int(cm[idx, class_to_idx[dst]]) for dst in cluster)
        lines.append(f"- `{src}` errors into keratinocyte-like cluster: `{total}`")

    return "\n".join(lines) + "\n"


def tail_gate_status(label: str, correct: int, f1: float) -> str:
    if label == "BEN_OTH":
        return "PASS" if correct > 2 or f1 > 0.3077 else "FAIL"
    if label == "INF":
        return "PASS" if correct >= 5 else "FAIL"
    if label == "MAL_OTH":
        return "PASS" if correct >= 1 else "UNRESOLVED"
    return "CHECK"


def load_prediction_csv(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    class_names = [col.removeprefix("prob_") for col in prob_cols]
    true_col = "y_true" if "y_true" in df.columns else "true_label"
    if true_col == "true_label" and not np.issubdtype(df[true_col].dtype, np.number):
        label_true_col = "label_true" if "label_true" in df.columns else "true_label"
        label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        y_true = df[label_true_col].map(label_to_idx).to_numpy(dtype=np.int64)
    else:
        y_true = df[true_col].to_numpy(dtype=np.int64)
        return df, y_true, df[prob_cols].to_numpy(dtype=np.float64), class_names

    return df, y_true, df[prob_cols].to_numpy(dtype=np.float64), class_names


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
