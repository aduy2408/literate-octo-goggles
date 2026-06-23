"""Run reporting helpers for MILK10k training outputs."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from milk10k_effb2_metadata.training_utils import json_safe


WATCHED_CONFUSIONS = {
    "INF": ["BEN_OTH", "NV", "BCC"],
    "BCC": ["AKIEC", "BKL", "SCCKA"],
    "SCCKA": ["AKIEC", "BKL"],
    "AKIEC": ["SCCKA"],
    "BKL": ["SCCKA"],
}


def collect_environment_info() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(Path.cwd()),
        "command": sys.argv,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch

        payload["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover - defensive only.
        payload["torch"] = {"error": repr(exc)}

    payload["git"] = git_info(Path.cwd())
    return payload


def git_info(cwd: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    if commit is None:
        return {"available": False}
    status = run_git(["status", "--short"]) or ""
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return {
        "available": True,
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def class_distribution(df: pd.DataFrame, class_names: list[str]) -> dict[str, Any]:
    counts = df["label"].value_counts().reindex(class_names, fill_value=0).astype(int).to_dict()
    is_augmented = synthetic_mask(df)
    ignore_metadata = (
        df["ignore_metadata"].fillna(False).astype(bool).to_numpy()
        if "ignore_metadata" in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    augmented_counts = (
        df.loc[is_augmented, "label"].value_counts().reindex(class_names, fill_value=0).astype(int).to_dict()
        if len(df)
        else {name: 0 for name in class_names}
    )
    mask_status_counts = (
        df["dermoscopic_mask_status"].fillna("not_audited").value_counts().sort_index().astype(int).to_dict()
        if "dermoscopic_mask_status" in df.columns
        else {}
    )
    return {
        "rows": int(len(df)),
        "class_counts": counts,
        "real_rows": int((~is_augmented).sum()),
        "synthetic_rows": int(is_augmented.sum()),
        "synthetic_class_counts": augmented_counts,
        "ignore_metadata_rows": int(ignore_metadata.sum()),
        "dermoscopic_mask_status_counts": mask_status_counts,
    }


def synthetic_mask(df: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(df), dtype=bool)
    if "is_augmented" in df.columns:
        mask |= df["is_augmented"].fillna(False).astype(bool).to_numpy()
    if "lesion_id" in df.columns:
        mask |= df["lesion_id"].astype(str).str.contains("__sdpair_", regex=False).to_numpy()
    return mask


def build_data_summary(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_names: list[str],
) -> dict[str, Any]:
    return {
        "full": class_distribution(full_df, class_names),
        "train": class_distribution(train_df, class_names),
        "val": class_distribution(val_df, class_names),
        "synthetic_train_only": bool(synthetic_mask(train_df).sum() and not synthetic_mask(val_df).sum()),
    }


def build_prediction_summary(y_prob: np.ndarray, class_names: list[str], low_confidence_threshold: float = 0.5) -> dict[str, Any]:
    if y_prob.size == 0:
        return {
            "rows": 0,
            "predicted_class_counts": {name: 0 for name in class_names},
            "mean_probability": {name: 0.0 for name in class_names},
        }
    y_pred = y_prob.argmax(axis=1)
    counts = np.bincount(y_pred, minlength=len(class_names))
    sorted_prob = np.sort(y_prob, axis=1)
    confidence = sorted_prob[:, -1]
    second = sorted_prob[:, -2] if y_prob.shape[1] > 1 else np.zeros_like(confidence)
    entropy = -np.sum(y_prob * np.log(np.clip(y_prob, 1e-12, 1.0)), axis=1)
    return {
        "rows": int(y_prob.shape[0]),
        "predicted_class_counts": {name: int(counts[idx]) for idx, name in enumerate(class_names)},
        "mean_probability": {name: float(y_prob[:, idx].mean()) for idx, name in enumerate(class_names)},
        "median_probability": {name: float(np.median(y_prob[:, idx])) for idx, name in enumerate(class_names)},
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "mean_top1_top2_gap": float((confidence - second).mean()),
        "median_top1_top2_gap": float(np.median(confidence - second)),
        "mean_entropy": float(entropy.mean()),
        "median_entropy": float(np.median(entropy)),
        "low_confidence_threshold": float(low_confidence_threshold),
        "low_confidence_rows": int((confidence < low_confidence_threshold).sum()),
    }


def build_confusion_analysis(cm: np.ndarray, class_names: list[str], top_k: int = 20) -> dict[str, Any]:
    false_negatives: dict[str, list[dict[str, Any]]] = {}
    false_positives: dict[str, list[dict[str, Any]]] = {}
    pairs = []
    watched = []

    for true_idx, true_name in enumerate(class_names):
        row_total = int(cm[true_idx, :].sum())
        entries = []
        for pred_idx, pred_name in enumerate(class_names):
            if pred_idx == true_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count <= 0:
                continue
            entry = {
                "true": true_name,
                "predicted": pred_name,
                "count": count,
                "rate_of_true": count / row_total if row_total else 0.0,
            }
            entries.append(entry)
            pairs.append(entry)
            if pred_name in WATCHED_CONFUSIONS.get(true_name, []):
                watched.append(entry)
        false_negatives[true_name] = sorted(entries, key=lambda item: item["count"], reverse=True)

    for pred_idx, pred_name in enumerate(class_names):
        col_total = int(cm[:, pred_idx].sum())
        entries = []
        for true_idx, true_name in enumerate(class_names):
            if pred_idx == true_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count <= 0:
                continue
            entries.append(
                {
                    "predicted": pred_name,
                    "true": true_name,
                    "count": count,
                    "rate_of_predicted": count / col_total if col_total else 0.0,
                }
            )
        false_positives[pred_name] = sorted(entries, key=lambda item: item["count"], reverse=True)

    pairs = sorted(pairs, key=lambda item: item["count"], reverse=True)
    watched = sorted(watched, key=lambda item: item["count"], reverse=True)
    return {
        "false_negatives_by_true_class": false_negatives,
        "false_positives_by_predicted_class": false_positives,
        "top_confusion_pairs": pairs[:top_k],
        "watched_confusion_patterns": watched,
    }


def build_run_warnings(
    metrics: dict[str, Any],
    per_class_df: pd.DataFrame,
    cm: np.ndarray,
    prediction_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    del metrics
    class_names = per_class_df["class"].tolist()
    warnings: list[dict[str, Any]] = []
    pred_counts = prediction_summary.get("predicted_class_counts", {})
    total_pred = max(int(prediction_summary.get("rows", 0)), 1)

    for class_name in ("INF", "BEN_OTH"):
        if class_name in pred_counts and int(pred_counts[class_name]) == 0:
            warnings.append(warning("tail_predicted_zero", "high", f"{class_name} has zero predicted rows.", class_name))

    mal_count = int(pred_counts.get("MAL_OTH", 0))
    if mal_count > max(2, math.ceil(total_pred * 0.01)):
        warnings.append(warning("mal_oth_many_predictions", "medium", f"MAL_OTH predicted {mal_count} times.", "MAL_OTH"))

    if "BCC" in class_names:
        bcc_idx = class_names.index("BCC")
        bcc_support = int(cm[bcc_idx, :].sum())
        bcc_pred = int(cm[:, bcc_idx].sum())
        if bcc_support and bcc_pred < max(1, int(bcc_support * 0.65)):
            warnings.append(
                warning("bcc_predicted_low", "high", f"BCC predicted {bcc_pred} times for {bcc_support} validation BCC rows.", "BCC")
            )
        drift_targets = [name for name in ("AKIEC", "BKL", "SCCKA") if name in class_names]
        drift_count = sum(int(cm[bcc_idx, class_names.index(name)]) for name in drift_targets)
        if bcc_support and drift_count >= max(3, int(bcc_support * 0.08)):
            warnings.append(
                warning("bcc_boundary_drift", "high", f"BCC -> AKIEC/BKL/SCCKA count is {drift_count}.", "BCC")
            )

    for row in per_class_df.to_dict("records"):
        class_name = str(row["class"])
        support = int(row.get("support", 0))
        precision = float(row.get("precision", 0.0))
        recall = float(row.get("recall_sensitivity", 0.0))
        if support <= 5:
            warnings.append(
                warning("tiny_validation_support", "medium", f"{class_name} validation support is only {support}.", class_name)
            )
        if class_name in {"BEN_OTH", "DF", "INF", "MAL_OTH", "VASC"} and recall >= 0.2 and precision < 0.2:
            warnings.append(
                warning(
                    "tail_precision_low",
                    "high",
                    f"{class_name} recall={recall:.3f} but precision={precision:.3f}.",
                    class_name,
                )
            )

    mean_conf = float(prediction_summary.get("mean_confidence", 0.0))
    mean_entropy = float(prediction_summary.get("mean_entropy", 0.0))
    if mean_conf > 0.9 and mean_entropy < 0.35:
        warnings.append(
            warning("high_confidence_low_entropy", "medium", f"mean_confidence={mean_conf:.3f}, mean_entropy={mean_entropy:.3f}.")
        )
    return warnings


def warning(code: str, severity: str, message: str, class_name: str | None = None) -> dict[str, Any]:
    payload = {"code": code, "severity": severity, "message": message}
    if class_name is not None:
        payload["class"] = class_name
    return payload


def save_data_summary(output_dir: Path, data_summary: dict[str, Any]) -> None:
    with open(output_dir / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(data_summary), f, indent=2)
    (output_dir / "split_summary.md").write_text(render_split_summary(data_summary), encoding="utf-8")


def save_run_diagnostics(
    output_dir: Path,
    args: Any,
    data_summary: dict[str, Any],
    metrics: dict[str, Any],
    per_class_df: pd.DataFrame,
    cm: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    fold: int | None = None,
) -> dict[str, Any]:
    prediction_summary = build_prediction_summary(y_prob, class_names)
    confusion_analysis = build_confusion_analysis(cm, class_names)
    warnings = build_run_warnings(metrics, per_class_df, cm, prediction_summary)
    diagnostics = {
        "fold": fold,
        "warnings": warnings,
        "prediction_summary": prediction_summary,
        "confusion_analysis": confusion_analysis,
    }
    with open(output_dir / "prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(prediction_summary), f, indent=2)
    with open(output_dir / "confusion_analysis.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(confusion_analysis), f, indent=2)
    with open(output_dir / "run_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(diagnostics), f, indent=2)
    (output_dir / "run_report.md").write_text(
        render_run_report(args, data_summary, metrics, per_class_df, prediction_summary, confusion_analysis, warnings, fold),
        encoding="utf-8",
    )
    return diagnostics


def save_kfold_report(fold_metrics: list[dict[str, Any]], output_dir: Path) -> None:
    diagnostics = []
    for fold_dir in sorted(output_dir.glob("fold_*/run_diagnostics.json")):
        with open(fold_dir, encoding="utf-8") as f:
            payload = json.load(f)
        payload["path"] = str(fold_dir)
        diagnostics.append(payload)
    (output_dir / "kfold_report.md").write_text(render_kfold_report(fold_metrics, diagnostics), encoding="utf-8")


def render_split_summary(data_summary: dict[str, Any]) -> str:
    lines = ["# Split Summary", ""]
    for split in ("full", "train", "val"):
        summary = data_summary[split]
        lines.extend(
            [
                f"## {split.title()}",
                "",
                f"- rows: {summary['rows']}",
                f"- real_rows: {summary['real_rows']}",
                f"- synthetic_rows: {summary['synthetic_rows']}",
                f"- ignore_metadata_rows: {summary['ignore_metadata_rows']}",
                "",
                "| class | count | synthetic |",
                "|---|---:|---:|",
            ]
        )
        for class_name, count in summary["class_counts"].items():
            lines.append(f"| {class_name} | {count} | {summary['synthetic_class_counts'].get(class_name, 0)} |")
        lines.append("")
    lines.append(f"- synthetic_train_only: {data_summary['synthetic_train_only']}")
    balance = data_summary.get("balance")
    if balance:
        lines.extend(
            [
                f"- balance_mode: {balance['mode']}",
                f"- effective_rows_per_epoch: {balance['effective_rows_per_epoch']}",
                f"- strong_augmentation_classes: {balance['strong_augmentation_classes']}",
                "",
                "| class | original train | effective per epoch |",
                "|---|---:|---:|",
            ]
        )
        for class_name, count in balance["original_class_counts"].items():
            effective = balance["effective_class_counts_per_epoch"][class_name]
            lines.append(f"| {class_name} | {count} | {effective} |")
    lines.append("")
    return "\n".join(lines)


def render_run_report(
    args: Any,
    data_summary: dict[str, Any],
    metrics: dict[str, Any],
    per_class_df: pd.DataFrame,
    prediction_summary: dict[str, Any],
    confusion_analysis: dict[str, Any],
    warnings: list[dict[str, Any]],
    fold: int | None,
) -> str:
    lines = ["# MILK10k Run Report", ""]
    lines.extend(
        [
            "## Config Summary",
            "",
            f"- fold: {fold}",
            f"- output_dir: {getattr(args, 'output_dir', None)}",
            f"- backbone: {getattr(args, 'backbone', None)}",
            f"- metadata_fusion: {getattr(args, 'metadata_fusion', None)}",
            f"- image_fusion: {getattr(args, 'image_fusion', None)}",
            f"- loss: {getattr(args, 'loss', None)}",
            f"- class_weight: {getattr(args, 'class_weight', None)}",
            f"- weighted_sampler: {getattr(args, 'weighted_sampler', None)}",
            f"- balance_mode: {getattr(args, 'balance_mode', None)}",
            f"- balance_head_ratio: {getattr(args, 'balance_head_ratio', None)}",
            f"- balance_tail_floor: {getattr(args, 'balance_tail_floor', None)}",
            f"- balance_min_source_count: {getattr(args, 'balance_min_source_count', None)}",
            f"- augmented_data_dir: {getattr(args, 'augmented_data_dir', None)}",
            f"- dermoscopic_mask_dir: {getattr(args, 'dermoscopic_mask_dir', None)}",
            f"- min_dermoscopic_mask_ratio: {getattr(args, 'min_dermoscopic_mask_ratio', None)}",
            f"- augmented_classes: {getattr(args, 'augmented_classes', None)}",
            f"- augmented_max_per_class: {getattr(args, 'augmented_max_per_class', None)}",
            f"- freeze_metadata_head: {getattr(args, 'freeze_metadata_head', None)}",
            f"- zero_augmented_metadata: {getattr(args, 'zero_augmented_metadata', None)}",
            "",
            "## Final Metrics",
            "",
        ]
    )
    for key in ("accuracy", "balanced_accuracy", "dice_macro", "f1_macro", "roc_auc_macro_ovr", "top2_accuracy", "top3_accuracy"):
        lines.append(f"- {key}: {metrics.get(key)}")
    lines.extend(["", "## Data Distribution", "", render_distribution_table(data_summary["train"], "Train"), ""])
    lines.extend([render_distribution_table(data_summary["val"], "Validation"), ""])
    lines.extend(["## Per-Class Metrics", "", dataframe_to_markdown(per_class_df), ""])
    weak = per_class_df.sort_values(["f1", "support"], ascending=[True, True]).head(5)
    lines.extend(["## Weak Classes", "", dataframe_to_markdown(weak), ""])
    lines.extend(["## Prediction Distribution", "", "| class | pred_count | mean_prob |", "|---|---:|---:|"])
    for class_name, count in prediction_summary.get("predicted_class_counts", {}).items():
        mean_prob = prediction_summary.get("mean_probability", {}).get(class_name, 0.0)
        lines.append(f"| {class_name} | {count} | {mean_prob:.4f} |")
    lines.extend(
        [
            "",
            f"- mean_confidence: {prediction_summary.get('mean_confidence')}",
            f"- median_confidence: {prediction_summary.get('median_confidence')}",
            f"- mean_top1_top2_gap: {prediction_summary.get('mean_top1_top2_gap')}",
            f"- mean_entropy: {prediction_summary.get('mean_entropy')}",
            f"- low_confidence_rows: {prediction_summary.get('low_confidence_rows')}",
            "",
            "## Top Confusion Pairs",
            "",
            "| true | predicted | count | rate_of_true |",
            "|---|---|---:|---:|",
        ]
    )
    for item in confusion_analysis.get("top_confusion_pairs", [])[:12]:
        lines.append(f"| {item['true']} | {item['predicted']} | {item['count']} | {item['rate_of_true']:.3f} |")
    lines.extend(["", "## Watched Confusion Patterns", "", "| true | predicted | count | rate_of_true |", "|---|---|---:|---:|"])
    for item in confusion_analysis.get("watched_confusion_patterns", [])[:12]:
        lines.append(f"| {item['true']} | {item['predicted']} | {item['count']} | {item['rate_of_true']:.3f} |")
    lines.extend(["", "## Warnings", ""])
    if warnings:
        for item in warnings:
            lines.append(f"- [{item['severity']}] {item['code']}: {item['message']}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def render_distribution_table(summary: dict[str, Any], title: str) -> str:
    lines = [
        f"### {title}",
        "",
        f"- rows: {summary['rows']}",
        f"- real_rows: {summary['real_rows']}",
        f"- synthetic_rows: {summary['synthetic_rows']}",
        f"- ignore_metadata_rows: {summary['ignore_metadata_rows']}",
        "",
        "| class | count | synthetic |",
        "|---|---:|---:|",
    ]
    for class_name, count in summary["class_counts"].items():
        lines.append(f"| {class_name} | {count} | {summary['synthetic_class_counts'].get(class_name, 0)} |")
    return "\n".join(lines)


def render_kfold_report(fold_metrics: list[dict[str, Any]], diagnostics: list[dict[str, Any]]) -> str:
    lines = ["# MILK10k K-Fold Report", ""]
    if fold_metrics:
        df = pd.DataFrame(fold_metrics)
        metric_cols = [
            col
            for col in ("accuracy", "balanced_accuracy", "dice_macro", "f1_macro", "roc_auc_macro_ovr", "top3_accuracy")
            if col in df.columns
        ]
        lines.extend(["## Fold Metrics", "", dataframe_to_markdown(df[["fold", *metric_cols]]), ""])
        rows = []
        for col in metric_cols:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            rows.append({"metric": col, "mean": values.mean() if len(values) else None, "std": values.std(ddof=0) if len(values) else None})
        lines.extend(["## Aggregate", "", dataframe_to_markdown(pd.DataFrame(rows)), ""])
    lines.extend(["## Fold Warnings", ""])
    any_warning = False
    for payload in diagnostics:
        fold = payload.get("fold")
        for item in payload.get("warnings", []):
            any_warning = True
            lines.append(f"- fold={fold} [{item['severity']}] {item['code']}: {item['message']}")
    if not any_warning:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = [format_markdown_value(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
