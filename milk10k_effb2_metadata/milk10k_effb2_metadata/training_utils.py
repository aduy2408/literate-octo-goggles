"""Shared training serialization and reporting helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - keeps json_safe importable in minimal CLI environments.
    np = None


def save_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    class_names: list[str],
    label_to_idx: dict[str, int],
    metadata_spec: dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    clinical_backbone_backend: str,
    dermoscopic_backbone_backend: str,
    fold: int | None = None,
) -> None:
    import pandas as pd

    from milk10k_effb2_metadata.reporting import collect_environment_info

    payload = {
        "args": json_safe(vars(args)),
        "environment": collect_environment_info(),
        "class_names": class_names,
        "label_to_idx": label_to_idx,
        "metadata_spec": json_safe(metadata_spec),
        "model_type": "DualConvNeXtMetadataClassifier" if args.backbone == "convnext_base" else "DualEffB2MetadataClassifier",
        "train_size": len(train_df),
        "val_size": len(val_df),
        "fold": fold,
        "metadata_fusion": args.metadata_fusion,
        "image_fusion": getattr(args, "image_fusion", "concat"),
        "clinical_backbone": f"{clinical_backbone_backend} {args.backbone}",
        "dermoscopic_backbone": f"{dermoscopic_backbone_backend} {args.backbone}",
        "paths": {
            "output_dir": str(output_dir),
            "data_dir": str(getattr(args, "data_dir", "")),
            "dermoscopic_mask_dir": str(getattr(args, "dermoscopic_mask_dir", "")),
            "clinical_checkpoint": str(getattr(args, "clinical_checkpoint", "")),
            "dermoscopic_checkpoint": str(getattr(args, "dermoscopic_checkpoint", "")),
            "resume_checkpoint": str(getattr(args, "resume_checkpoint", "")),
            "augmented_data_dir": str(getattr(args, "augmented_data_dir", "")),
        },
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_kfold_summary(fold_metrics: list[dict[str, Any]], output_dir: Path) -> None:
    import pandas as pd

    summary_keys = [
        "best_val_f1_macro",
        "best_selection_metric",
        "best_val_tail_recall_macro",
        "accuracy",
        "balanced_accuracy",
        "dice_macro",
        "f1_macro",
        "roc_auc_macro_ovr",
        "top3_accuracy",
    ]
    rows = []
    for metrics in fold_metrics:
        rows.append({key: metrics.get(key) for key in ["fold", *summary_keys]})
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_dir / "kfold_summary.csv", index=False)

    aggregate: dict[str, Any] = {"folds": json_safe(rows), "mean": {}, "std": {}}
    for key in summary_keys:
        values = pd.to_numeric(summary_df[key], errors="coerce").dropna()
        aggregate["mean"][key] = None if values.empty else float(values.mean())
        aggregate["std"][key] = None if values.empty else float(values.std(ddof=0))
    with open(output_dir / "kfold_summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value
