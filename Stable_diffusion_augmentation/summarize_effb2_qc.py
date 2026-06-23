#!/usr/bin/env python3
"""Summarize EffB2 predictions for paired diffusion augmentation QC."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create QC summary from paired manifest and EffB2 debug predictions.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("Stable_diffusion_augmentation/out_minority_pairs/paired_augmentation_manifest.csv"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("Stable_diffusion_augmentation/out_minority_pairs/effb2_qc_predictions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Stable_diffusion_augmentation/out_minority_pairs/effb2_qc_summary.csv"),
    )
    return parser.parse_args()


def read_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="") as f:
        return {row[key]: row for row in csv.DictReader(f)}


def probability_for(row: dict[str, str], class_name: str) -> float:
    for key in (class_name, f"prob_{class_name}"):
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    return 0.0


def main() -> None:
    args = parse_args()
    manifest = read_by_key(args.manifest.expanduser().resolve(), "synthetic_lesion_id")
    predictions = read_by_key(args.predictions.expanduser().resolve(), "lesion_id")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "synthetic_lesion_id",
        "source_lesion_id",
        "target_class",
        "label_pred",
        "confidence",
        "target_class_probability",
        "is_target_predicted",
        "clinical_generated_path",
        "dermoscopic_generated_path",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for lesion_id, manifest_row in sorted(manifest.items()):
            pred_row = predictions.get(lesion_id, {})
            target_class = manifest_row["class_name"]
            label_pred = pred_row.get("label_pred", "")
            target_prob = probability_for(pred_row, target_class) if pred_row else 0.0
            writer.writerow(
                {
                    "synthetic_lesion_id": lesion_id,
                    "source_lesion_id": manifest_row.get("source_lesion_id", ""),
                    "target_class": target_class,
                    "label_pred": label_pred,
                    "confidence": pred_row.get("confidence", ""),
                    "target_class_probability": target_prob,
                    "is_target_predicted": str(label_pred == target_class),
                    "clinical_generated_path": manifest_row["clinical_generated_path"],
                    "dermoscopic_generated_path": manifest_row["dermoscopic_generated_path"],
                }
            )

    print(f"Saved QC summary: {output}")


if __name__ == "__main__":
    main()
