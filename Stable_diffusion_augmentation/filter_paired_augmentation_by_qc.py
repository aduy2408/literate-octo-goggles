#!/usr/bin/env python3
"""Filter paired augmentation manifest using EffB2 QC predictions."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter paired augmentation manifest by QC target probability/prediction.")
    parser.add_argument("--manifest", type=Path, required=True, help="Input paired_augmentation_manifest.csv.")
    parser.add_argument("--qc-summary", type=Path, required=True, help="effb2_qc_summary.csv.")
    parser.add_argument("--output", type=Path, required=True, help="Filtered paired augmentation manifest.")
    parser.add_argument("--min-target-prob", type=float, default=0.4)
    parser.add_argument("--require-target-pred", action="store_true", help="Keep only rows where label_pred == target_class.")
    parser.add_argument(
        "--exclude-source-lesions",
        nargs="*",
        default=[],
        help="Source lesion IDs to drop completely, e.g. IL_4939300 IL_8173942.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def print_counts(title: str, rows: list[dict[str, str]], class_key: str = "class_name") -> None:
    counts = Counter(row[class_key] for row in rows)
    print(title)
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")
    print(f"  total: {len(rows)}")


def print_source_breakdown(rows: list[dict[str, str]], qc_by_id: dict[str, dict[str, str]]) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("source_lesion_id", "")].append(row)

    print("Kept by source lesion")
    for source_lesion_id, source_rows in sorted(grouped.items()):
        predicted = sum(1 for row in source_rows if qc_by_id[row["synthetic_lesion_id"]].get("is_target_predicted") == "True")
        avg_target_prob = sum(as_float(qc_by_id[row["synthetic_lesion_id"]].get("target_class_probability", "")) for row in source_rows) / len(source_rows)
        print(f"  {source_lesion_id}: kept={len(source_rows)}, target_predicted={predicted}, avg_target_prob={avg_target_prob:.4f}")


def main() -> None:
    args = parse_args()
    manifest_rows = read_rows(args.manifest.expanduser().resolve())
    qc_rows = read_rows(args.qc_summary.expanduser().resolve())
    qc_by_id = {row["synthetic_lesion_id"]: row for row in qc_rows}
    excluded = set(args.exclude_source_lesions)

    kept = []
    for row in manifest_rows:
        synthetic_id = row["synthetic_lesion_id"]
        qc = qc_by_id.get(synthetic_id)
        if qc is None:
            continue
        if row.get("source_lesion_id", "") in excluded:
            continue
        if args.require_target_pred and qc.get("is_target_predicted") != "True":
            continue
        if as_float(qc.get("target_class_probability", "")) < args.min_target_prob:
            continue
        kept.append(row)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if manifest_rows:
        with output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(kept)

    print_counts("Original manifest counts", manifest_rows)
    print_counts("Filtered manifest counts", kept)
    print_source_breakdown(kept, qc_by_id)
    print(f"Saved filtered manifest: {output}")


if __name__ == "__main__":
    main()
