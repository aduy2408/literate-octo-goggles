#!/usr/bin/env python3
"""Merge paired augmentation manifests for training."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge paired augmentation manifests into one training manifest.")
    parser.add_argument("--manifest", type=Path, action="append", required=True, help="Input manifest. Pass multiple times.")
    parser.add_argument("--output", type=Path, required=True, help="Merged output manifest.")
    parser.add_argument(
        "--replace-class",
        action="append",
        default=[],
        help="For this class, keep rows only from the last input manifest containing that class. Can be repeated.",
    )
    parser.add_argument("--dedupe", action="store_true", help="Drop duplicate synthetic_lesion_id rows, keeping later inputs.")
    return parser.parse_args()


def read_manifest(path: Path, input_index: int) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["_input_index"] = str(input_index)
        row["_input_manifest"] = str(path)
    return rows


def class_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["class_name"] for row in rows).items()))


def main() -> None:
    args = parse_args()
    manifests = [path.expanduser().resolve() for path in args.manifest]
    for path in manifests:
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

    rows_by_input = [read_manifest(path, idx) for idx, path in enumerate(manifests)]
    rows = [row for group in rows_by_input for row in group]
    if not rows:
        raise ValueError("No rows found in input manifests.")

    replace_classes = set(args.replace_class)
    if replace_classes:
        last_input_for_class: dict[str, int] = {}
        for row in rows:
            class_name = row["class_name"]
            if class_name in replace_classes:
                last_input_for_class[class_name] = int(row["_input_index"])
        rows = [
            row
            for row in rows
            if row["class_name"] not in replace_classes
            or int(row["_input_index"]) == last_input_for_class.get(row["class_name"])
        ]

    if args.dedupe:
        by_id = {row["synthetic_lesion_id"]: row for row in rows}
        rows = list(by_id.values())

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [key for key in rows[0].keys() if not key.startswith("_")]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print("Merged paired augmentation manifests")
    for path, group in zip(manifests, rows_by_input):
        print(f"  input: {path}")
        print(f"    rows={len(group)}, classes={class_counts(group)}")
    print(f"  output: {output}")
    print(f"    rows={len(rows)}, classes={class_counts(rows)}")


if __name__ == "__main__":
    main()
