#!/usr/bin/env python3
"""Run EffB2 prediction QC for paired diffusion augmentation and print confidence."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EffB2 QC prediction and print confidence summary.")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=Path, help="Path to one classifier best.pt checkpoint.")
    checkpoint_group.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Run directory containing fold_*/best.pt; all folds are ensembled for QC.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("Stable_diffusion_augmentation/out_minority_pairs"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--python", default=sys.executable, help="Python executable to use for prediction.")
    parser.add_argument(
        "--predict-script",
        type=Path,
        default=None,
        help="Path to predict_milk10k_effb2_dual_metadata.py. Defaults to auto-detect from repo root.",
    )
    parser.add_argument(
        "--summary-script",
        type=Path,
        default=None,
        help="Path to summarize_effb2_qc.py. Defaults to this script's folder.",
    )
    parser.add_argument("--print-misses", type=int, default=20, help="Number of wrong-target rows to print.")
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("Running:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_script(path: Path | None, default_path: Path, label: str) -> Path:
    script = (path or default_path).expanduser().resolve()
    if not script.exists():
        raise FileNotFoundError(f"{label} not found: {script}")
    return script


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def print_confidence_summary(summary_path: Path, print_misses: int) -> None:
    rows = read_rows(summary_path)
    if not rows:
        print("No QC rows found.")
        return

    correct = [row for row in rows if row["is_target_predicted"] == "True"]
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_class[row["target_class"]].append(row)

    print("")
    print("EffB2 QC confidence summary")
    print(f"  Total synthetic pairs: {len(rows)}")
    print(f"  Target predicted: {len(correct)}/{len(rows)} ({len(correct) / len(rows):.1%})")

    for class_name in sorted(by_class):
        class_rows = by_class[class_name]
        class_correct = [row for row in class_rows if row["is_target_predicted"] == "True"]
        avg_conf = sum(as_float(row["confidence"]) for row in class_rows) / len(class_rows)
        avg_target_prob = sum(as_float(row["target_class_probability"]) for row in class_rows) / len(class_rows)
        pred_counts: dict[str, int] = defaultdict(int)
        for row in class_rows:
            pred_counts[row["label_pred"]] += 1
        top_preds = ", ".join(f"{label}:{count}" for label, count in sorted(pred_counts.items(), key=lambda item: (-item[1], item[0]))[:5])
        print(
            f"  {class_name}: target_predicted={len(class_correct)}/{len(class_rows)} "
            f"({len(class_correct) / len(class_rows):.1%}), "
            f"avg_confidence={avg_conf:.4f}, avg_target_prob={avg_target_prob:.4f}, "
            f"top_preds=[{top_preds}]"
        )

    misses = [row for row in rows if row["is_target_predicted"] != "True"]
    misses.sort(key=lambda row: as_float(row["target_class_probability"]))
    if misses and print_misses > 0:
        print("")
        print(f"Lowest target-probability misses, first {min(print_misses, len(misses))}:")
        for row in misses[:print_misses]:
            print(
                f"  {row['synthetic_lesion_id']}: target={row['target_class']} "
                f"pred={row['label_pred']} conf={as_float(row['confidence']):.4f} "
                f"target_prob={as_float(row['target_class_probability']):.4f}"
            )

    if any(row.get("source_lesion_id") for row in rows):
        print("")
        print("Worst source lesions by target probability:")
        by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_source[row.get("source_lesion_id", "")].append(row)
        source_stats = []
        for source_lesion_id, source_rows in by_source.items():
            avg_target_prob = sum(as_float(row["target_class_probability"]) for row in source_rows) / len(source_rows)
            target_predicted = sum(1 for row in source_rows if row["is_target_predicted"] == "True")
            source_stats.append((avg_target_prob, source_lesion_id, target_predicted, len(source_rows)))
        for avg_target_prob, source_lesion_id, target_predicted, total in sorted(source_stats)[:10]:
            print(
                f"  {source_lesion_id}: target_predicted={target_predicted}/{total} "
                f"({target_predicted / total:.1%}), avg_target_prob={avg_target_prob:.4f}"
            )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve() if args.checkpoint_dir else None
    manifest = output_dir / "paired_augmentation_manifest.csv"
    metadata_csv = output_dir / "metadata_for_prediction.csv"
    groundtruth_csv = output_dir / "groundtruth_for_prediction.csv"
    input_dir = output_dir / "prediction_input"
    predictions = output_dir / "effb2_qc_predictions.csv"
    summary = output_dir / "effb2_qc_summary.csv"
    repo_root = default_repo_root()
    predict_script = resolve_script(args.predict_script, repo_root / "predict_milk10k_effb2_dual_metadata.py", "Predict script")
    summary_script = resolve_script(
        args.summary_script,
        Path(__file__).resolve().parent / "summarize_effb2_qc.py",
        "Summary script",
    )

    for path in (checkpoint or checkpoint_dir, manifest, metadata_csv, groundtruth_csv, input_dir):
        if not path.exists():
            raise FileNotFoundError(f"Required QC input not found: {path}")

    predict_command = [
            args.python,
            str(predict_script),
    ]
    if checkpoint is not None:
        predict_command.extend(["--checkpoint", str(checkpoint)])
    else:
        predict_command.extend(["--checkpoint-dir", str(checkpoint_dir)])
    predict_command.extend([
            "--input-dir",
            str(input_dir),
            "--metadata-csv",
            str(metadata_csv),
            "--groundtruth-csv",
            str(groundtruth_csv),
            "--output",
            str(predictions),
            "--include-debug-columns",
            "--batch-size",
            str(args.batch_size),
            "--image-size",
            str(args.image_size),
            "--num-workers",
            str(args.num_workers),
        ])
    run_command(predict_command)
    run_command(
        [
            args.python,
            str(summary_script),
            "--manifest",
            str(manifest),
            "--predictions",
            str(predictions),
            "--output",
            str(summary),
        ]
    )
    print_confidence_summary(summary, args.print_misses)
    print("")
    print(f"Predictions CSV: {predictions}")
    print(f"QC summary CSV: {summary}")


if __name__ == "__main__":
    main()
