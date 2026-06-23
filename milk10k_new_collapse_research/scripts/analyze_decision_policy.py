"""Optimize rare-class logit bias on an existing validation prediction CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from milk10k_new_collapse_research.compat import ensure_legacy_package_path

ensure_legacy_package_path()
from milk10k_effb2_metadata.metrics import apply_class_bias, compute_metrics, optimize_class_bias
from milk10k_new_collapse_research.config import RESULTS_ROOT, TAIL_GATE_CLASSES
from milk10k_new_collapse_research.metrics_ext import build_collapse_report, json_safe, load_prediction_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "decision_policy")
    parser.add_argument("--metric", choices=["f1_macro", "dice_macro"], default="f1_macro")
    parser.add_argument("--max-bias", type=float, default=2.0)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--tail-only", action="store_true", help="Optimize only BEN_OTH/INF/MAL_OTH biases.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df, y_true, y_prob, class_names = load_prediction_csv(args.predictions)
    base_metrics, _, base_cm = compute_metrics(y_true, y_prob, class_names)
    if args.tail_only:
        bias, best_score = optimize_tail_bias(y_true, y_prob, class_names, args)
    else:
        bias, best_score = optimize_class_bias(
            y_true,
            y_prob,
            class_names,
            metric_name=args.metric,
            max_bias=args.max_bias,
            step=args.step,
            passes=args.passes,
        )
    adjusted_prob = apply_class_bias(y_prob, bias)
    adjusted_metrics, per_class_df, adjusted_cm = compute_metrics(y_true, adjusted_prob, class_names)
    out_df = df.copy()
    pred = adjusted_prob.argmax(axis=1)
    out_df["policy_y_pred"] = pred
    out_df["policy_label_pred"] = [class_names[idx] for idx in pred]
    for idx, label in enumerate(class_names):
        out_df[f"policy_prob_{label}"] = adjusted_prob[:, idx]
    out_df.to_csv(args.output_dir / "val_predictions.csv", index=False)
    per_class_df.to_csv(args.output_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(adjusted_cm, index=class_names, columns=class_names).to_csv(args.output_dir / "confusion_matrix.csv")
    payload = {
        "experiment": "decision_policy",
        "source_predictions": str(args.predictions),
        "metric": args.metric,
        "tail_only": args.tail_only,
        "base_metric": base_metrics[args.metric],
        "best_search_metric": best_score,
        "bias": {label: float(bias[idx]) for idx, label in enumerate(class_names)},
        "adjusted": adjusted_metrics,
        "base": base_metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    (args.output_dir / "collapse_report.md").write_text(
        "# Base Collapse Report\n\n"
        + build_collapse_report(base_metrics, base_cm, class_names)
        + "\n# Adjusted Collapse Report\n\n"
        + build_collapse_report(adjusted_metrics, adjusted_cm, class_names),
        encoding="utf-8",
    )


def optimize_tail_bias(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str], args: argparse.Namespace):
    bias = np.zeros(len(class_names), dtype=np.float32)
    tail_indices = [class_names.index(label) for label in TAIL_GATE_CLASSES if label in class_names]
    best_score = float(compute_metrics(y_true, y_prob, class_names)[0][args.metric])
    deltas = np.arange(-args.max_bias, args.max_bias + args.step * 0.5, args.step, dtype=np.float32)
    for _ in range(max(1, args.passes)):
        improved = False
        for idx in tail_indices:
            current = float(bias[idx])
            best_local = current
            for delta in deltas:
                trial = bias.copy()
                trial[idx] = current + float(delta)
                score = float(compute_metrics(y_true, apply_class_bias(y_prob, trial), class_names)[0][args.metric])
                if score > best_score + 1e-12:
                    best_score = score
                    best_local = float(trial[idx])
                    improved = True
            bias[idx] = best_local
        if not improved:
            break
    return bias, best_score


if __name__ == "__main__":
    main()
