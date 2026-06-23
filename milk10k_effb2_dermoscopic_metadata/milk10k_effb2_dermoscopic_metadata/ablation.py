"""Run and summarize matched no-metadata vs metadata training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .training import parse_args as parse_training_args, run as run_training


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run matched dermoscopic metadata ablation.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("training_args", nargs=argparse.REMAINDER, help="Extra training arguments after --, e.g. -- --amp --loss ce_dice")
    return parser.parse_args(argv)


def summarize(output_dir: Path):
    runs = {"none": output_dir / "no_metadata", "concat": output_dir / "with_metadata"}
    rows = []
    payload = {"runs": {}, "delta_with_minus_without": {}}
    keys = ["f1_macro", "balanced_accuracy", "accuracy", "roc_auc_macro_ovr", "f1_weighted"]
    for mode, directory in runs.items():
        if (directory / "metrics.json").exists():
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        else:
            kfold = json.loads((directory / "kfold_summary.json").read_text(encoding="utf-8"))
            metrics = kfold["mean"]
        payload["runs"][mode] = metrics
        rows.append({"metadata_mode": mode, **{key: metrics.get(key) for key in keys}})
    for key in keys:
        left = payload["runs"]["none"].get(key); right = payload["runs"]["concat"].get(key)
        payload["delta_with_minus_without"][key] = None if left is None or right is None else right - left
    class_rows = []
    def per_class(directory):
        direct=directory/"per_class_metrics.csv"
        paths=[direct] if direct.exists() else sorted(directory.glob("fold_*/per_class_metrics.csv"))
        frames=[pd.read_csv(path) for path in paths]
        if not frames:raise FileNotFoundError(f"No per-class metrics under {directory}")
        return pd.concat(frames).groupby("class",as_index=True).mean(numeric_only=True)
    none_pc = per_class(runs["none"]); with_pc = per_class(runs["concat"])
    for name in none_pc.index:
        class_rows.append({"class": name, "f1_no_metadata": none_pc.loc[name, "f1"], "f1_with_metadata": with_pc.loc[name, "f1"],
                           "f1_delta": with_pc.loc[name, "f1"] - none_pc.loc[name, "f1"],
                           "recall_delta": with_pc.loc[name, "recall_sensitivity"] - none_pc.loc[name, "recall_sensitivity"]})
    pd.DataFrame(rows).to_csv(output_dir / "ablation_summary.csv", index=False)
    pd.DataFrame(class_rows).to_csv(output_dir / "ablation_per_class.csv", index=False)
    (output_dir / "ablation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(args):
    output_dir = args.output_dir.expanduser().resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    extras = args.training_args[1:] if args.training_args[:1] == ["--"] else args.training_args
    forbidden = {"--data-dir", "--input-dir", "--output-dir", "--split-manifest", "--metadata-mode", "--calibrate-bias"}
    if forbidden.intersection(extras):
        raise ValueError(f"Do not override ablation-controlled arguments: {sorted(forbidden.intersection(extras))}")
    for mode, name in (("none", "no_metadata"), ("concat", "with_metadata")):
        cli = ["--data-dir", str(args.data_dir), "--output-dir", str(output_dir / name), "--split-manifest", str(args.split_manifest), "--metadata-mode", mode]
        if args.input_dir is not None:
            cli.extend(["--input-dir", str(args.input_dir)])
        cli.extend(extras)
        run_training(parse_training_args(cli))
    summarize(output_dir)
    print(f"Ablation summary saved under {output_dir}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
