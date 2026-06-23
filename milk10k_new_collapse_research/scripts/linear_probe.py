"""Train a linear or small MLP probe on frozen paired features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from milk10k_new_collapse_research.config import CLASS_NAMES, RESULTS_ROOT
from milk10k_new_collapse_research.features import make_feature_matrix
from milk10k_new_collapse_research.metrics_ext import write_standard_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "linear_probe")
    parser.add_argument("--feature-mode", choices=["pair", "clinical", "dermoscopic"], default="pair")
    parser.add_argument("--probe", choices=["logistic", "mlp"], default="logistic")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = dict(np.load(args.train_features, allow_pickle=True))
    val = dict(np.load(args.val_features, allow_pickle=True))
    class_names = infer_class_names(train, val)
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    x_train = make_feature_matrix(train, args.feature_mode)
    x_val = make_feature_matrix(val, args.feature_mode)
    y_train = np.asarray([label_to_idx[str(label)] for label in train["label"]], dtype=np.int64)
    y_val = np.asarray([label_to_idx[str(label)] for label in val["label"]], dtype=np.int64)

    if args.probe == "logistic":
        clf = LogisticRegression(
            max_iter=args.max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=args.seed,
        )
    elif args.probe == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
        )
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(512,),
            alpha=1e-4,
            batch_size=64,
            max_iter=args.max_iter,
            early_stopping=True,
            random_state=args.seed,
        )
    model = make_pipeline(StandardScaler(), clf)
    model.fit(x_train, y_train)
    y_prob = model.predict_proba(x_val)
    y_prob = align_probabilities(y_prob, model.classes_, len(class_names))
    val_df = prediction_frame(val)
    metrics = write_standard_outputs(
        args.output_dir,
        val_df,
        y_val,
        y_prob,
        class_names,
        extra={
            "experiment": "linear_probe",
            "feature_mode": args.feature_mode,
            "probe": args.probe,
            "train_features": str(args.train_features),
            "val_features": str(args.val_features),
        },
    )
    (args.output_dir / "probe_config.json").write_text(json.dumps({"metrics_f1_macro": metrics["f1_macro"]}, indent=2))


def infer_class_names(*arrays: dict[str, np.ndarray]) -> list[str]:
    observed = sorted({str(label) for data in arrays for label in data["label"].tolist()})
    preferred = [label for label in CLASS_NAMES if label in observed]
    extras = [label for label in observed if label not in preferred]
    return preferred + extras


def align_probabilities(y_prob: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    aligned = np.zeros((y_prob.shape[0], n_classes), dtype=np.float64)
    for src_idx, class_idx in enumerate(classes):
        aligned[:, int(class_idx)] = y_prob[:, src_idx]
    row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / np.clip(row_sum, 1e-12, None)


def prediction_frame(data: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lesion_id": [str(item) for item in data["lesion_id"]],
            "clinical_path": [""] * len(data["label"]),
            "dermoscopic_path": [""] * len(data["label"]),
        }
    )


if __name__ == "__main__":
    main()

