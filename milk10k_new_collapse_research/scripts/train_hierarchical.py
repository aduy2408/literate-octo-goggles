"""Train a two-stage hierarchy on frozen paired features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from milk10k_new_collapse_research.config import CLASS_NAMES, RESULTS_ROOT
from milk10k_new_collapse_research.features import make_feature_matrix
from milk10k_new_collapse_research.hierarchy import combine_group_and_expert_probabilities, default_hierarchy
from milk10k_new_collapse_research.metrics_ext import write_standard_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--val-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "hierarchical")
    parser.add_argument("--feature-mode", choices=["pair", "clinical", "dermoscopic"], default="pair")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = dict(np.load(args.train_features, allow_pickle=True))
    val = dict(np.load(args.val_features, allow_pickle=True))
    class_names = infer_class_names(train, val)
    spec = default_hierarchy(class_names)
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    x_train = make_feature_matrix(train, args.feature_mode)
    x_val = make_feature_matrix(val, args.feature_mode)
    train_labels = [str(label) for label in train["label"]]
    val_labels = [str(label) for label in val["label"]]
    y_train = np.asarray([label_to_idx[label] for label in train_labels], dtype=np.int64)
    y_val = np.asarray([label_to_idx[label] for label in val_labels], dtype=np.int64)
    y_train_group = spec.group_labels_for_class_labels(train_labels)

    group_model = classifier(args.max_iter, args.seed)
    group_model.fit(x_train, y_train_group)
    group_prob = align_probabilities(group_model.predict_proba(x_val), group_model.classes_, len(spec.group_names))

    expert_prob_by_group: dict[str, np.ndarray] = {}
    for group_name, labels in spec.groups.items():
        local_label_to_idx = {label: idx for idx, label in enumerate(labels)}
        mask = np.asarray([label in local_label_to_idx for label in train_labels], dtype=bool)
        if len(labels) == 1:
            expert_prob_by_group[group_name] = np.ones((x_val.shape[0], 1), dtype=np.float64)
            continue
        expert = classifier(args.max_iter, args.seed)
        expert.fit(x_train[mask], np.asarray([local_label_to_idx[label] for label in np.asarray(train_labels)[mask]]))
        expert_prob_by_group[group_name] = align_probabilities(expert.predict_proba(x_val), expert.classes_, len(labels))

    y_prob = combine_group_and_expert_probabilities(group_prob, expert_prob_by_group, spec)
    val_df = pd.DataFrame(
        {
            "lesion_id": [str(item) for item in val["lesion_id"]],
            "clinical_path": [""] * len(val_labels),
            "dermoscopic_path": [""] * len(val_labels),
        }
    )
    metrics = write_standard_outputs(
        args.output_dir,
        val_df,
        y_val,
        y_prob,
        class_names,
        extra={"experiment": "hierarchical_probe", "feature_mode": args.feature_mode, "groups": spec.groups},
    )
    (args.output_dir / "hierarchy_config.json").write_text(
        json.dumps({"groups": spec.groups, "metrics_f1_macro": metrics["f1_macro"]}, indent=2),
        encoding="utf-8",
    )


def classifier(max_iter: int, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        ),
    )


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


if __name__ == "__main__":
    main()

