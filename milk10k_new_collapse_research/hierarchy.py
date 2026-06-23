"""Hierarchical class mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from milk10k_new_collapse_research.config import CLASS_NAMES, DEFAULT_HIERARCHY


@dataclass(frozen=True)
class HierarchySpec:
    groups: dict[str, list[str]]
    class_names: list[str]

    @property
    def group_names(self) -> list[str]:
        return list(self.groups)

    @property
    def class_to_group(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for group, labels in self.groups.items():
            for label in labels:
                if label in mapping:
                    raise ValueError(f"Class {label!r} appears in multiple hierarchy groups.")
                mapping[label] = group
        missing = sorted(set(self.class_names) - set(mapping))
        extra = sorted(set(mapping) - set(self.class_names))
        if missing:
            raise ValueError(f"Hierarchy missing classes: {missing}")
        if extra:
            raise ValueError(f"Hierarchy references unknown classes: {extra}")
        return mapping

    def group_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.group_names)}

    def group_labels_for_class_labels(self, labels: list[str]) -> np.ndarray:
        c2g = self.class_to_group
        g2i = self.group_index()
        return np.asarray([g2i[c2g[label]] for label in labels], dtype=np.int64)

    def class_group_indices(self) -> list[int]:
        c2g = self.class_to_group
        g2i = self.group_index()
        return [g2i[c2g[label]] for label in self.class_names]


def default_hierarchy(class_names: list[str] | None = None) -> HierarchySpec:
    names = class_names or CLASS_NAMES
    name_set = set(names)
    groups = {
        group: [label for label in labels if label in name_set]
        for group, labels in DEFAULT_HIERARCHY.items()
    }
    groups = {group: labels for group, labels in groups.items() if labels}
    return HierarchySpec(groups=groups, class_names=names)


def combine_group_and_expert_probabilities(
    group_prob: np.ndarray,
    expert_prob_by_group: dict[str, np.ndarray],
    spec: HierarchySpec,
) -> np.ndarray:
    """Return flat class probabilities from group and in-group probabilities."""
    flat = np.zeros((group_prob.shape[0], len(spec.class_names)), dtype=np.float64)
    group_to_idx = spec.group_index()
    class_to_idx = {label: idx for idx, label in enumerate(spec.class_names)}
    for group_name, labels in spec.groups.items():
        expert_prob = expert_prob_by_group[group_name]
        if expert_prob.shape[1] != len(labels):
            raise ValueError(f"Expert {group_name!r} has {expert_prob.shape[1]} columns, expected {len(labels)}.")
        for local_idx, label in enumerate(labels):
            flat[:, class_to_idx[label]] = group_prob[:, group_to_idx[group_name]] * expert_prob[:, local_idx]
    row_sum = flat.sum(axis=1, keepdims=True)
    return flat / np.clip(row_sum, 1e-12, None)
