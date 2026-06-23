"""Shared defaults for experimental MILK10k collapse research."""

from __future__ import annotations

from pathlib import Path


CLASS_NAMES = [
    "AKIEC",
    "BCC",
    "BEN_OTH",
    "BKL",
    "DF",
    "INF",
    "MAL_OTH",
    "MEL",
    "NV",
    "SCCKA",
    "VASC",
]

DEFAULT_HIERARCHY = {
    "melanocytic": ["MEL", "NV"],
    "keratinocyte_like": ["AKIEC", "BCC", "BKL", "SCCKA"],
    "rare_other": ["BEN_OTH", "INF", "MAL_OTH", "DF", "VASC"],
}

TAIL_GATE_CLASSES = ["BEN_OTH", "INF", "MAL_OTH"]
RESULTS_ROOT = Path("results/new_collapse_research")
BASELINE_F1_MACRO = 0.5828


def class_to_idx(class_names: list[str] | None = None) -> dict[str, int]:
    names = class_names or CLASS_NAMES
    return {name: idx for idx, name in enumerate(names)}

