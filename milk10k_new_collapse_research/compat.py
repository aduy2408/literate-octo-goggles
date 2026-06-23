"""Compatibility helpers for reusing the existing local MILK10k package."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_legacy_package_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    legacy_root = repo_root / "milk10k_effb2_metadata"
    legacy_path = str(legacy_root)
    if legacy_root.is_dir() and legacy_path not in sys.path:
        sys.path.insert(0, legacy_path)

