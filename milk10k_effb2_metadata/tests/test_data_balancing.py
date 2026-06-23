from __future__ import annotations

import argparse
import tempfile
from collections import Counter
from pathlib import Path
import unittest
from unittest.mock import patch

MISSING_DEPENDENCY: str | None = None

try:
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from milk10k_effb2_metadata.data import (
        HybridEpochSampler,
        PairedMilk10kMetadataDataset,
        hybrid_target_counts,
    )
    from milk10k_effb2_metadata.training import validate_balance_args
    from milk10k_effb2_metadata.runner import append_augmented_train_rows
except ModuleNotFoundError as exc:  # pragma: no cover - local minimal env may omit ML deps.
    MISSING_DEPENDENCY = exc.name


def balance_args(**overrides):
    values = {
        "balance_mode": "hybrid",
        "weighted_sampler": False,
        "balance_head_ratio": 2.0,
        "balance_tail_floor": 100,
        "balance_min_source_count": 20,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@unittest.skipIf(MISSING_DEPENDENCY is not None, f"Missing ML test dependency: {MISSING_DEPENDENCY}")
class HybridBalanceTest(unittest.TestCase):
    def setUp(self) -> None:
        # BCC=250, NV=100, INF=40, MAL_OTH=9.
        self.labels = [0] * 250 + [1] * 100 + [2] * 40 + [3] * 9
        self.targets, self.strong_labels = hybrid_target_counts(self.labels, balance_args())

    def test_targets_cap_head_oversample_tail_and_leave_ultra_rare_alone(self) -> None:
        self.assertEqual(self.targets.tolist(), [200, 100, 100, 9])
        self.assertEqual(self.strong_labels, {2})

    def test_sampler_has_expected_counts_and_only_tail_duplicates(self) -> None:
        sampler = HybridEpochSampler(self.labels, self.targets, seed=42)
        indices = list(sampler)
        sampled_labels = Counter(self.labels[index] for index in indices)
        self.assertEqual(sampled_labels, Counter({0: 200, 1: 100, 2: 100, 3: 9}))

        bcc_indices = [index for index in indices if self.labels[index] == 0]
        inf_indices = [index for index in indices if self.labels[index] == 2]
        self.assertEqual(len(bcc_indices), len(set(bcc_indices)))
        self.assertLess(len(set(inf_indices)), len(inf_indices))

    def test_sampler_is_reproducible_and_changes_head_subset_by_epoch(self) -> None:
        first = HybridEpochSampler(self.labels, self.targets, seed=7)
        second = HybridEpochSampler(self.labels, self.targets, seed=7)
        first.set_epoch(3)
        second.set_epoch(3)
        self.assertEqual(list(first), list(second))

        epoch_three = set(index for index in first if self.labels[index] == 0)
        first.set_epoch(4)
        epoch_four = set(index for index in first if self.labels[index] == 0)
        self.assertNotEqual(epoch_three, epoch_four)

    def test_dataset_routes_only_tail_to_strong_transform(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "image.png"
            Image.new("RGB", (4, 4), color=(10, 20, 30)).save(image_path)
            rows = []
            for label in ("BCC", "INF"):
                rows.append(
                    {
                        "lesion_id": label,
                        "label": label,
                        "clinical_path": str(image_path),
                        "dermoscopic_path": str(image_path),
                        "clinical_age_approx": 50,
                        "dermoscopic_age_approx": 50,
                        "clinical_skin_tone_class": 2,
                        "dermoscopic_skin_tone_class": 2,
                        "clinical_sex": "unknown",
                        "dermoscopic_sex": "unknown",
                        "clinical_site": "unknown",
                        "dermoscopic_site": "unknown",
                    }
                )
            spec = {"sex_values": ["unknown"], "site_values": ["unknown"], "monet_columns": []}
            regular = lambda image: torch.zeros(3, image.height, image.width)
            strong = lambda image: torch.ones(3, image.height, image.width)
            dataset = PairedMilk10kMetadataDataset(
                pd.DataFrame(rows),
                {"BCC": 0, "INF": 1},
                spec,
                regular,
                strong_transform=strong,
                strong_augment_labels={1},
            )
            self.assertTrue(torch.equal(dataset[0]["clinical"], torch.zeros(3, 4, 4)))
            self.assertTrue(torch.equal(dataset[1]["clinical"], torch.ones(3, 4, 4)))
            self.assertTrue(torch.equal(dataset[1]["dermoscopic"], torch.ones(3, 4, 4)))

    def test_balance_argument_validation(self) -> None:
        validate_balance_args(balance_args())
        with self.assertRaisesRegex(ValueError, "weighted-sampler"):
            validate_balance_args(balance_args(weighted_sampler=True))
        with self.assertRaisesRegex(ValueError, "head-ratio"):
            validate_balance_args(balance_args(balance_head_ratio=0))
        with self.assertRaisesRegex(ValueError, "tail-floor"):
            validate_balance_args(balance_args(balance_tail_floor=-1))
        with self.assertRaisesRegex(ValueError, "min-source-count"):
            validate_balance_args(balance_args(balance_min_source_count=0))

    def test_augmented_rows_are_filtered_by_original_train_source(self) -> None:
        base = pd.DataFrame({"lesion_id": ["TRAIN", "VAL"], "label": ["A", "A"]})
        train = base.iloc[[0]].copy()
        val = base.iloc[[1]].copy()
        augmented = pd.DataFrame(
            {
                "lesion_id": ["TRAIN__sdpair_000", "VAL__sdpair_000"],
                "label": ["A", "A"],
                "is_augmented": [True, True],
                "ignore_metadata": [False, False],
            }
        )
        args = argparse.Namespace(
            augmented_data_dir=Path("augmented"),
            augmented_max_per_class=0,
            zero_augmented_metadata=False,
            seed=42,
        )
        with patch("milk10k_effb2_metadata.runner.load_augmented_subset", return_value=augmented):
            result = append_augmented_train_rows(base, train, val, ["A"], args)
        self.assertEqual(result["lesion_id"].tolist(), ["TRAIN", "TRAIN__sdpair_000"])


if __name__ == "__main__":
    unittest.main()
