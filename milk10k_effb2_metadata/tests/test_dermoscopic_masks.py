from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

MISSING_DEPENDENCY: str | None = None

try:
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from milk10k_effb2_metadata.data import (
        DERMOSCOPIC_MASK_PATH_COLUMN,
        PairedMilk10kMetadataDataset,
        apply_dermoscopic_mask,
        audit_dermoscopic_masks,
    )
    from milk10k_effb2_metadata.inference import InferencePairedDataset
except ModuleNotFoundError as exc:  # pragma: no cover - allows collection in minimal environments.
    MISSING_DEPENDENCY = exc.name


@unittest.skipIf(MISSING_DEPENDENCY is not None, f"Missing ML test dependency: {MISSING_DEPENDENCY}")
class DermoscopicMaskTest(unittest.TestCase):
    @staticmethod
    def _save_rgb(path: Path, size: tuple[int, int] = (10, 10)) -> None:
        Image.new("RGB", size, color=(100, 150, 200)).save(path)

    @staticmethod
    def _save_mask(path: Path, foreground_pixels: int, size: tuple[int, int] = (10, 10)) -> None:
        pixels = np.zeros(size[0] * size[1], dtype=np.uint8)
        pixels[:foreground_pixels] = 255
        Image.fromarray(pixels.reshape(size[1], size[0]), mode="L").save(path)

    def test_apply_mask_blacks_background_and_keeps_foreground(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mask_path = Path(tmp) / "mask.png"
            self._save_mask(mask_path, foreground_pixels=1, size=(2, 2))
            image = Image.new("RGB", (2, 2), color=(100, 150, 200))
            result = np.asarray(apply_dermoscopic_mask(image, mask_path))
            self.assertTrue(np.array_equal(result[0, 0], [100, 150, 200]))
            self.assertTrue(np.all(result.reshape(-1, 3)[1:] == 0))

    def test_audit_statuses_and_one_percent_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_dir = root / "masks"
            mask_dir.mkdir()
            rows = []
            for lesion_id in ("valid", "small", "missing", "broken", "mismatch"):
                image_path = root / f"{lesion_id}.png"
                self._save_rgb(image_path)
                rows.append({"lesion_id": lesion_id, "dermoscopic_path": str(image_path)})

            self._save_mask(mask_dir / "valid_dermoscopic_mask.png", foreground_pixels=1)
            self._save_mask(mask_dir / "small_dermoscopic_mask.png", foreground_pixels=0)
            (mask_dir / "broken_dermoscopic_mask.png").write_text("not an image")
            self._save_mask(mask_dir / "mismatch_dermoscopic_mask.png", foreground_pixels=1, size=(5, 5))

            audited, audit = audit_dermoscopic_masks(pd.DataFrame(rows), mask_dir, 0.01)
            statuses = dict(zip(audit["lesion_id"], audit["status"]))
            self.assertEqual(
                statuses,
                {
                    "valid": "valid",
                    "small": "too_small",
                    "missing": "missing",
                    "broken": "unreadable",
                    "mismatch": "size_mismatch",
                },
            )
            valid_row = audited[audited["lesion_id"] == "valid"].iloc[0]
            self.assertAlmostEqual(float(valid_row["dermoscopic_mask_ratio"]), 0.01)
            self.assertIsInstance(valid_row[DERMOSCOPIC_MASK_PATH_COLUMN], str)
            fallback_paths = audited[audited["lesion_id"] != "valid"][DERMOSCOPIC_MASK_PATH_COLUMN]
            self.assertTrue(fallback_paths.isna().all())

    def test_dataset_supports_mask_enabled_and_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            self._save_rgb(image_path, size=(2, 2))
            self._save_mask(mask_path, foreground_pixels=1, size=(2, 2))
            base_row = {
                "lesion_id": "L1",
                "label": "BCC",
                "clinical_path": str(image_path),
                "dermoscopic_path": str(image_path),
                "clinical_age_approx": 60,
                "dermoscopic_age_approx": 60,
                "clinical_skin_tone_class": 3,
                "dermoscopic_skin_tone_class": 3,
                "clinical_sex": "female",
                "dermoscopic_sex": "female",
                "clinical_site": "arm",
                "dermoscopic_site": "arm",
            }
            metadata_spec = {"sex_values": ["female"], "site_values": ["arm"], "monet_columns": []}
            transform = lambda image: torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)

            original_ds = PairedMilk10kMetadataDataset(
                pd.DataFrame([base_row]), {"BCC": 0}, metadata_spec, transform
            )
            masked_ds = PairedMilk10kMetadataDataset(
                pd.DataFrame([{**base_row, DERMOSCOPIC_MASK_PATH_COLUMN: str(mask_path)}]),
                {"BCC": 0},
                metadata_spec,
                transform,
            )

            self.assertTrue(torch.all(original_ds[0]["dermoscopic"] > 0))
            self.assertEqual(int((masked_ds[0]["dermoscopic"] != 0).any(dim=0).sum()), 1)
            self.assertTrue(torch.equal(masked_ds[0]["clinical"], original_ds[0]["clinical"]))

            inference_ds = InferencePairedDataset(
                pd.DataFrame([{**base_row, DERMOSCOPIC_MASK_PATH_COLUMN: str(mask_path)}]),
                metadata_spec,
                transform,
            )
            self.assertTrue(torch.equal(inference_ds[0]["dermoscopic"], masked_ds[0]["dermoscopic"]))

    def test_inference_audit_maps_masks_by_dermoscopic_isic_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_dir = root / "masks"
            mask_dir.mkdir()
            image_path = root / "image.png"
            self._save_rgb(image_path)
            self._save_mask(mask_dir / "ISIC_123_mask.png", foreground_pixels=1)
            df = pd.DataFrame(
                [{"lesion_id": "IL_999", "dermoscopic_isic_id": "ISIC_123", "dermoscopic_path": image_path}]
            )

            audited, audit = audit_dermoscopic_masks(
                df,
                mask_dir,
                0.01,
                mask_id_column="dermoscopic_isic_id",
                mask_suffix="_mask.png",
            )

            self.assertEqual(audit.iloc[0]["mask_id"], "ISIC_123")
            self.assertEqual(audit.iloc[0]["status"], "valid")
            self.assertTrue(str(audited.iloc[0][DERMOSCOPIC_MASK_PATH_COLUMN]).endswith("ISIC_123_mask.png"))


if __name__ == "__main__":
    unittest.main()
