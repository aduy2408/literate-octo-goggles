from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

MISSING_DEPENDENCY: str | None = None

try:
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - local minimal env may omit ML deps.
    MISSING_DEPENDENCY = exc.name

if MISSING_DEPENDENCY is None:
    try:
        import numpy as np
        import pandas as pd
        from PIL import Image
        from milk10k_effb2_metadata.data import PairedMilk10kMetadataDataset
        from milk10k_effb2_metadata.inference import build_model_from_checkpoint
        from milk10k_effb2_metadata.losses import SoftMacroF1Loss, f1_class_weight_tensor
        from milk10k_effb2_metadata.models import DualEffB2MetadataClassifier
    except ModuleNotFoundError as exc:  # pragma: no cover - local minimal env may omit ML deps.
        MISSING_DEPENDENCY = exc.name


if MISSING_DEPENDENCY is not None:
    class MissingDependencyTest(unittest.TestCase):
        @unittest.skip(f"Missing ML test dependency: {MISSING_DEPENDENCY}")
        def test_missing_dependency(self) -> None:
            pass


if MISSING_DEPENDENCY is None:
    class FakeEncoder(nn.Module):
        def __init__(self, feature_dim: int) -> None:
            super().__init__()
            self.feature_dim = feature_dim

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            pooled = images.mean(dim=(2, 3))
            repeats = (self.feature_dim + pooled.size(1) - 1) // pooled.size(1)
            return pooled.repeat(1, repeats)[:, : self.feature_dim]

        def features(self, images: torch.Tensor) -> torch.Tensor:
            return self.forward(images).view(images.size(0), self.feature_dim, 1, 1)


    def fake_build_feature_encoder(backbone: str, backbone_backend: str, imagenet_pretrained: bool):
        feature_dim = 16
        return FakeEncoder(feature_dim), feature_dim


    class FusionSmokeTest(unittest.TestCase):
        def test_all_image_and_metadata_fusions_forward(self) -> None:
            modes = [
                "concat",
                "cross_attention",
                "co_attention",
                "compact_bilinear",
                "low_rank_bilinear",
                "adaptive_gate",
                "moe",
                "shared_private",
            ]
            metadata_modes = ["concat", "gated_concat", "gated_only"]
            with patch("milk10k_effb2_metadata.models.build_feature_encoder", side_effect=fake_build_feature_encoder):
                for mode in modes:
                    for metadata_mode in metadata_modes:
                        with self.subTest(mode=mode, metadata_mode=metadata_mode):
                            model = DualEffB2MetadataClassifier(
                                num_classes=4,
                                metadata_input_dim=5,
                                branch_dim=8,
                                metadata_dim=6,
                                classifier_hidden_dim=12,
                                dropout=0.0,
                                imagenet_pretrained=False,
                                clinical_backbone_backend="torchvision",
                                dermoscopic_backbone_backend="torchvision",
                                backbone="efficientnet_b2",
                                metadata_fusion=metadata_mode,
                                image_fusion=mode,
                            )
                            logits = model(
                                torch.randn(2, 3, 8, 8),
                                torch.randn(2, 3, 8, 8),
                                torch.randn(2, 5),
                            )
                            self.assertEqual(tuple(logits.shape), (2, 4))

        def test_expected_fused_dims(self) -> None:
            branch_dim = 8
            metadata_dim = 6
            expected = {
                "concat": 22,
                "cross_attention": 22,
                "co_attention": 38,
                "compact_bilinear": 30,
                "low_rank_bilinear": 30,
                "adaptive_gate": 30,
                "shared_private": 30,
                "moe": 22,
                "single_encoder_canvas": 14,
                "shared_encoder_pool": 30,
            }
            for mode, expected_dim in expected.items():
                with self.subTest(mode=mode):
                    self.assertEqual(DualEffB2MetadataClassifier._fusion_dim(branch_dim, metadata_dim, mode), expected_dim)
                    gated_only_dim = expected_dim - metadata_dim
                    self.assertEqual(DualEffB2MetadataClassifier._fusion_dim(branch_dim, 0, mode), gated_only_dim)

        def test_one_encoder_fusions_forward_with_and_without_metadata(self) -> None:
            with patch("milk10k_effb2_metadata.models.build_feature_encoder", side_effect=fake_build_feature_encoder):
                for mode in ("single_encoder_canvas", "shared_encoder_pool"):
                    for disable_metadata in (False, True):
                        with self.subTest(mode=mode, disable_metadata=disable_metadata):
                            model = DualEffB2MetadataClassifier(
                                num_classes=4,
                                metadata_input_dim=5,
                                branch_dim=8,
                                metadata_dim=6,
                                classifier_hidden_dim=12,
                                dropout=0.0,
                                imagenet_pretrained=False,
                                clinical_backbone_backend="torchvision",
                                dermoscopic_backbone_backend="torchvision",
                                backbone="efficientnet_b2",
                                disable_metadata=disable_metadata,
                                image_fusion=mode,
                            )
                            self.assertTrue(hasattr(model, "shared_encoder"))
                            self.assertFalse(hasattr(model, "clinical_encoder"))
                            self.assertFalse(hasattr(model, "dermoscopic_encoder"))
                            logits = model(
                                torch.randn(2, 3, 8, 8),
                                torch.randn(2, 3, 8, 8),
                                torch.randn(2, 5),
                            )
                            self.assertEqual(tuple(logits.shape), (2, 4))

        def test_checkpoint_reconstructs_one_encoder_model(self) -> None:
            checkpoint_args = {
                "branch_dim": 8,
                "metadata_dim": 6,
                "classifier_hidden_dim": 12,
                "dropout": 0.0,
                "metadata_fusion": "concat",
                "image_fusion": "single_encoder_canvas",
                "logit_fusion_mode": "single",
                "backbone": "efficientnet_b2",
            }
            with patch("milk10k_effb2_metadata.models.build_feature_encoder", side_effect=fake_build_feature_encoder):
                model = DualEffB2MetadataClassifier(
                    num_classes=4,
                    metadata_input_dim=5,
                    branch_dim=8,
                    metadata_dim=6,
                    classifier_hidden_dim=12,
                    dropout=0.0,
                    imagenet_pretrained=False,
                    clinical_backbone_backend="timm",
                    dermoscopic_backbone_backend="timm",
                    backbone="efficientnet_b2",
                    image_fusion="single_encoder_canvas",
                )
                with patch("milk10k_effb2_metadata.inference.infer_backend_from_model_state", return_value="timm"):
                    loaded = build_model_from_checkpoint(
                        {
                            "model_state": model.state_dict(),
                            "class_names": ["A", "B", "C", "D"],
                            "args": checkpoint_args,
                        },
                        metadata_dim=5,
                        device=torch.device("cpu"),
                    )
            self.assertTrue(hasattr(loaded, "shared_encoder"))
            self.assertFalse(hasattr(loaded, "clinical_encoder"))


    class MetadataAugmentationTest(unittest.TestCase):
        def test_image_transform_does_not_change_metadata(self) -> None:
            with tempfile.TemporaryDirectory() as tmp_dir:
                image_path = Path(tmp_dir) / "image.jpg"
                Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(image_path)
                df = pd.DataFrame(
                    [
                        {
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
                    ]
                )
                metadata_spec = {"sex_values": ["female"], "site_values": ["arm"], "monet_columns": []}

                def noisy_transform(image):
                    return torch.rand(3, image.height, image.width)

                dataset = PairedMilk10kMetadataDataset(df, {"BCC": 0}, metadata_spec, noisy_transform)
                first = dataset[0]["metadata"]
                second = dataset[0]["metadata"]
                self.assertTrue(torch.equal(first, second))


    class F1LossControlTest(unittest.TestCase):
        def test_f1_class_controls_ignore_and_weight_classes(self) -> None:
            label_to_idx = {"BCC": 0, "MAL_OTH": 1, "MEL": 2}
            args = argparse.Namespace(f1_ignore_classes=["MAL_OTH"], f1_class_weight=["BCC=2.5"])
            weights = f1_class_weight_tensor(label_to_idx, args, torch.device("cpu"))
            self.assertTrue(torch.equal(weights, torch.tensor([2.5, 0.0, 1.0])))

        def test_soft_f1_ignores_zero_weight_class(self) -> None:
            logits = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
            labels = torch.tensor([0, 1, 2])
            masked = SoftMacroF1Loss(torch.tensor([1.0, 0.0, 1.0]))(logits, labels)
            probs = torch.softmax(logits, dim=1)
            one_hot = torch.nn.functional.one_hot(labels, num_classes=3).float()
            tp = (probs * one_hot).sum(dim=0)
            fp = (probs * (1.0 - one_hot)).sum(dim=0)
            fn = ((1.0 - probs) * one_hot).sum(dim=0)
            f1 = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
            manual = 1.0 - (f1[0] + f1[2]) / 2.0
            self.assertAlmostEqual(float(masked), float(manual), places=6)


if __name__ == "__main__":
    unittest.main()
