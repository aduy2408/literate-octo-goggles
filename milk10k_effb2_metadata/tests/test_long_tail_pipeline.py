from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

MISSING_DEPENDENCY: str | None = None

try:
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover
    MISSING_DEPENDENCY = exc.name

if MISSING_DEPENDENCY is None:
    from milk10k_effb2_metadata.inference import predict_dataframe
    from milk10k_effb2_metadata.losses import GeneralizedBalancedSoftmaxLoss
    from milk10k_effb2_metadata.model_setup import load_model_state_compat
    from milk10k_effb2_metadata.runner import fit_global_temperature, train_lws_post_training


if MISSING_DEPENDENCY is not None:
    class MissingDependencyTest(unittest.TestCase):
        @unittest.skip(f"Missing ML test dependency: {MISSING_DEPENDENCY}")
        def test_missing_dependency(self) -> None:
            pass


if MISSING_DEPENDENCY is None:
    class TinyDataset(Dataset):
        def __init__(self) -> None:
            self.labels = [0, 0, 0, 1]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            value = float(idx)
            return {
                "clinical": torch.tensor([value]),
                "dermoscopic": torch.tensor([0.0]),
                "metadata": torch.tensor([0.0]),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }


    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = nn.Parameter(torch.tensor([[1.0], [-1.0]]))
            self.class_scales = nn.Parameter(torch.ones(2), requires_grad=False)

        def forward(self, clinical, dermoscopic, metadata):
            return F.linear(clinical, self.base) * self.class_scales


    class LongTailPipelineTest(unittest.TestCase):
        def test_generalized_balanced_softmax_is_train_only(self) -> None:
            logits = torch.tensor([[1.0, 2.0]])
            labels = torch.tensor([0])
            counts = torch.tensor([10.0, 2.0])
            criterion = GeneralizedBalancedSoftmaxLoss(counts, tau=0.25)
            criterion.train()
            expected = F.cross_entropy(logits + 0.25 * counts.log(), labels)
            self.assertTrue(torch.allclose(criterion(logits, labels), expected))
            criterion.eval()
            self.assertTrue(torch.allclose(criterion(logits, labels), F.cross_entropy(logits, labels)))

        def test_legacy_state_defaults_lws_scales_to_one(self) -> None:
            model = TinyModel()
            legacy_state = {"base": torch.tensor([[2.0], [-2.0]])}
            load_model_state_compat(model, legacy_state)
            self.assertTrue(torch.equal(model.class_scales, torch.ones(2)))

        def test_lws_only_updates_bounded_scales(self) -> None:
            model = TinyModel()
            loader = DataLoader(TinyDataset(), batch_size=2)
            original_base = model.base.detach().clone()
            args = argparse.Namespace(
                lws_epochs=2,
                lws_lr=0.1,
                lws_sampler_power=0.5,
                lws_min_scale=0.75,
                lws_max_scale=1.5,
                batch_size=2,
                num_workers=0,
                seed=42,
                selection_metric="f1_macro",
            )
            checkpoint = {"class_names": ["A", "B"], "model_state": model.state_dict()}
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "best_lws.pt"
                metrics = train_lws_post_training(
                    model, loader, loader, torch.device("cpu"), args, checkpoint, output
                )
                self.assertTrue(output.exists())
                self.assertIsNotNone(metrics)
            self.assertTrue(torch.equal(model.base, original_base))
            self.assertGreaterEqual(float(model.class_scales.min()), 0.75)
            self.assertLessEqual(float(model.class_scales.max()), 1.5)

        def test_temperature_is_positive_and_applied_at_inference(self) -> None:
            model = TinyModel()
            loader = DataLoader(TinyDataset(), batch_size=2)
            temperature = fit_global_temperature(model, loader, torch.device("cpu"))
            self.assertGreater(temperature, 0.0)
            cold = predict_dataframe(model, loader, torch.device("cpu"), temperature=0.5)
            warm = predict_dataframe(model, loader, torch.device("cpu"), temperature=2.0)
            self.assertFalse(np.allclose(cold, warm))


if __name__ == "__main__":
    unittest.main()
