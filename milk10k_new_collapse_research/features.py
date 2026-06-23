"""Frozen feature extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Dataset


DINOV2_MODELS = {"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"}


@dataclass
class FeatureBatch:
    lesion_ids: list[str]
    labels: list[str]
    clinical: np.ndarray
    dermoscopic: np.ndarray


class PairImageDataset(Dataset):
    def __init__(self, df, transform) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, path: str) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.df.iloc[idx]
        return {
            "lesion_id": str(row["lesion_id"]),
            "label": str(row["label"]),
            "clinical": self._load(row["clinical_path"]),
            "dermoscopic": self._load(row["dermoscopic_path"]),
        }


def build_feature_model(model_name: str, device: torch.device, pretrained: bool = True):
    if model_name in DINOV2_MODELS:
        if not pretrained:
            raise ValueError("DINOv2 torch.hub models require pretrained weights; use a timm model for --no-pretrained smoke tests.")
        model = torch.hub.load("facebookresearch/dinov2", model_name)
        image_size = 224
        transform = dinov2_transform(image_size)
    else:
        import timm

        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        cfg = resolve_data_config({}, model=model)
        image_size = int(cfg.get("input_size", (3, 224, 224))[-1])
        transform = create_transform(**cfg, is_training=False)
    model.to(device).eval()
    return model, transform, image_size


def dinov2_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


@torch.no_grad()
def extract_pair_features(model, loader: DataLoader, device: torch.device, l2_normalize: bool = True) -> FeatureBatch:
    lesion_ids: list[str] = []
    labels: list[str] = []
    clinical_features: list[np.ndarray] = []
    dermoscopic_features: list[np.ndarray] = []
    for batch in loader:
        clinical = batch["clinical"].to(device, non_blocking=True)
        dermoscopic = batch["dermoscopic"].to(device, non_blocking=True)
        clinical_z = model(clinical)
        dermoscopic_z = model(dermoscopic)
        if l2_normalize:
            clinical_z = F.normalize(clinical_z, dim=1)
            dermoscopic_z = F.normalize(dermoscopic_z, dim=1)
        clinical_features.append(clinical_z.cpu().numpy())
        dermoscopic_features.append(dermoscopic_z.cpu().numpy())
        lesion_ids.extend(str(item) for item in batch["lesion_id"])
        labels.extend(str(item) for item in batch["label"])
    return FeatureBatch(
        lesion_ids=lesion_ids,
        labels=labels,
        clinical=np.concatenate(clinical_features, axis=0),
        dermoscopic=np.concatenate(dermoscopic_features, axis=0),
    )


def save_feature_npz(path: Path, batch: FeatureBatch, model_name: str, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        lesion_id=np.asarray(batch.lesion_ids, dtype=object),
        label=np.asarray(batch.labels, dtype=object),
        clinical=batch.clinical,
        dermoscopic=batch.dermoscopic,
        model_name=np.asarray(model_name),
        split=np.asarray(split),
    )


def make_feature_matrix(data: dict[str, np.ndarray], mode: str = "pair") -> np.ndarray:
    clinical = data["clinical"]
    dermoscopic = data["dermoscopic"]
    if mode == "clinical":
        return clinical
    if mode == "dermoscopic":
        return dermoscopic
    if mode == "pair":
        return np.concatenate([clinical, dermoscopic, np.abs(clinical - dermoscopic), clinical * dermoscopic], axis=1)
    raise ValueError(f"Unsupported feature mode: {mode}")
