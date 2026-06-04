#!/usr/bin/env python3
"""
MILK10k dataset utilities shared by training scripts.

Keep dataframe construction and torch Dataset classes here; training scripts
should build transforms/loaders and own model/training logic.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

REQUIRED_DATA_FILES = (
    "MILK10k_Training_GroundTruth.csv",
    "MILK10k_Training_Metadata.csv",
    "MILK10k_Training_Input",
)

LABEL_COLUMNS = [
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


class Milk10kDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict[str, int], transform=None) -> None:
        self.paths = df["path"].tolist()
        self.labels = [label_to_idx[label] for label in df["label"].tolist()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[idx]) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


class PairedMilk10kDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict[str, int], transform=None) -> None:
        self.clinical_paths = df["clinical_path"].tolist()
        self.dermoscopic_paths = df["dermoscopic_path"].tolist()
        self.labels = [label_to_idx[label] for label in df["label"].tolist()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def _load_image(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        clinical = self._load_image(self.clinical_paths[idx])
        dermoscopic = self._load_image(self.dermoscopic_paths[idx])
        return torch.stack([clinical, dermoscopic], dim=0), self.labels[idx]


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def normalize_image_type(image_type: str) -> str:
    if image_type == "clinical: close-up":
        return "clinical_close_up"
    return image_type.replace(" ", "_").replace(":", "").replace("-", "_")


def has_milk10k_files(path: Path) -> bool:
    return all((path / name).exists() for name in REQUIRED_DATA_FILES)


def resolve_data_dir(data_dir: Path | None) -> Path:
    if data_dir is not None:
        data_dir = data_dir.expanduser().resolve()
        if not has_milk10k_files(data_dir):
            expected = ", ".join(REQUIRED_DATA_FILES)
            raise FileNotFoundError(f"--data-dir={data_dir} does not contain required MILK10k files: {expected}")
        return data_dir

    candidates = [Path.cwd()]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(path.parent for path in kaggle_input.rglob("MILK10k_Training_GroundTruth.csv"))

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if has_milk10k_files(candidate):
            return candidate

    expected = ", ".join(REQUIRED_DATA_FILES)
    raise FileNotFoundError(
        f"Could not auto-detect MILK10k data dir. Pass --data-dir PATH containing: {expected}"
    )


def load_dataframe(data_dir: Path, image_type: str) -> pd.DataFrame:
    input_dir = data_dir / "MILK10k_Training_Input"
    gt = pd.read_csv(data_dir / "MILK10k_Training_GroundTruth.csv")
    meta = pd.read_csv(data_dir / "MILK10k_Training_Metadata.csv")

    gt["label"] = gt[LABEL_COLUMNS].idxmax(axis=1)
    df = meta.merge(gt[["lesion_id", "label"]], on="lesion_id", how="inner")
    df["image_type_norm"] = df["image_type"].map(normalize_image_type)

    if image_type != "all":
        df = df[df["image_type_norm"] == image_type].copy()

    df["path"] = df.apply(lambda r: input_dir / r["lesion_id"] / f"{r['isic_id']}.jpg", axis=1)
    df = df[df["path"].map(lambda p: p.exists())].copy()
    df["path"] = df["path"].map(str)

    if df.empty:
        raise ValueError(f"No images found for image_type={image_type!r} under {input_dir}")
    return df[["path", "label", "lesion_id", "isic_id", "image_type_norm"]]


def to_paired_lesion_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clinical = (
        df[df["image_type_norm"] == "clinical_close_up"][["lesion_id", "path"]]
        .rename(columns={"path": "clinical_path"})
        .drop_duplicates("lesion_id")
    )
    dermoscopic = (
        df[df["image_type_norm"] == "dermoscopic"][["lesion_id", "path"]]
        .rename(columns={"path": "dermoscopic_path"})
        .drop_duplicates("lesion_id")
    )
    labels = df[["lesion_id", "label"]].drop_duplicates("lesion_id")
    paired = labels.merge(clinical, on="lesion_id", how="inner").merge(dermoscopic, on="lesion_id", how="inner")
    if paired.empty:
        raise ValueError("No paired clinical/dermoscopic lesions found.")
    return paired[["lesion_id", "label", "clinical_path", "dermoscopic_path"]]


def lesion_level_train_val_split(
    df: pd.DataFrame,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lesion_df = df[["lesion_id", "label"]].drop_duplicates("lesion_id")

    train_lesions, val_lesions = train_test_split(
        lesion_df,
        test_size=val_size,
        stratify=lesion_df["label"],
        random_state=seed,
    )

    train_df = df[df["lesion_id"].isin(train_lesions["lesion_id"])].copy()
    val_df = df[df["lesion_id"].isin(val_lesions["lesion_id"])].copy()
    return train_df, val_df
