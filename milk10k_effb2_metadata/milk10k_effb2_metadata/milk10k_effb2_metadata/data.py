"""Dataframe, metadata, split, and dataloader helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from datasets import LABEL_COLUMNS, normalize_image_type

ImageFile.LOAD_TRUNCATED_IMAGES = True

METADATA_COLUMNS = ("age_approx", "sex", "skin_tone_class", "site")


class PairedMilk10kMetadataDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_to_idx: dict[str, int],
        metadata_spec: dict[str, Any],
        transform=None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.labels = [label_to_idx[label] for label in self.df["label"].tolist()]
        self.metadata = np.stack([metadata_vector(row, metadata_spec) for _, row in self.df.iterrows()])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        return {
            "clinical": self._load_image(row["clinical_path"]),
            "dermoscopic": self._load_image(row["dermoscopic_path"]),
            "metadata": torch.from_numpy(self.metadata[idx]),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_paired_dataframe(data_dir: Path) -> pd.DataFrame:
    input_dir = data_dir / "MILK10k_Training_Input"
    gt = pd.read_csv(data_dir / "MILK10k_Training_GroundTruth.csv")
    meta = pd.read_csv(data_dir / "MILK10k_Training_Metadata.csv")
    monet_columns = resolve_monet_columns(meta)

    gt["label"] = gt[LABEL_COLUMNS].idxmax(axis=1)
    meta["image_type_norm"] = meta["image_type"].map(normalize_image_type)
    meta["path"] = meta.apply(lambda r: input_dir / r["lesion_id"] / f"{r['isic_id']}.jpg", axis=1)
    meta = meta[meta["path"].map(lambda p: p.exists())].copy()
    meta["path"] = meta["path"].map(str)

    keep = ["lesion_id", "path", *METADATA_COLUMNS, *monet_columns]
    clinical = meta[meta["image_type_norm"] == "clinical_close_up"][keep].drop_duplicates("lesion_id")
    dermoscopic = meta[meta["image_type_norm"] == "dermoscopic"][keep].drop_duplicates("lesion_id")
    paired = (
        gt[["lesion_id", "label"]]
        .merge(clinical.add_prefix("clinical_"), left_on="lesion_id", right_on="clinical_lesion_id")
        .merge(dermoscopic.add_prefix("dermoscopic_"), left_on="lesion_id", right_on="dermoscopic_lesion_id")
        .drop(columns=["clinical_lesion_id", "dermoscopic_lesion_id"])
    )
    if paired.empty:
        raise ValueError(f"No paired clinical/dermoscopic lesions found under {input_dir}")
    return paired


def resolve_monet_columns(meta: pd.DataFrame) -> list[str]:
    try:
        from milk10k_dual_encoder.config import MONET_COLUMNS

        configured = [column for column in MONET_COLUMNS if column in meta.columns]
        if configured:
            return configured
    except Exception:
        pass
    return sorted(column for column in meta.columns if column.startswith("MONET_"))


def lesion_split(df: pd.DataFrame, val_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    lesion_df = df[["lesion_id", "label"]].drop_duplicates("lesion_id")
    train_lesions, val_lesions = train_test_split(
        lesion_df,
        test_size=val_size,
        stratify=lesion_df["label"],
        random_state=seed,
    )
    return split_by_lesion_ids(df, train_lesions["lesion_id"], val_lesions["lesion_id"])


def kfold_splits(df: pd.DataFrame, k_folds: int, seed: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    if k_folds < 2:
        raise ValueError("--k-folds must be 1 for single split or at least 2 for k-fold training.")

    lesion_df = df[["lesion_id", "label"]].drop_duplicates("lesion_id").reset_index(drop=True)
    min_class_count = int(lesion_df["label"].value_counts().min())
    if k_folds > min_class_count:
        raise ValueError(
            f"--k-folds={k_folds} is larger than the smallest class count ({min_class_count}). "
            "Use fewer folds or merge/remove ultra-rare classes."
        )

    splitter = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in splitter.split(lesion_df["lesion_id"], lesion_df["label"]):
        train_lesions = lesion_df.iloc[train_idx]["lesion_id"]
        val_lesions = lesion_df.iloc[val_idx]["lesion_id"]
        splits.append(split_by_lesion_ids(df, train_lesions, val_lesions))
    return splits


def split_by_lesion_ids(
    df: pd.DataFrame,
    train_lesions: pd.Series,
    val_lesions: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        df[df["lesion_id"].isin(train_lesions)].copy(),
        df[df["lesion_id"].isin(val_lesions)].copy(),
    )


def fit_metadata_spec(train_df: pd.DataFrame) -> dict[str, Any]:
    sex_values = sorted({"unknown"} | collect_string_values(train_df, "sex"))
    site_values = sorted({"unknown"} | collect_string_values(train_df, "site"))
    return {
        "sex_values": sex_values,
        "site_values": site_values,
        "monet_columns": infer_paired_monet_columns(train_df),
    }


def collect_string_values(df: pd.DataFrame, field: str) -> set[str]:
    values: set[str] = set()
    for prefix in ("clinical", "dermoscopic"):
        series = df[f"{prefix}_{field}"].fillna("unknown").astype(str).str.strip()
        values.update(value if value else "unknown" for value in series.tolist())
    return values


def infer_paired_monet_columns(df: pd.DataFrame) -> list[str]:
    clinical_prefix = "clinical_MONET_"
    return sorted(
        column.removeprefix("clinical_")
        for column in df.columns
        if column.startswith(clinical_prefix) and f"dermoscopic_{column.removeprefix('clinical_')}" in df.columns
    )


def metadata_vector(row: pd.Series, spec: dict[str, Any]) -> np.ndarray:
    age = first_numeric(row, "age_approx")
    skin_tone = first_numeric(row, "skin_tone_class")
    sex = first_string(row, "sex")
    site = first_string(row, "site")

    values: list[float] = [
        0.0 if age is None else float(age) / 100.0,
        0.0 if skin_tone is None else float(skin_tone) / 6.0,
    ]
    values.extend(1.0 if sex == item else 0.0 for item in spec["sex_values"])
    values.extend(1.0 if site == item else 0.0 for item in spec["site_values"])

    for prefix in ("clinical", "dermoscopic"):
        for column in spec.get("monet_columns", []):
            value = pd.to_numeric(row.get(f"{prefix}_{column}"), errors="coerce")
            values.append(0.0 if pd.isna(value) else float(value))

    return np.asarray(values, dtype=np.float32)


def first_numeric(row: pd.Series, field: str) -> float | None:
    for prefix in ("clinical", "dermoscopic"):
        value = pd.to_numeric(row.get(f"{prefix}_{field}"), errors="coerce")
        if not pd.isna(value):
            return float(value)
    return None


def first_string(row: pd.Series, field: str) -> str:
    for prefix in ("clinical", "dermoscopic"):
        value = row.get(f"{prefix}_{field}")
        if pd.notna(value):
            value = str(value).strip()
            if value:
                return value
    return "unknown"


def make_transforms(image_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_resize = round(image_size * 1.12)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(1.2, 1.45)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(eval_resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_to_idx: dict[str, int],
    metadata_spec: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    train_transform, eval_transform = make_transforms(args.image_size)
    train_ds = PairedMilk10kMetadataDataset(train_df, label_to_idx, metadata_spec, train_transform)
    val_ds = PairedMilk10kMetadataDataset(val_df, label_to_idx, metadata_spec, eval_transform)
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    sampler = build_weighted_sampler(train_ds, args) if args.weighted_sampler else None
    train_loader = DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


def build_weighted_sampler(
    dataset: PairedMilk10kMetadataDataset,
    args: argparse.Namespace,
) -> WeightedRandomSampler:
    labels = np.asarray(dataset.labels)
    counts = np.bincount(labels)
    if np.any(counts == 0):
        raise ValueError("Cannot build weighted sampler because at least one class has zero training samples.")
    class_weights = 1.0 / np.power(counts.astype(np.float64), args.sampler_power)
    sample_weights = torch.as_tensor(class_weights[labels], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    return WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True, generator=generator)
