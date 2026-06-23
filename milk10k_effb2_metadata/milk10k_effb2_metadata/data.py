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
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torchvision import transforms

from datasets import LABEL_COLUMNS, normalize_image_type

ImageFile.LOAD_TRUNCATED_IMAGES = True

METADATA_COLUMNS = ("age_approx", "sex", "skin_tone_class", "site")
DERMOSCOPIC_MASK_PATH_COLUMN = "dermoscopic_mask_path"
DERMOSCOPIC_MASK_RATIO_COLUMN = "dermoscopic_mask_ratio"
DERMOSCOPIC_MASK_STATUS_COLUMN = "dermoscopic_mask_status"


def apply_dermoscopic_mask(image: Image.Image, mask_path: str | Path | None) -> Image.Image:
    """Return an RGB image with non-mask pixels black, or the original RGB image on read failure."""
    image = image.convert("RGB")
    if not isinstance(mask_path, (str, Path)) or not str(mask_path):
        return image
    try:
        with Image.open(mask_path) as mask_image:
            mask = mask_image.convert("L")
            if mask.size != image.size:
                return image
            binary_mask = mask.point(lambda value: 255 if value else 0)
            return Image.composite(image, Image.new("RGB", image.size), binary_mask)
    except (OSError, ValueError):
        return image


def audit_dermoscopic_masks(
    df: pd.DataFrame,
    mask_dir: Path,
    min_foreground_ratio: float = 0.01,
    mask_id_column: str = "lesion_id",
    mask_suffix: str = "_dermoscopic_mask.png",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach valid mask paths and return one audit row per paired dermoscopic image."""
    if not 0.0 <= min_foreground_ratio <= 1.0:
        raise ValueError("--min-dermoscopic-mask-ratio must be between 0 and 1.")
    mask_dir = mask_dir.expanduser().resolve()
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Dermoscopic mask directory does not exist: {mask_dir}")
    if mask_id_column not in df.columns:
        raise ValueError(f"Mask ID column is missing from dataframe: {mask_id_column}")

    audited_df = df.copy()
    mask_paths: list[str | None] = []
    ratios: list[float | None] = []
    statuses: list[str] = []
    audit_rows: list[dict[str, Any]] = []

    for _, row in audited_df.iterrows():
        lesion_id = str(row["lesion_id"])
        mask_id = str(row[mask_id_column])
        image_path = Path(row["dermoscopic_path"])
        mask_path = mask_dir / f"{mask_id}{mask_suffix}"
        ratio: float | None = None
        status = "valid"
        image_size: tuple[int, int] | None = None
        mask_size: tuple[int, int] | None = None

        if not mask_path.is_file():
            status = "missing"
        else:
            try:
                with Image.open(image_path) as image:
                    image_size = image.size
                with Image.open(mask_path) as mask_image:
                    mask = mask_image.convert("L")
                    mask.load()
                    mask_size = mask.size
                    histogram = mask.histogram()
                    total_pixels = mask.width * mask.height
                    ratio = (total_pixels - histogram[0]) / total_pixels if total_pixels else 0.0
                if mask_size != image_size:
                    status = "size_mismatch"
                elif ratio < min_foreground_ratio:
                    status = "too_small"
            except (OSError, ValueError):
                status = "unreadable"

        valid_path = str(mask_path) if status == "valid" else None
        mask_paths.append(valid_path)
        ratios.append(ratio)
        statuses.append(status)
        audit_rows.append(
            {
                "lesion_id": lesion_id,
                "mask_id": mask_id,
                "dermoscopic_path": str(image_path),
                "mask_path": str(mask_path),
                "foreground_ratio": ratio,
                "status": status,
                "image_size": None if image_size is None else f"{image_size[0]}x{image_size[1]}",
                "mask_size": None if mask_size is None else f"{mask_size[0]}x{mask_size[1]}",
            }
        )

    audited_df[DERMOSCOPIC_MASK_PATH_COLUMN] = mask_paths
    audited_df[DERMOSCOPIC_MASK_RATIO_COLUMN] = ratios
    audited_df[DERMOSCOPIC_MASK_STATUS_COLUMN] = statuses
    return audited_df, pd.DataFrame(audit_rows)


def print_mask_audit_summary(audit_df: pd.DataFrame, min_foreground_ratio: float) -> None:
    counts = audit_df["status"].value_counts().sort_index().to_dict()
    valid = int(counts.get("valid", 0))
    print(
        "Dermoscopic masks: "
        f"total={len(audit_df)}, valid={valid}, fallback={len(audit_df) - valid}, "
        f"min_foreground_ratio={min_foreground_ratio:.6f}, status_counts={counts}"
    )


class PairedMilk10kMetadataDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_to_idx: dict[str, int],
        metadata_spec: dict[str, Any],
        transform=None,
        strong_transform=None,
        strong_augment_labels: set[int] | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.labels = [label_to_idx[label] for label in self.df["label"].tolist()]
        self.metadata = np.stack([metadata_vector(row, metadata_spec) for _, row in self.df.iterrows()])
        if "ignore_metadata" in self.df.columns:
            ignore_mask = self.df["ignore_metadata"].fillna(False).astype(bool).to_numpy()
            self.metadata[ignore_mask] = 0.0
        self.transform = transform
        self.strong_transform = strong_transform
        self.strong_augment_labels = strong_augment_labels or set()

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(
        self,
        path: str,
        mask_path: str | Path | None = None,
        transform=None,
    ) -> torch.Tensor:
        with Image.open(path) as img:
            image = apply_dermoscopic_mask(img, mask_path)
        transform = self.transform if transform is None else transform
        if transform is not None:
            image = transform(image)
        return image

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        label = self.labels[idx]
        transform = self.strong_transform if label in self.strong_augment_labels else self.transform
        return {
            "clinical": self._load_image(row["clinical_path"], transform=transform),
            "dermoscopic": self._load_image(
                row["dermoscopic_path"],
                row.get(DERMOSCOPIC_MASK_PATH_COLUMN),
                transform,
            ),
            "metadata": torch.from_numpy(self.metadata[idx]),
            "label": torch.tensor(label, dtype=torch.long),
        }


class HybridEpochSampler(Sampler[int]):
    """Cap the largest class and oversample eligible tail classes per epoch."""

    def __init__(
        self,
        labels: list[int],
        target_counts: np.ndarray,
        seed: int,
        label_names: dict[int, str] | None = None,
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.target_counts = np.asarray(target_counts, dtype=np.int64)
        self.seed = int(seed)
        self.epoch = 0
        self.label_names = label_names or {}
        self.class_indices = [np.flatnonzero(self.labels == idx) for idx in range(len(self.target_counts))]
        self.original_counts = np.asarray([len(indices) for indices in self.class_indices], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.target_counts.sum())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        selected: list[torch.Tensor] = []
        for indices, target in zip(self.class_indices, self.target_counts):
            source = torch.as_tensor(indices, dtype=torch.long)
            target = int(target)
            if target <= len(source):
                selected.append(source[torch.randperm(len(source), generator=generator)[:target]])
                continue
            full_repeats, remainder = divmod(target, len(source))
            chunks = [source[torch.randperm(len(source), generator=generator)] for _ in range(full_repeats)]
            if remainder:
                chunks.append(source[torch.randperm(len(source), generator=generator)[:remainder]])
            selected.append(torch.cat(chunks))
        epoch_indices = torch.cat(selected)
        order = torch.randperm(len(epoch_indices), generator=generator)
        return iter(epoch_indices[order].tolist())

    def exposure_summary(self) -> dict[str, int]:
        return {
            self.label_names.get(idx, str(idx)): int(count)
            for idx, count in enumerate(self.target_counts)
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


def make_strong_train_transform(image_size: int):
    """A conservative stronger variant used only for oversampled tail classes."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(1.15, 1.5)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def hybrid_target_counts(labels: list[int], args: argparse.Namespace) -> tuple[np.ndarray, set[int]]:
    """Return per-class epoch targets and classes eligible for strong augmentation."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64))
    if np.any(counts == 0):
        raise ValueError("Cannot build hybrid sampler because at least one class has zero training samples.")
    targets = counts.copy()

    if len(counts) >= 2:
        descending = np.argsort(-counts, kind="stable")
        head_idx, second_idx = int(descending[0]), int(descending[1])
        head_cap = max(1, int(np.floor(counts[second_idx] * args.balance_head_ratio)))
        targets[head_idx] = min(int(counts[head_idx]), head_cap)

    strong_labels: set[int] = set()
    for idx, count in enumerate(counts):
        if args.balance_min_source_count <= count < args.balance_tail_floor:
            targets[idx] = args.balance_tail_floor
            strong_labels.add(idx)
    return targets, strong_labels


def hybrid_balance_summary(
    labels: list[int],
    label_names: dict[int, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64))
    targets, strong_labels = hybrid_target_counts(labels, args)
    return {
        "mode": "hybrid",
        "original_class_counts": {label_names[idx]: int(count) for idx, count in enumerate(counts)},
        "effective_class_counts_per_epoch": {
            label_names[idx]: int(count) for idx, count in enumerate(targets)
        },
        "strong_augmentation_classes": [label_names[idx] for idx in sorted(strong_labels)],
        "effective_rows_per_epoch": int(targets.sum()),
    }


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_to_idx: dict[str, int],
    metadata_spec: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    train_transform, eval_transform = make_transforms(args.image_size)
    label_names = {idx: label for label, idx in label_to_idx.items()}
    train_labels = [label_to_idx[label] for label in train_df["label"].tolist()]
    sampler = None
    strong_transform = None
    strong_labels: set[int] = set()
    if args.balance_mode == "hybrid":
        targets, strong_labels = hybrid_target_counts(train_labels, args)
        sampler = HybridEpochSampler(train_labels, targets, args.seed, label_names)
        strong_transform = make_strong_train_transform(args.image_size)

    train_ds = PairedMilk10kMetadataDataset(
        train_df,
        label_to_idx,
        metadata_spec,
        train_transform,
        strong_transform=strong_transform,
        strong_augment_labels=strong_labels,
    )
    val_ds = PairedMilk10kMetadataDataset(val_df, label_to_idx, metadata_spec, eval_transform)
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if args.weighted_sampler:
        sampler = build_weighted_sampler(train_ds, args)
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
