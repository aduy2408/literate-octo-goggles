"""Dermoscopic-only dataframe, metadata, split, transform, and loader helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

LABEL_COLUMNS = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
BASE_METADATA_COLUMNS = ("age_approx", "sex", "skin_tone_class", "site")


def normalize_image_type(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace(":", "").replace("-", "_")


def resolve_training_paths(data_dir: Path, input_dir: Path | None = None) -> tuple[Path, Path, Path]:
    data_dir = data_dir.expanduser().resolve()
    if input_dir is None:
        local_input = data_dir / "MILK10k_Training_Input"
        sibling_input = data_dir.parent / "MILK10k_Training_Input"
        input_dir = local_input if local_input.exists() else sibling_input
    else:
        input_dir = input_dir.expanduser().resolve()
    metadata_csv = data_dir / "MILK10k_Training_Metadata.csv"
    groundtruth_csv = data_dir / "MILK10k_Training_GroundTruth.csv"
    missing = [path for path in (input_dir, metadata_csv, groundtruth_csv) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing MILK10k training input: " + ", ".join(map(str, missing)))
    return input_dir, metadata_csv, groundtruth_csv


def resolve_monet_columns(meta: pd.DataFrame) -> list[str]:
    return sorted(column for column in meta.columns if column.startswith("MONET_"))


def load_dermoscopic_dataframe(data_dir: Path, input_dir: Path | None = None) -> pd.DataFrame:
    input_dir, metadata_csv, groundtruth_csv = resolve_training_paths(data_dir, input_dir)
    meta = pd.read_csv(metadata_csv)
    gt = pd.read_csv(groundtruth_csv)
    required = {"lesion_id", "isic_id", "image_type", *BASE_METADATA_COLUMNS}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing columns: {sorted(missing)}")
    label_columns = [column for column in LABEL_COLUMNS if column in gt.columns]
    if not label_columns:
        raise ValueError("Ground-truth CSV contains no recognized class columns.")

    meta["image_type_norm"] = meta["image_type"].map(normalize_image_type)
    dermoscopic = meta[meta["image_type_norm"] == "dermoscopic"].copy()
    duplicate_counts = dermoscopic.groupby("lesion_id").size()
    duplicates = duplicate_counts[duplicate_counts > 1]
    if not duplicates.empty:
        sample = duplicates.head(5).to_dict()
        raise ValueError(f"Expected one dermoscopic image per lesion; duplicates found: {sample}")
    dermoscopic["image_path"] = dermoscopic.apply(
        lambda row: input_dir / str(row["lesion_id"]) / f"{row['isic_id']}.jpg", axis=1
    )
    dermoscopic = dermoscopic[dermoscopic["image_path"].map(Path.exists)].copy()
    dermoscopic["image_path"] = dermoscopic["image_path"].map(str)

    gt = gt.copy()
    gt["label"] = gt[label_columns].idxmax(axis=1)
    df = gt[["lesion_id", "label"]].merge(dermoscopic, on="lesion_id", how="inner", validate="one_to_one")
    if df.empty:
        raise ValueError(f"No labeled dermoscopic images found under {input_dir}")
    return df.reset_index(drop=True)


def fit_metadata_spec(train_df: pd.DataFrame) -> dict[str, Any]:
    def categories(column: str) -> list[str]:
        values = train_df[column].fillna("unknown").astype(str).str.strip().replace("", "unknown")
        return sorted(set(values.tolist()) | {"unknown"})

    return {
        "sex_values": categories("sex"),
        "site_values": categories("site"),
        "monet_columns": resolve_monet_columns(train_df),
    }


def metadata_vector(row: pd.Series, spec: dict[str, Any]) -> np.ndarray:
    age = pd.to_numeric(row.get("age_approx"), errors="coerce")
    skin = pd.to_numeric(row.get("skin_tone_class"), errors="coerce")
    sex = str(row.get("sex", "unknown")).strip() if pd.notna(row.get("sex")) else "unknown"
    site = str(row.get("site", "unknown")).strip() if pd.notna(row.get("site")) else "unknown"
    sex = sex or "unknown"
    site = site or "unknown"
    values = [0.0 if pd.isna(age) else float(age) / 100.0, 0.0 if pd.isna(skin) else float(skin) / 6.0]
    values.extend(float(sex == item) for item in spec["sex_values"])
    values.extend(float(site == item) for item in spec["site_values"])
    for column in spec.get("monet_columns", []):
        value = pd.to_numeric(row.get(column), errors="coerce")
        values.append(0.0 if pd.isna(value) else float(value))
    return np.asarray(values, dtype=np.float32)


def synthetic_mask(df: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(df), dtype=bool)
    if "is_augmented" in df:
        mask |= df["is_augmented"].fillna(False).astype(bool).to_numpy()
    mask |= df["lesion_id"].astype(str).str.contains("__sdpair_", regex=False).to_numpy()
    return mask


def source_lesion_id(value: Any) -> str:
    return str(value).split("__sdpair_", 1)[0]


def create_or_load_split(
    df: pd.DataFrame, manifest: Path, val_size: float, seed: int,
    synthetic_train_only: bool = False, fold_index: int = 0, k_folds: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = manifest.expanduser().resolve()
    all_ids = set(df["lesion_id"].astype(str))
    if manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if "folds" in payload:
            if int(payload.get("k_folds", 1)) != k_folds:
                raise ValueError("Split manifest k_folds does not match --k-folds.")
            if bool(payload.get("synthetic_train_only", False)) != synthetic_train_only:
                raise ValueError("Split manifest synthetic_train_only does not match current CLI; use a separate manifest.")
            try: selected = payload["folds"][fold_index]
            except IndexError as exc: raise ValueError(f"Split manifest has no fold {fold_index}.") from exc
            train_ids = set(map(str, selected["train_lesion_ids"])); val_ids = set(map(str, selected["val_lesion_ids"]))
        else:  # v1 manifest compatibility
            if k_folds != 1: raise ValueError("Legacy split manifest cannot be used with k-fold training.")
            train_ids = set(map(str, payload["train_lesion_ids"])); val_ids = set(map(str, payload["val_lesion_ids"]))
            if synthetic_train_only and any("__sdpair_" in item for item in val_ids):
                raise ValueError("Legacy manifest contains synthetic validation IDs; remove it to create a safe v2 manifest.")
        if train_ids & val_ids:
            raise ValueError(f"Split manifest has overlapping train/validation IDs: {manifest}")
        unknown = (train_ids | val_ids) - all_ids
        missing = all_ids - (train_ids | val_ids)
        allowed_missing = set()
        if synthetic_train_only:
            allowed_missing = {
                lesion_id for lesion_id in missing
                if "__sdpair_" in lesion_id and source_lesion_id(lesion_id) in val_ids
            }
        unexpected_missing = missing - allowed_missing
        if unknown or unexpected_missing:
            raise ValueError(f"Split manifest does not match dataset (unknown={len(unknown)}, missing={len(missing)}).")
    else:
        synthetic = synthetic_mask(df)
        base = df.loc[~synthetic].copy() if synthetic_train_only else df.copy()
        folds = []
        if k_folds == 1:
            train_rows, val_rows = train_test_split(base, test_size=val_size, stratify=base["label"], random_state=seed)
            pairs = [(train_rows, val_rows)]
        else:
            if k_folds < 2: raise ValueError("--k-folds must be 1 or >=2.")
            minimum = int(base["label"].value_counts().min())
            if k_folds > minimum: raise ValueError(f"--k-folds={k_folds} exceeds smallest class count={minimum}.")
            splitter = StratifiedKFold(k_folds, shuffle=True, random_state=seed)
            pairs = [(base.iloc[tr], base.iloc[va]) for tr, va in splitter.split(base, base["label"])]
        for train_rows, val_rows in pairs:
            train_real_ids = set(train_rows["lesion_id"].astype(str))
            val_real_ids = set(val_rows["lesion_id"].astype(str))
            extra_train_ids = set()
            excluded_synthetic_ids = set()
            if synthetic_train_only:
                for lesion_id in df.loc[synthetic, "lesion_id"].astype(str):
                    source_id = source_lesion_id(lesion_id)
                    if source_id in train_real_ids:
                        extra_train_ids.add(lesion_id)
                    elif source_id in val_real_ids:
                        excluded_synthetic_ids.add(lesion_id)
                    else:
                        raise ValueError(f"Synthetic lesion has unknown source ID: {lesion_id}")
            folds.append({
                "train_lesion_ids": sorted(set(train_rows["lesion_id"].astype(str)) | extra_train_ids),
                "val_lesion_ids": sorted(set(val_rows["lesion_id"].astype(str))),
                "excluded_synthetic_lesion_ids": sorted(excluded_synthetic_ids),
            })
        train_ids = set(folds[fold_index]["train_lesion_ids"]); val_ids = set(folds[fold_index]["val_lesion_ids"])
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 2, "seed": seed, "val_size": val_size, "k_folds": k_folds,
                    "synthetic_train_only": synthetic_train_only, "folds": folds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    train_df = df[df["lesion_id"].astype(str).isin(train_ids)].copy()
    val_df = df[df["lesion_id"].astype(str).isin(val_ids)].copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def append_augmented_rows(base_df: pd.DataFrame, train_df: pd.DataFrame, args) -> pd.DataFrame:
    if args.augmented_data_dir is None: return train_df
    augmented = load_dermoscopic_dataframe(args.augmented_data_dir)
    augmented = augmented[~augmented["lesion_id"].astype(str).isin(set(base_df["lesion_id"].astype(str)))].copy()
    train_source_ids = set(train_df["lesion_id"].astype(str).map(source_lesion_id))
    base_source_ids = set(base_df["lesion_id"].astype(str).map(source_lesion_id))
    augmented["source_lesion_id"] = augmented["lesion_id"].astype(str).map(source_lesion_id)
    unknown = ~augmented["source_lesion_id"].isin(base_source_ids)
    if unknown.any():
        examples = augmented.loc[unknown, "lesion_id"].astype(str).head(5).tolist()
        raise ValueError(f"Augmented lesions have unknown source IDs. Examples: {examples}")
    augmented = augmented[augmented["source_lesion_id"].isin(train_source_ids)].copy()
    if args.augmented_classes:
        allowed = {name.upper() for name in args.augmented_classes}
        unknown = allowed - {name.upper() for name in base_df["label"].unique()}
        if unknown: raise ValueError(f"Unknown augmented classes: {sorted(unknown)}")
        augmented = augmented[augmented["label"].str.upper().isin(allowed)]
    if args.augmented_max_per_class < 0: raise ValueError("--augmented-max-per-class must be >=0.")
    if args.augmented_max_per_class:
        augmented = augmented.sample(frac=1, random_state=args.seed).groupby("label", group_keys=False).head(args.augmented_max_per_class)
    augmented["is_augmented"] = True; augmented["ignore_metadata"] = bool(args.zero_augmented_metadata)
    return pd.concat([train_df, augmented], ignore_index=True, sort=False)


def make_transforms(image_size: int):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize(round(image_size * 1.12)), transforms.CenterCrop(image_size), transforms.ToTensor(), normalize]
    )
    return train_transform, eval_transform


class DermoscopicMetadataDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict[str, int] | None, metadata_spec: dict[str, Any], transform=None):
        self.df = df.reset_index(drop=True)
        self.labels = None if label_to_idx is None or "label" not in df else [label_to_idx[x] for x in self.df["label"]]
        self.metadata = np.stack([metadata_vector(row, metadata_spec) for _, row in self.df.iterrows()])
        if "ignore_metadata" in self.df:
            self.metadata[self.df["ignore_metadata"].fillna(False).astype(bool).to_numpy()] = 0
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        if self.transform:
            image = self.transform(image)
        item = {"image": image, "metadata": torch.from_numpy(self.metadata[index])}
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def make_loaders(train_df, val_df, label_to_idx, metadata_spec, args):
    train_transform, eval_transform = make_transforms(args.image_size)
    train_ds = DermoscopicMetadataDataset(train_df, label_to_idx, metadata_spec, train_transform)
    val_ds = DermoscopicMetadataDataset(val_df, label_to_idx, metadata_spec, eval_transform)
    sampler = None
    if args.weighted_sampler:
        labels = np.asarray(train_ds.labels)
        counts = np.bincount(labels, minlength=len(label_to_idx))
        if np.any(counts == 0):
            raise ValueError("Cannot use weighted sampler with an empty training class.")
        weights = (1.0 / np.power(counts.astype(np.float64), args.sampler_power))[labels]
        generator = torch.Generator().manual_seed(args.seed)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(labels), True, generator=generator)
    common = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, shuffle=sampler is None, sampler=sampler, **common),
        DataLoader(val_ds, shuffle=False, **common),
    )
