"""Paired MILK10k dataframe, metadata, and dataloader helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from timm.data import create_transform, resolve_data_config
from torch.utils.data import DataLoader, Dataset

from datasets import LABEL_COLUMNS, normalize_image_type
from milk10k_dual_encoder.config import MONET_COLUMNS


def load_paired_dataframe_with_extras(data_dir: Path) -> pd.DataFrame:
    input_dir = data_dir / "MILK10k_Training_Input"
    gt = pd.read_csv(data_dir / "MILK10k_Training_GroundTruth.csv")
    meta = pd.read_csv(data_dir / "MILK10k_Training_Metadata.csv")
    gt["label"] = gt[LABEL_COLUMNS].idxmax(axis=1)
    meta["image_type_norm"] = meta["image_type"].map(normalize_image_type)
    meta["path"] = meta.apply(lambda r: input_dir / r["lesion_id"] / f"{r['isic_id']}.jpg", axis=1)
    meta = meta[meta["path"].map(lambda p: p.exists())].copy()
    meta["path"] = meta["path"].map(str)

    clinical = meta[meta["image_type_norm"] == "clinical_close_up"].drop_duplicates("lesion_id")
    dermoscopic = meta[meta["image_type_norm"] == "dermoscopic"].drop_duplicates("lesion_id")
    keep = ["lesion_id", "path", "age_approx", "sex", "skin_tone_class", "site", *MONET_COLUMNS]
    paired = (
        gt[["lesion_id", "label"]]
        .merge(clinical[keep].add_prefix("clinical_"), left_on="lesion_id", right_on="clinical_lesion_id")
        .merge(dermoscopic[keep].add_prefix("dermoscopic_"), left_on="lesion_id", right_on="dermoscopic_lesion_id")
    )
    return paired.drop(columns=["clinical_lesion_id", "dermoscopic_lesion_id"])


def fit_metadata_spec(train_df: pd.DataFrame) -> dict[str, Any]:
    sex_values = sorted(
        set(train_df["clinical_sex"].fillna("unknown").astype(str))
        | set(train_df["dermoscopic_sex"].fillna("unknown").astype(str))
    )
    site_values = sorted(
        set(train_df["clinical_site"].fillna("unknown").astype(str))
        | set(train_df["dermoscopic_site"].fillna("unknown").astype(str))
    )
    return {"sex_values": sex_values, "site_values": site_values}


def metadata_vector(row: pd.Series, spec: dict[str, Any]) -> np.ndarray:
    values: list[float] = []
    for prefix in ("clinical", "dermoscopic"):
        age = pd.to_numeric(row.get(f"{prefix}_age_approx"), errors="coerce")
        tone = pd.to_numeric(row.get(f"{prefix}_skin_tone_class"), errors="coerce")
        values.append(0.0 if pd.isna(age) else float(age) / 100.0)
        values.append(0.0 if pd.isna(tone) else float(tone) / 6.0)
        sex = str(row.get(f"{prefix}_sex", "unknown"))
        site = str(row.get(f"{prefix}_site", "unknown"))
        values.extend(1.0 if sex == item else 0.0 for item in spec["sex_values"])
        values.extend(1.0 if site == item else 0.0 for item in spec["site_values"])
    return np.asarray(values, dtype=np.float32)


def attribute_vector(row: pd.Series) -> np.ndarray:
    values = []
    for col in MONET_COLUMNS:
        c = pd.to_numeric(row.get(f"clinical_{col}"), errors="coerce")
        d = pd.to_numeric(row.get(f"dermoscopic_{col}"), errors="coerce")
        values.append(float(np.nanmean([c, d])))
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)


class PairedExtrasMilk10kDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        label_to_idx: dict[str, int],
        transform=None,
        metadata_spec: dict[str, Any] | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.labels = [label_to_idx[label] for label in self.df["label"].tolist()]
        self.transform = transform
        self.metadata_spec = metadata_spec or {"sex_values": [], "site_values": []}
        self.metadata = np.stack([metadata_vector(row, self.metadata_spec) for _, row in self.df.iterrows()])
        self.attributes = np.stack([attribute_vector(row) for _, row in self.df.iterrows()])

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> torch.Tensor:
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        clinical = self._load_image(row["clinical_path"])
        dermoscopic = self._load_image(row["dermoscopic_path"])
        return {
            "images": torch.stack([clinical, dermoscopic], dim=0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "metadata": torch.from_numpy(self.metadata[idx]),
            "attributes": torch.from_numpy(self.attributes[idx]),
        }


def make_loaders(train_df, val_df, label_to_idx, model, image_size, metadata_spec, args):
    data_config = resolve_data_config({}, model=model.clinical_encoder)
    data_config["input_size"] = (3, image_size, image_size)
    train_transform = create_transform(**data_config, is_training=True)
    eval_transform = create_transform(**data_config, is_training=False)
    train_ds = PairedExtrasMilk10kDataset(train_df, label_to_idx, train_transform, metadata_spec)
    val_ds = PairedExtrasMilk10kDataset(val_df, label_to_idx, eval_transform, metadata_spec)
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return DataLoader(train_ds, shuffle=True, **common), DataLoader(val_ds, shuffle=False, **common)
