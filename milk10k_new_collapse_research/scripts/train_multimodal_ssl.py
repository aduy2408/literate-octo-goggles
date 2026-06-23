"""Contrastively pretrain a dual-use encoder on clinical/dermoscopic MILK10k pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from timm.data import create_transform, resolve_data_config
from torch import nn
from torch.utils.data import DataLoader, Dataset

from milk10k_new_collapse_research.compat import ensure_legacy_package_path
from milk10k_new_collapse_research.config import CLASS_NAMES, RESULTS_ROOT
from milk10k_new_collapse_research.metrics_ext import write_standard_outputs

ensure_legacy_package_path()
from milk10k_effb2_metadata.data import lesion_split, load_paired_dataframe


class SslPairDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform) -> None:
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
            "clinical": self._load(row["clinical_path"]),
            "dermoscopic": self._load(row["dermoscopic_path"]),
            "label": str(row["label"]),
            "lesion_id": str(row["lesion_id"]),
        }


class ContrastiveEncoder(nn.Module):
    def __init__(self, backbone_name: str, projection_dim: int, pretrained: bool) -> None:
        super().__init__()
        import timm

        self.encoder = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        feature_dim = int(self.encoder.num_features)
        self.projector = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, projection_dim),
        )

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projector(self.forward_features(images)), dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "multimodal_ssl")
    parser.add_argument("--backbone", default="convnext_base")
    parser.add_argument("--imagenet-pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Optional per-split row cap for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_paired_dataframe(args.data_dir)
    train_df, val_df = lesion_split(df, args.val_size, args.seed)
    if args.limit > 0:
        train_df = train_df.head(args.limit).copy()
        val_df = val_df.head(args.limit).copy()

    model = ContrastiveEncoder(args.backbone, args.projection_dim, args.imagenet_pretrained).to(device)
    train_transform, eval_transform = make_transforms(model.encoder, args.image_size)
    train_loader = DataLoader(
        SslPairDataset(train_df, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            clinical = batch["clinical"].to(device, non_blocking=True)
            dermoscopic = batch["dermoscopic"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            z1 = model(clinical)
            z2 = model(dermoscopic)
            loss = symmetric_clip_loss(z1, z2, args.temperature)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append({"epoch": epoch, "ssl_loss": float(np.mean(losses)) if losses else None})
        print(f"epoch={epoch} ssl_loss={history[-1]['ssl_loss']}")

    torch.save(
        {
            "backbone": args.backbone,
            "model_state": model.state_dict(),
            "projection_dim": args.projection_dim,
            "image_size": args.image_size,
            "history": history,
        },
        args.output_dir / "ssl_encoder.pt",
    )

    x_train, y_train, class_names = extract_features_for_probe(model, train_df, eval_transform, args, device)
    x_val, y_val, _ = extract_features_for_probe(model, val_df, eval_transform, args, device, class_names=class_names)
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=args.seed),
    )
    probe.fit(x_train, y_train)
    y_prob = align_probabilities(probe.predict_proba(x_val), probe.classes_, len(class_names))
    metrics = write_standard_outputs(
        args.output_dir,
        val_df,
        y_val,
        y_prob,
        class_names,
        extra={"experiment": "multimodal_ssl_linear_eval", "ssl_history": history},
    )
    (args.output_dir / "ssl_config.json").write_text(
        json.dumps({"backbone": args.backbone, "metrics_f1_macro": metrics["f1_macro"], "history": history}, indent=2),
        encoding="utf-8",
    )


def make_transforms(model: nn.Module, image_size: int):
    cfg = resolve_data_config({}, model=model)
    cfg["input_size"] = (3, image_size, image_size)
    train_transform = create_transform(**cfg, is_training=True)
    eval_transform = create_transform(**cfg, is_training=False)
    return train_transform, eval_transform


def symmetric_clip_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


@torch.no_grad()
def extract_features_for_probe(
    model: ContrastiveEncoder,
    df: pd.DataFrame,
    transform,
    args: argparse.Namespace,
    device: torch.device,
    class_names: list[str] | None = None,
):
    dataset = SslPairDataset(df, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model.eval()
    features = []
    labels = []
    for batch in loader:
        clinical = batch["clinical"].to(device, non_blocking=True)
        dermoscopic = batch["dermoscopic"].to(device, non_blocking=True)
        c = F.normalize(model.forward_features(clinical), dim=1)
        d = F.normalize(model.forward_features(dermoscopic), dim=1)
        pair = torch.cat([c, d, torch.abs(c - d), c * d], dim=1)
        features.append(pair.cpu().numpy())
        labels.extend(str(label) for label in batch["label"])
    if class_names is None:
        observed = sorted(set(labels))
        class_names = [label for label in CLASS_NAMES if label in observed] + [label for label in observed if label not in CLASS_NAMES]
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    y = np.asarray([label_to_idx[label] for label in labels], dtype=np.int64)
    return np.concatenate(features, axis=0), y, class_names


def align_probabilities(y_prob: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
    aligned = np.zeros((y_prob.shape[0], n_classes), dtype=np.float64)
    for src_idx, class_idx in enumerate(classes):
        aligned[:, int(class_idx)] = y_prob[:, src_idx]
    row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / np.clip(row_sum, 1e-12, None)


if __name__ == "__main__":
    main()

