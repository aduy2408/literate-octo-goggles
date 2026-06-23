#!/usr/bin/env python3
"""Supervised HAM10000 pretraining with reusable backbone checkpoints."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REQUIRED_COLUMNS = {"image_id", "lesion_id", "dx"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain a reusable image backbone on the seven HAM10000 classes."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--metadata-file", default="HAM10000_metadata.csv")
    parser.add_argument("--backbone", default="efficientnet_b2")
    parser.add_argument(
        "--backend", choices=("torchvision", "timm", "custom"), default="torchvision"
    )
    parser.add_argument("--custom-module", help="Dotted module name or path to a .py file.")
    parser.add_argument("--custom-factory", default="build_backbone")
    parser.add_argument(
        "--pretrained", action=argparse.BooleanOptionalAction, default=True,
        help="Use ImageNet weights for torchvision/timm or pass pretrained=True to a custom factory.",
    )
    parser.add_argument("--img-size", type=int, default=260)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--early-stopping", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--class-weighted", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("ham10k/runs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=torch.cuda.is_available()
    )
    args = parser.parse_args(argv)

    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val-ratio must be between 0 and 1.")
    if args.backend == "custom" and not args.custom_module:
        parser.error("--custom-module is required when --backend=custom.")
    if args.epochs < 1 or args.batch_size < 1 or args.img_size < 1:
        parser.error("--epochs, --batch-size, and --img-size must be positive.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_metadata(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(
            f"Metadata is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        counts = df[list(REQUIRED_COLUMNS)].isna().sum()
        raise ValueError(f"Required metadata columns contain null values: {counts.to_dict()}")

    lesion_label_counts = df.groupby("lesion_id")["dx"].nunique()
    conflicts = lesion_label_counts[lesion_label_counts > 1]
    if not conflicts.empty:
        raise ValueError(
            "A lesion_id must have exactly one dx label. Conflicting lesion IDs: "
            + ", ".join(map(str, conflicts.index[:10]))
        )


def index_images(data_root: Path) -> dict[str, Path]:
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist or is not a directory: {data_root}")

    by_id: dict[str, list[Path]] = {}
    for path in data_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            by_id.setdefault(path.stem, []).append(path.resolve())

    duplicates = {key: paths for key, paths in by_id.items() if len(paths) > 1}
    if duplicates:
        examples = "; ".join(
            f"{key}: {[str(p) for p in paths[:3]]}"
            for key, paths in list(duplicates.items())[:10]
        )
        raise ValueError(f"Duplicate image IDs found below {data_root}: {examples}")
    return {key: paths[0] for key, paths in by_id.items()}


def attach_image_paths(df: pd.DataFrame, image_index: dict[str, Path]) -> pd.DataFrame:
    result = df.copy()
    result["image_id"] = result["image_id"].astype(str).str.strip().map(lambda x: Path(x).stem)
    missing = sorted(set(result["image_id"]).difference(image_index))
    if missing:
        preview = ", ".join(missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        raise FileNotFoundError(
            f"Could not find {len(missing)} metadata images below the data root: {preview}{suffix}"
        )
    result["image_path"] = result["image_id"].map(lambda image_id: str(image_index[image_id]))
    return result


def split_by_lesion(
    df: pd.DataFrame, val_ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lesion_table = df.groupby("lesion_id", as_index=False).agg(dx=("dx", "first"))
    class_group_counts = lesion_table["dx"].value_counts()
    too_small = class_group_counts[class_group_counts < 2]
    if not too_small.empty:
        raise ValueError(
            "Group-stratified split requires at least two lesions per class; got "
            + str(too_small.to_dict())
        )
    try:
        train_lesions, val_lesions = train_test_split(
            lesion_table["lesion_id"],
            test_size=val_ratio,
            random_state=seed,
            stratify=lesion_table["dx"],
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create a lesion-stratified split. Increase the dataset size or "
            "adjust --val-ratio so every class is represented in both splits."
        ) from exc

    train_ids, val_ids = set(train_lesions), set(val_lesions)
    if train_ids.intersection(val_ids):
        raise RuntimeError("Internal error: lesion leakage detected between train and validation.")
    train_df = df[df["lesion_id"].isin(train_ids)].copy()
    val_df = df[df["lesion_id"].isin(val_ids)].copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def build_transforms(img_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    resize_size = int(round(img_size / 0.875))
    val_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


class Ham10kDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, class_to_idx: dict[str, int], transform: Callable[[Image.Image], Any]
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[index]
        try:
            with Image.open(row["image_path"]) as image:
                tensor = self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {row['image_path']}") from exc
        return tensor, self.class_to_idx[str(row["dx"])]


def _replace_last_linear(sequence: nn.Sequential) -> tuple[nn.Sequential, int] | None:
    modules = list(sequence.children())
    for index in range(len(modules) - 1, -1, -1):
        if isinstance(modules[index], nn.Linear):
            feature_dim = modules[index].in_features
            modules[index] = nn.Identity()
            return nn.Sequential(*modules), feature_dim
    return None


def _remove_torchvision_head(model: nn.Module) -> int:
    for attribute in ("fc", "head"):
        layer = getattr(model, attribute, None)
        if isinstance(layer, nn.Linear):
            setattr(model, attribute, nn.Identity())
            return layer.in_features

    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Linear):
        model.classifier = nn.Identity()
        return classifier.in_features
    if isinstance(classifier, nn.Sequential):
        replacement = _replace_last_linear(classifier)
        if replacement is not None:
            model.classifier, feature_dim = replacement
            return feature_dim

    heads = getattr(model, "heads", None)
    if isinstance(heads, nn.Linear):
        model.heads = nn.Identity()
        return heads.in_features
    if isinstance(heads, nn.Sequential):
        replacement = _replace_last_linear(heads)
        if replacement is not None:
            model.heads, feature_dim = replacement
            return feature_dim
    raise ValueError(
        "Could not identify the classification head for this torchvision model. "
        "Use --backend=custom and provide a factory for this architecture."
    )


class FeatureEncoder(nn.Module):
    """Normalize common encoder outputs to one [batch, feature_dim] tensor."""

    def __init__(self, model: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.model = model
        self.feature_dim = int(feature_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        if not isinstance(features, torch.Tensor):
            raise TypeError("Backbone forward must return a torch.Tensor.")
        if features.ndim > 2:
            features = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(features, 1), 1)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                "Backbone must produce [batch, feature_dim]; got "
                f"{tuple(features.shape)}, expected feature_dim={self.feature_dim}."
            )
        return features


def _load_custom_module(module_spec: str):
    candidate = Path(module_spec).expanduser()
    if candidate.suffix == ".py" or candidate.exists():
        if not candidate.is_file():
            raise FileNotFoundError(f"Custom module file does not exist: {candidate}")
        spec = importlib.util.spec_from_file_location("ham10k_custom_backbone", candidate.resolve())
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import custom module from {candidate}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_spec)


def build_encoder(args: argparse.Namespace) -> tuple[FeatureEncoder, int]:
    if args.backend == "timm":
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("The timm backend requires `pip install timm`.") from exc
        model = timm.create_model(
            args.backbone, pretrained=args.pretrained, num_classes=0, global_pool="avg"
        )
        feature_dim = int(model.num_features)
    elif args.backend == "torchvision":
        import torchvision.models as tv_models

        try:
            weights = tv_models.get_model_weights(args.backbone).DEFAULT if args.pretrained else None
            model = tv_models.get_model(args.backbone, weights=weights)
        except Exception as exc:
            raise ValueError(
                f"Could not build torchvision backbone {args.backbone!r}. "
                "Use a valid torchvision model name or another backend."
            ) from exc
        feature_dim = _remove_torchvision_head(model)
    else:
        module = _load_custom_module(args.custom_module)
        factory = getattr(module, args.custom_factory, None)
        if not callable(factory):
            raise AttributeError(
                f"Custom module {args.custom_module!r} has no callable "
                f"{args.custom_factory!r}."
            )
        result = factory(pretrained=args.pretrained)
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("Custom factory must return (nn.Module, feature_dim).")
        model, feature_dim = result
        if not isinstance(model, nn.Module) or not isinstance(feature_dim, int) or feature_dim < 1:
            raise TypeError("Custom factory must return (nn.Module, positive int feature_dim).")

    return FeatureEncoder(model, feature_dim), feature_dim


class Ham10kClassifier(nn.Module):
    def __init__(self, encoder: FeatureEncoder, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(images))


def make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def classification_metrics(targets: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "precision_macro": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(targets, predictions, average="macro", zero_division=0)),
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
) -> tuple[float, dict[str, float], list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []

    context = torch.enable_grad if training else torch.no_grad
    with context():
        progress = tqdm(loader, desc="train" if training else "val", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.item()) * labels.size(0)
            batch_predictions = logits.argmax(dim=1)
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(batch_predictions.detach().cpu().tolist())
            progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset), classification_metrics(targets, predictions), targets, predictions


def save_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def checkpoint_payload(
    model: Ham10kClassifier,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    best_f1: float,
    patience_counter: int,
    args: argparse.Namespace,
    class_to_idx: dict[str, int],
    feature_dim: int,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "best_f1_macro": best_f1,
        "patience_counter": patience_counter,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "class_to_idx": class_to_idx,
        "feature_dim": feature_dim,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }


def save_best_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    model: Ham10kClassifier,
    args: argparse.Namespace,
    class_to_idx: dict[str, int],
    feature_dim: int,
    targets: list[int],
    predictions: list[int],
) -> None:
    save_checkpoint(payload, output_dir / "best_full.pth")
    save_checkpoint(
        {
            "encoder_state_dict": model.encoder.model.state_dict(),
            "backbone": args.backbone,
            "backend": args.backend,
            "custom_module": args.custom_module,
            "custom_factory": args.custom_factory,
            "feature_dim": feature_dim,
            "class_to_idx": class_to_idx,
            "preprocessing": {
                "image_size": args.img_size,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "validation_resize": int(round(args.img_size / 0.875)),
            },
        },
        output_dir / "best_backbone.pth",
    )
    labels = list(range(len(class_to_idx)))
    idx_to_class = {index: name for name, index in class_to_idx.items()}
    matrix = confusion_matrix(targets, predictions, labels=labels)
    names = [idx_to_class[index] for index in labels]
    pd.DataFrame(matrix, index=names, columns=names).to_csv(output_dir / "confusion_matrix.csv")
    report = classification_report(
        targets, predictions, labels=labels, target_names=names, zero_division=0, output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv(output_dir / "classification_report.csv")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    data_root = args.data_root.expanduser().resolve()
    metadata_path = Path(args.metadata_file).expanduser()
    if not metadata_path.is_absolute():
        metadata_path = data_root / metadata_path
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    df = pd.read_csv(metadata_path)
    validate_metadata(df)
    df = attach_image_paths(df, index_images(data_root))
    train_df, val_df = split_by_lesion(df, args.val_ratio, args.seed)
    classes = sorted(df["dx"].astype(str).unique())
    class_to_idx = {name: index for index, name in enumerate(classes)}
    if len(classes) != 7:
        print(f"Warning: expected seven HAM10000 classes, found {len(classes)}: {classes}")

    output_dir = args.output_dir.expanduser().resolve() / f"{args.backend}_{args.backbone}"
    output_dir.mkdir(parents=True, exist_ok=True)
    train_export = train_df.copy().assign(split="train")
    val_export = val_df.copy().assign(split="val")
    pd.concat([train_export, val_export], ignore_index=True).to_csv(output_dir / "splits.csv", index=False)
    save_json(
        {
            "metadata": str(metadata_path),
            "train_images": len(train_df),
            "val_images": len(val_df),
            "train_lesions": int(train_df["lesion_id"].nunique()),
            "val_lesions": int(val_df["lesion_id"].nunique()),
            "class_to_idx": class_to_idx,
            "train_class_counts": train_df["dx"].value_counts().to_dict(),
            "val_class_counts": val_df["dx"].value_counts().to_dict(),
        },
        output_dir / "dataset_summary.json",
    )

    train_transform, val_transform = build_transforms(args.img_size)
    train_dataset = Ham10kDataset(train_df, class_to_idx, train_transform)
    val_dataset = Ham10kDataset(val_df, class_to_idx, val_transform)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    train_loader = make_loader(
        train_dataset, args.batch_size, args.num_workers, True, args.seed, device.type == "cuda"
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, args.num_workers, False, args.seed, device.type == "cuda"
    )

    encoder, feature_dim = build_encoder(args)
    model = Ham10kClassifier(encoder, feature_dim, len(classes)).to(device)
    model.encoder.eval()
    with torch.no_grad():
        probe = torch.zeros(1, 3, args.img_size, args.img_size, device=device)
        model.encoder(probe)

    train_labels = train_df["dx"].astype(str).map(class_to_idx).to_numpy()
    loss_weights = None
    if args.class_weighted:
        weights = compute_class_weight(
            class_weight="balanced", classes=np.arange(len(classes)), y=train_labels
        )
        loss_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 1
    best_f1 = -math.inf
    patience_counter = 0
    history: list[dict[str, Any]] = []
    if args.resume:
        resume_path = args.resume.expanduser().resolve()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("class_to_idx") != class_to_idx:
            raise ValueError("Resume checkpoint class_to_idx does not match the current metadata.")
        if int(checkpoint.get("feature_dim", -1)) != feature_dim:
            raise ValueError("Resume checkpoint feature_dim does not match the current backbone.")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint.get("best_f1_macro", best_f1))
        patience_counter = int(checkpoint.get("patience_counter", 0))
        history_path = output_dir / "history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        print(f"Resumed {resume_path} at epoch {start_epoch}; best macro-F1={best_f1:.4f}")

    print(
        f"Training {args.backend}:{args.backbone} on {len(train_df)} images; "
        f"validating on {len(val_df)} images; classes={classes}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_metrics, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, amp_enabled
        )
        val_loss, val_metrics, targets, predictions = run_epoch(
            model, val_loader, criterion, device, None, None, amp_enabled
        )
        scheduler.step(val_metrics["f1_macro"])
        improved = val_metrics["f1_macro"] > best_f1 + args.min_delta
        if improved:
            best_f1 = val_metrics["f1_macro"]
            patience_counter = 0
        else:
            patience_counter += 1

        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(record)
        save_json(history, output_dir / "history.json")
        payload = checkpoint_payload(
            model, optimizer, scheduler, epoch, best_f1, patience_counter,
            args, class_to_idx, feature_dim,
        )
        save_checkpoint(payload, output_dir / "last_full.pth")
        if improved:
            save_best_outputs(
                output_dir, payload, model, args, class_to_idx, feature_dim, targets, predictions
            )

        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_macro_f1={val_metrics['f1_macro']:.4f}, best={best_f1:.4f}"
        )
        if args.early_stopping and patience_counter >= args.patience:
            print(f"Early stopping after {patience_counter} epochs without improvement.")
            break

    print(f"Finished. Best validation macro-F1: {best_f1:.4f}")
    print(f"Reusable backbone: {output_dir / 'best_backbone.pth'}")


if __name__ == "__main__":
    main()
