import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

try:
    from torchvision.models import (
        efficientnet_b0,
        efficientnet_b1,
        efficientnet_b2,
        mobilenet_v3_large,
        convnext_base,
        resnet50,
    )
except ImportError:
    efficientnet_b0 = efficientnet_b1 = efficientnet_b2 = None
    mobilenet_v3_large = convnext_base = resnet50 = None


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # older torch versions may not support this
        pass


class PadUefsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: Path, transform=None, image_col=None, label_col=None):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col

        self.df["image_path"] = self.df.apply(self._resolve_image_path, axis=1)
        self.classes = sorted(self.df[self.label_col].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.df["label_idx"] = self.df[self.label_col].map(self.class_to_idx)

    def _resolve_image_path(self, row):
        # Try direct path from metadata (may include subfolder or extension)
        if self.image_col and pd.notna(row[self.image_col]):
            raw = str(row[self.image_col]).strip()
            if raw != "":
                candidate = self.root / raw
                if candidate.exists():
                    return candidate

        # Fallback: resolve by image id or filename (strip extension when needed)
        raw_val = str(row[self.image_col] if self.image_col else row.get("image_id", "")).strip()
        if raw_val == "":
            raise ValueError("Could not determine image id for row: {}".format(row))

        p = Path(raw_val)
        base = p.stem

        exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

        # Try exact name in root
        for ext in exts:
            candidate = self.root / f"{base}{ext}"
            if candidate.exists():
                return candidate

        # If metadata already included extension (e.g., 'PAT_...png'), try to find that exact filename anywhere under root
        filename = p.name
        for candidate in self.root.rglob(filename):
            if candidate.is_file():
                return candidate

        # Last resort: search for base with any allowed extension recursively
        for candidate in self.root.rglob(f"{base}.*"):
            if candidate.suffix.lower() in exts and candidate.is_file():
                return candidate

        raise FileNotFoundError(f"Image not found for id={raw_val} in {self.root}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = torch.zeros(3, 1, 1)
        from PIL import Image

        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(row["label_idx"])
        return image, label


def detect_columns(df: pd.DataFrame):
    lower_cols = [c.lower() for c in df.columns]
    image_col = None
    label_col = None

    for col in ["image_id", "img_id", "image", "filename", "file_name", "file", "path"]:
        if col in lower_cols:
            image_col = df.columns[lower_cols.index(col)]
            break

    for col in ["sign", "label", "class", "target", "dx", "diagnosis", "diagnostic"]:
        if col in lower_cols:
            label_col = df.columns[lower_cols.index(col)]
            break

    if image_col is None and "image_id" in lower_cols:
        image_col = df.columns[lower_cols.index("image_id")]

    if label_col is None and "diagnostic" in lower_cols:
        label_col = df.columns[lower_cols.index("diagnostic")]

    if label_col is None:
        raise ValueError(
            "Cannot find label column in metadata. Supported names: sign, label, class, target, dx, diagnosis, diagnostic."
        )

    return image_col, label_col


def build_model(backbone: str, num_classes: int, backend: str = "torchvision", pretrained: bool = False):
    backbone = backbone.lower()
    backend = backend.lower()

    if backend == "timm":
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("The timm backend requires `pip install timm`.") from exc
        try:
            return timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=num_classes,
            )
        except Exception as exc:
            raise ValueError(f"Could not build timm backbone: {backbone}") from exc

    if backend != "torchvision":
        raise ValueError(f"Unsupported backbone backend: {backend}")

    weights = "DEFAULT" if pretrained else None
    if backbone == "resnet50":
        model = resnet50(weights=weights) if resnet50 else None
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if backbone in {"effnetb0", "effnet_b0", "efficientnetb0", "efficientnet_b0"}:
        return _build_efficientnet(efficientnet_b0, num_classes, weights)
    if backbone in {"effnetb1", "effnet_b1", "efficientnetb1", "efficientnet_b1"}:
        return _build_efficientnet(efficientnet_b1, num_classes, weights)
    if backbone in {"effnetb2", "effnet_b2", "efficientnetb2", "efficientnet_b2"}:
        return _build_efficientnet(efficientnet_b2, num_classes, weights)
    if backbone in {"mobilenet_v3_large", "mobilenetv3large", "mobilenetv3_large", "mobilenet large"}:
        model = mobilenet_v3_large(weights=weights) if mobilenet_v3_large else None
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if backbone in {"convnext_base", "convnextbase", "convnext base"}:
        model = convnext_base(weights=weights) if convnext_base else None
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model

    raise ValueError(f"Unsupported backbone: {backbone}")


def _build_efficientnet(builder, num_classes, weights=None):
    if builder is None:
        raise RuntimeError("EfficientNet builder not available in torchvision.")
    model = builder(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def prepare_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def make_data_loaders(
    df: pd.DataFrame,
    root: Path,
    image_col: str,
    label_col: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    img_size: int,
    weighted_sampler: bool,
):
    train_tf, val_tf = prepare_transforms(img_size)

    # Provide deterministic worker initialization and generator for DataLoader
    def _worker_init_fn(worker_id):
        worker_seed = int(seed) + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        try:
            torch.manual_seed(worker_seed)
        except Exception:
            pass

    generator = torch.Generator()
    try:
        generator.manual_seed(int(seed))
    except Exception:
        pass

    if "split" in [c.lower() for c in df.columns]:
        split_col = [c for c in df.columns if c.lower() == "split"][0]
        train_df = df[df[split_col].astype(str).str.lower().isin(["train", "training", "tr", "0"])].copy()
        val_df = df[df[split_col].astype(str).str.lower().isin(["val", "validation", "valid", "1"])].copy()
        if len(train_df) == 0 or len(val_df) == 0:
            train_df, val_df = train_test_split(
                df,
                test_size=val_ratio,
                stratify=df[label_col],
                random_state=seed,
            )
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            stratify=df[label_col],
            random_state=seed,
        )

    train_dataset = PadUefsDataset(train_df, root, transform=train_tf, image_col=image_col, label_col=label_col)
    val_dataset = PadUefsDataset(val_df, root, transform=val_tf, image_col=image_col, label_col=label_col)

    if weighted_sampler:
        counts = train_dataset.df["label_idx"].value_counts().sort_index().to_numpy()
        class_weights = 1.0 / counts
        sample_weights = train_dataset.df["label_idx"].map(lambda x: class_weights[x]).to_numpy()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            worker_init_fn=_worker_init_fn,
            generator=generator,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=_worker_init_fn,
            generator=generator,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn,
        generator=generator,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes


def compute_metrics(y_true, y_pred, average="macro"):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def training_step(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    iterator = loader
    if isinstance(loader, (list, tuple)) or hasattr(loader, '__len__'):
        iterator = tqdm(loader, desc="train", leave=False)

    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(labels.detach().cpu().numpy().tolist())

        if isinstance(iterator, tqdm):
            iterator.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds)
    return avg_loss, metrics


@torch.no_grad()
def validation_step(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    iterator = loader
    if isinstance(loader, (list, tuple)) or hasattr(loader, '__len__'):
        iterator = tqdm(loader, desc="val", leave=False)

    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(labels.detach().cpu().numpy().tolist())

        if isinstance(iterator, tqdm):
            iterator.set_postfix({'loss': f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds)
    return avg_loss, metrics, all_targets, all_preds


def save_checkpoint(state, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / filename)


def load_checkpoint(model, optimizer, path: Path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train PAD-UEFS-20 backbones with imbalance-aware logging")
    parser.add_argument("--data-root", type=str, required=True, help="Root path for PAD-UEFS-20 metadata and image files")
    parser.add_argument("--metadata-file", type=str, default="PAD-UFES-20_metadata.csv", help="Metadata CSV file name in data root")
    parser.add_argument("--image-col", type=str, default=None, help="Explicit image ID/path column in metadata")
    parser.add_argument("--label-col", type=str, default=None, help="Explicit label/diagnosis column in metadata")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone model name (for example, tf_efficientnetv2_b2 with timm)")
    parser.add_argument("--backend", choices=["torchvision", "timm"], default="torchvision", help="Backbone implementation")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False, help="Initialize the backbone with ImageNet pretrained weights")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume training")
    parser.add_argument("--output-dir", type=str, default="./runs", help="Base directory for run outputs")
    parser.add_argument("--use-weighted-sampler", action="store_true", help="Use weighted sampler for training loader to address class imbalance")
    parser.add_argument("--use-class-weights", action="store_true", help="Use class-balanced weights in CrossEntropyLoss")
    parser.add_argument("--early-stop", action="store_true", help="Enable early stopping by validation balanced accuracy")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=0.001, help="Minimum improvement threshold for early stopping")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-file", type=str, default="train_log.json", help="JSON log file for training metrics")

    args = parser.parse_args()
    run_name = args.backbone if args.backend == "torchvision" else f"{args.backend}_{args.backbone}"
    output_dir = Path(args.output_dir) / run_name
    args.output_dir = str(output_dir)
    set_seed(args.seed)

    root = Path(args.data_root)
    metadata_path = root / args.metadata_file
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    if args.image_col is not None and args.label_col is not None:
        image_col = args.image_col
        label_col = args.label_col
    else:
        image_col, label_col = detect_columns(df)
    print(f"Using image column: {image_col}, label column: {label_col}")

    train_loader, val_loader, classes = make_data_loaders(
        df=df,
        root=root,
        image_col=image_col,
        label_col=label_col,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
        img_size=args.img_size,
        weighted_sampler=args.use_weighted_sampler,
    )

    num_classes = len(classes)
    model = build_model(args.backbone, num_classes, args.backend, args.pretrained)
    if model is None:
        raise RuntimeError("Failed to build model. Verify the selected backend supports the backbone.")

    device = torch.device(args.device)
    model = model.to(device)

    class_weights = None
    if args.use_class_weights:
        labels = train_loader.dataset.df["label_idx"].to_numpy()
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    start_epoch = 1
    best_score = -math.inf
    history = []
    if args.resume:
        checkpoint = load_checkpoint(model, optimizer, Path(args.resume), device)
        start_epoch = checkpoint.get("epoch", 1) + 1
        best_score = checkpoint.get("best_score", best_score)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    best_checkpoint_path = output_dir / f"best_{args.backbone}.pth"
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_metrics = training_step(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, y_true, y_pred = validation_step(model, val_loader, criterion, device)

        # Choose monitoring metric: use F1 when early-stop requested, otherwise balanced accuracy
        monitor = "f1" if args.early_stop else "balanced_accuracy"
        monitor_value = float(val_metrics.get(monitor, val_metrics.get("balanced_accuracy", 0.0)))
        scheduler.step(monitor_value)

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "train_precision": train_metrics["precision"],
            "val_precision": val_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_recall": val_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
            "confusion_matrix": val_metrics["confusion_matrix"],
            "classification_report": classification_report(y_true, y_pred, target_names=[str(c) for c in classes], zero_division=0, output_dict=True),
        }

        history.append(epoch_result)
        # Ensure output directory exists before writing logs
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / args.log_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        print(f"Epoch {epoch}/{args.epochs}")
        print(f"  train loss: {train_loss:.4f}, accuracy: {train_metrics['accuracy']:.4f}, balanced acc: {train_metrics['balanced_accuracy']:.4f}, f1: {train_metrics['f1']:.4f}")
        print(f"  val   loss: {val_loss:.4f}, accuracy: {val_metrics['accuracy']:.4f}, balanced acc: {val_metrics['balanced_accuracy']:.4f}, f1: {val_metrics['f1']:.4f}")
        print(f"  best score: {best_score:.4f} (monitor: {monitor})")

        current_score = float(val_metrics.get(monitor, val_metrics.get("balanced_accuracy", 0.0)))
        is_best = current_score > best_score + args.min_delta
        if is_best:
            best_score = current_score
            patience_counter = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_score": best_score,
                    "classes": classes,
                    "args": vars(args),
                },
                output_dir,
                best_checkpoint_path.name,
            )
            print(f"  Saved best checkpoint to {best_checkpoint_path}")
        else:
            patience_counter += 1

        if args.early_stop and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}. No improvement in {patience_counter} epochs.")
            break

    print("Training complete.")
    print(f"Best balanced accuracy: {best_score:.4f}")


if __name__ == "__main__":
    main()
