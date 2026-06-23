"""Inference CLI for dual-image metadata checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from datasets import LABEL_COLUMNS, normalize_image_type
from milk10k_effb2_metadata.data import (
    DERMOSCOPIC_MASK_PATH_COLUMN,
    METADATA_COLUMNS,
    apply_dermoscopic_mask,
    audit_dermoscopic_masks,
    make_transforms,
    metadata_vector,
    print_mask_audit_summary,
    resolve_monet_columns,
)
from milk10k_effb2_metadata.metrics import apply_class_bias, compute_metrics
from milk10k_effb2_metadata.model_setup import load_model_state_compat
from milk10k_effb2_metadata.models import (
    DualEffB2MetadataClassifier,
    is_one_encoder_image_fusion,
    model_class_for_backbone,
    normalize_backbone_name,
    resolve_image_size,
)
from milk10k_effb2_metadata.training import json_safe


class InferencePairedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, metadata_spec: dict[str, Any], transform=None) -> None:
        self.df = df.reset_index(drop=True)
        self.metadata = np.stack([metadata_vector(row, metadata_spec) for _, row in self.df.iterrows()])
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str, mask_path: str | Path | None = None) -> torch.Tensor:
        with Image.open(path) as img:
            image = apply_dermoscopic_mask(img, mask_path)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        return {
            "clinical": self._load_image(row["clinical_path"]),
            "dermoscopic": self._load_image(
                row["dermoscopic_path"],
                row.get(DERMOSCOPIC_MASK_PATH_COLUMN),
            ),
            "metadata": torch.from_numpy(self.metadata[idx]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a MILK10k dual-image metadata checkpoint.")
    parser.add_argument("--checkpoint", type=Path, nargs="*", default=None, help="One or more checkpoint paths.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional run directory. If it contains fold_*/best.pt those checkpoints are ensembled; otherwise uses best.pt in the directory.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing MILK10k input/metadata files.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Image root. Overrides --data-dir/MILK10k_Training_Input.")
    parser.add_argument("--metadata-csv", type=Path, default=None, help="Metadata CSV. Overrides --data-dir/MILK10k_Training_Metadata.csv.")
    parser.add_argument(
        "--dermoscopic-mask-dir",
        type=Path,
        default=None,
        help="Optional directory containing <lesion_id>_dermoscopic_mask.png files.",
    )
    parser.add_argument(
        "--min-dermoscopic-mask-ratio",
        type=float,
        default=0.01,
        help="Fallback to the original dermoscopic image when mask foreground ratio is below this value.",
    )
    parser.add_argument("--groundtruth-csv", type=Path, default=None, help="Optional ground-truth CSV for metrics.")
    parser.add_argument("--output", type=Path, default=Path("test_predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=None, help="Defaults to checkpoint args image_size.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--tta-flips", action="store_true", help="Average original, H-flip, V-flip, and HV-flip predictions.")
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=None,
        help="Optional calibration.json override. By default calibration.json next to each checkpoint is loaded automatically.",
    )
    parser.add_argument("--no-auto-calibration", action="store_true", help="Disable auto-loading calibration.json next to checkpoints.")
    parser.add_argument("--include-debug-columns", action="store_true", help="Include lesion/file IDs and predicted labels before class probabilities.")
    return parser.parse_args()


def load_inference_dataframe(
    input_dir: Path,
    metadata_csv: Path,
    groundtruth_csv: Path | None,
) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv)
    monet_columns = resolve_monet_columns(meta)
    meta["image_type_norm"] = meta["image_type"].map(normalize_image_type)
    meta["path"] = meta.apply(lambda r: input_dir / r["lesion_id"] / f"{r['isic_id']}.jpg", axis=1)
    meta = meta[meta["path"].map(lambda p: p.exists())].copy()
    meta["path"] = meta["path"].map(str)

    keep = ["lesion_id", "isic_id", "path", *METADATA_COLUMNS, *monet_columns]
    if "id" in meta.columns:
        keep.insert(0, "id")
    clinical = meta[meta["image_type_norm"] == "clinical_close_up"][keep].drop_duplicates("lesion_id")
    dermoscopic = meta[meta["image_type_norm"] == "dermoscopic"][keep].drop_duplicates("lesion_id")
    paired = (
        clinical.add_prefix("clinical_")
        .merge(dermoscopic.add_prefix("dermoscopic_"), left_on="clinical_lesion_id", right_on="dermoscopic_lesion_id")
        .rename(columns={"clinical_lesion_id": "lesion_id"})
        .drop(columns=["dermoscopic_lesion_id"])
    )
    if "clinical_id" in paired.columns:
        paired["id"] = paired["clinical_id"]

    if groundtruth_csv is not None and groundtruth_csv.exists():
        gt = pd.read_csv(groundtruth_csv)
        gt["label"] = gt[LABEL_COLUMNS].idxmax(axis=1)
        paired = paired.merge(gt[["lesion_id", "label"]], on="lesion_id", how="left")

    if paired.empty:
        raise ValueError(f"No paired clinical/dermoscopic lesions found under {input_dir}")
    return paired


def resolve_input_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    if args.data_dir is None and (args.input_dir is None or args.metadata_csv is None):
        raise ValueError("Pass --data-dir, or pass both --input-dir and --metadata-csv.")

    data_dir = args.data_dir.expanduser().resolve() if args.data_dir is not None else None
    input_dir = args.input_dir or data_dir / "MILK10k_Training_Input"
    metadata_csv = args.metadata_csv or data_dir / "MILK10k_Training_Metadata.csv"
    groundtruth_csv = args.groundtruth_csv
    return input_dir.expanduser().resolve(), metadata_csv.expanduser().resolve(), groundtruth_csv


def infer_backend_from_model_state(state: dict[str, torch.Tensor], branch_prefix: str) -> str:
    keys = [key.removeprefix(branch_prefix) for key in state if key.startswith(branch_prefix)]
    timm_prefixes = ("conv_stem.", "bn1.", "blocks.", "conv_head.", "bn2.", "stages.", "stem.")
    torchvision_prefixes = ("features.", "avgpool.", "classifier.")
    timm_hits = sum(key.startswith(timm_prefixes) for key in keys)
    torchvision_hits = sum(key.startswith(torchvision_prefixes) for key in keys)
    if timm_hits > torchvision_hits:
        return "timm"
    if torchvision_hits > timm_hits:
        return "torchvision"
    if any(key.startswith("layer") for key in keys):
        return "timm"
    raise RuntimeError(f"Cannot infer backend for checkpoint branch prefix {branch_prefix!r}.")


def checkpoint_arg(checkpoint_args: dict[str, Any], key: str, default: Any) -> Any:
    value = checkpoint_args.get(key, default)
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def resolve_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    checkpoint_paths = [path.expanduser().resolve() for path in (args.checkpoint or [])]
    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
        fold_paths = sorted(path for path in checkpoint_dir.glob("fold_*/best.pt") if path.is_file())
        if fold_paths:
            checkpoint_paths.extend(fold_paths)
        else:
            best_path = checkpoint_dir / "best.pt"
            if best_path.is_file():
                checkpoint_paths.append(best_path)
    if not checkpoint_paths:
        raise ValueError("Pass --checkpoint, or pass --checkpoint-dir containing best.pt or fold_*/best.pt.")
    return checkpoint_paths


def build_model_from_checkpoint(checkpoint: dict[str, Any], metadata_dim: int, device: torch.device) -> DualEffB2MetadataClassifier:
    state = checkpoint["model_state"]
    checkpoint_args = checkpoint.get("args", {})
    class_names = checkpoint["class_names"]
    backbone = normalize_backbone_name(checkpoint_arg(checkpoint_args, "backbone", "efficientnet_b2"))
    image_fusion = checkpoint_arg(checkpoint_args, "image_fusion", "concat")
    if is_one_encoder_image_fusion(image_fusion):
        shared_backend = infer_backend_from_model_state(state, "shared_encoder.")
        clinical_backend = shared_backend
        dermoscopic_backend = shared_backend
    else:
        clinical_backend = infer_backend_from_model_state(state, "clinical_encoder.")
        dermoscopic_backend = infer_backend_from_model_state(state, "dermoscopic_encoder.")
    model_class = model_class_for_backbone(backbone)
    saved_model_type = checkpoint.get("model_type")
    if saved_model_type is not None and saved_model_type != model_class.__name__:
        raise ValueError(
            f"Checkpoint model_type {saved_model_type!r} does not match backbone "
            f"{backbone!r} ({model_class.__name__})."
        )
    model = model_class(
        num_classes=len(class_names),
        metadata_input_dim=metadata_dim,
        branch_dim=checkpoint_arg(checkpoint_args, "branch_dim", 512),
        metadata_dim=checkpoint_arg(checkpoint_args, "metadata_dim", 64),
        classifier_hidden_dim=checkpoint_arg(checkpoint_args, "classifier_hidden_dim", 512),
        dropout=checkpoint_arg(checkpoint_args, "dropout", 0.3),
        imagenet_pretrained=False,
        clinical_backbone_backend=clinical_backend,
        dermoscopic_backbone_backend=dermoscopic_backend,
        backbone=backbone,
        disable_metadata=checkpoint_arg(checkpoint_args, "disable_metadata", False),
        metadata_fusion=checkpoint_arg(checkpoint_args, "metadata_fusion", "concat"),
        image_fusion=image_fusion,
        metadata_gate_hidden_dim=checkpoint_args.get("metadata_gate_hidden_dim"),
        classifier_style=checkpoint_arg(checkpoint_args, "classifier_style", "legacy"),
        logit_fusion_mode=checkpoint_arg(checkpoint_args, "logit_fusion_mode", "single"),
        fusion_logit_weight=checkpoint_arg(checkpoint_args, "fusion_logit_weight", 0.6),
        clinical_logit_weight=checkpoint_arg(checkpoint_args, "clinical_logit_weight", 0.2),
        dermoscopic_logit_weight=checkpoint_arg(checkpoint_args, "dermoscopic_logit_weight", 0.2),
    ).to(device)
    load_model_state_compat(model, state)
    model.eval()
    return model


@torch.no_grad()
def predict_dataframe(
    model: DualEffB2MetadataClassifier,
    loader: DataLoader,
    device: torch.device,
    tta_flips: bool = False,
    temperature: float = 1.0,
) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError(f"Checkpoint temperature must be positive, got {temperature}.")
    probs_all = []
    for batch in tqdm(loader, leave=False):
        clinical = batch["clinical"].to(device, non_blocking=True)
        dermoscopic = batch["dermoscopic"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        views = [(clinical, dermoscopic)]
        if tta_flips:
            views.extend(
                [
                    (torch.flip(clinical, dims=(-1,)), torch.flip(dermoscopic, dims=(-1,))),
                    (torch.flip(clinical, dims=(-2,)), torch.flip(dermoscopic, dims=(-2,))),
                    (torch.flip(clinical, dims=(-2, -1)), torch.flip(dermoscopic, dims=(-2, -1))),
                ]
            )
        probs = None
        for clinical_view, dermoscopic_view in views:
            logits = model(clinical_view, dermoscopic_view, metadata)
            view_prob = torch.softmax(logits / temperature, dim=1)
            probs = view_prob if probs is None else probs + view_prob
        probs_all.append((probs / len(views)).cpu().numpy())
    return np.concatenate(probs_all)


def load_calibration_bias(
    checkpoint_path: Path,
    args: argparse.Namespace,
    expected_class_names: list[str],
) -> np.ndarray | None:
    if args.calibration_file is not None:
        calibration_path = args.calibration_file.expanduser().resolve()
    elif args.no_auto_calibration:
        return None
    else:
        calibration_path = checkpoint_path.parent / "calibration.json"
    if not calibration_path.exists():
        return None
    with open(calibration_path, encoding="utf-8") as f:
        payload = json.load(f)
    class_names = payload.get("class_names", [])
    if class_names != expected_class_names:
        raise ValueError(
            f"Calibration class_names mismatch for {calibration_path}: "
            f"expected {expected_class_names}, got {class_names}"
        )
    return np.asarray(payload["class_bias"], dtype=np.float32)


def save_inference_outputs(
    df: pd.DataFrame,
    y_prob: np.ndarray,
    class_names: list[str],
    output: Path,
    include_debug_columns: bool = False,
) -> None:
    probability_df = pd.DataFrame(y_prob, columns=class_names)
    probability_df.insert(0, "lesion_id", df["lesion_id"].tolist())
    output.parent.mkdir(parents=True, exist_ok=True)
    if not include_debug_columns:
        probability_df.to_csv(output, index=False)
        return

    y_pred = y_prob.argmax(axis=1)
    prediction_df = pd.DataFrame(
        {
            "lesion_id": df["lesion_id"].tolist(),
            "clinical_file": [Path(path).name for path in df["clinical_path"].tolist()],
            "dermoscopic_file": [Path(path).name for path in df["dermoscopic_path"].tolist()],
            "clinical_isic_id": df.get("clinical_isic_id", pd.Series([""] * len(df))).tolist(),
            "dermoscopic_isic_id": df.get("dermoscopic_isic_id", pd.Series([""] * len(df))).tolist(),
            "y_pred": y_pred,
            "label_pred": [class_names[idx] for idx in y_pred],
            "confidence": y_prob.max(axis=1),
        }
    )
    if "label" in df.columns:
        prediction_df["label_true"] = df["label"].tolist()
    pd.concat([prediction_df, probability_df], axis=1).to_csv(output, index=False)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir, metadata_csv, groundtruth_csv = resolve_input_paths(args)
    df = load_inference_dataframe(input_dir, metadata_csv, groundtruth_csv)
    if not 0.0 <= args.min_dermoscopic_mask_ratio <= 1.0:
        raise ValueError("--min-dermoscopic-mask-ratio must be between 0 and 1.")
    if args.dermoscopic_mask_dir is not None:
        args.dermoscopic_mask_dir = args.dermoscopic_mask_dir.expanduser().resolve()
        df, mask_audit = audit_dermoscopic_masks(
            df,
            args.dermoscopic_mask_dir,
            args.min_dermoscopic_mask_ratio,
            mask_id_column="dermoscopic_isic_id",
            mask_suffix="_mask.png",
        )
        audit_output = args.output.with_name(f"{args.output.stem}.mask_audit.csv")
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        mask_audit.to_csv(audit_output, index=False)
        print_mask_audit_summary(mask_audit, args.min_dermoscopic_mask_ratio)
    checkpoint_paths = resolve_checkpoint_paths(args)
    ensemble_probs = []
    class_names: list[str] | None = None

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        checkpoint_class_names = checkpoint["class_names"]
        if class_names is None:
            class_names = checkpoint_class_names
        elif checkpoint_class_names != class_names:
            raise ValueError(
                f"Checkpoint class_names mismatch: expected {class_names}, got {checkpoint_class_names} from {checkpoint_path}"
            )
        checkpoint_args = checkpoint.get("args", {})
        backbone = checkpoint_args.get("backbone", "efficientnet_b2")
        checkpoint_image_size = checkpoint_args.get("image_size")
        image_size = resolve_image_size(
            backbone,
            args.image_size if args.image_size is not None else checkpoint_image_size,
        )
        _, eval_transform = make_transforms(image_size)
        dataset = InferencePairedDataset(df, checkpoint["metadata_spec"], eval_transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            shuffle=False,
        )
        model = build_model_from_checkpoint(checkpoint, dataset.metadata.shape[1], device)
        temperature = float(checkpoint.get("temperature", 1.0))
        y_prob = predict_dataframe(
            model,
            loader,
            device,
            tta_flips=args.tta_flips,
            temperature=temperature,
        )
        print(
            f"Checkpoint {checkpoint_path.name}: variant={checkpoint.get('checkpoint_variant', 'legacy')}, "
            f"temperature={temperature:.4f}"
        )
        class_bias = load_calibration_bias(checkpoint_path, args, checkpoint_class_names)
        if class_bias is not None:
            y_prob = apply_class_bias(y_prob, class_bias)
        ensemble_probs.append(y_prob)

    assert class_names is not None
    y_prob = np.mean(ensemble_probs, axis=0)
    save_inference_outputs(df, y_prob, class_names, args.output, args.include_debug_columns)

    y_pred = y_prob.argmax(axis=1)
    for tail_name in ("DF", "INF"):
        if tail_name not in class_names:
            continue
        idx = class_names.index(tail_name)
        predicted_count = int((y_pred == idx).sum())
        max_probability = float(y_prob[:, idx].max())
        mean_probability = float(y_prob[:, idx].mean())
        print(
            f"Tail audit {tail_name}: predicted_count={predicted_count}, "
            f"mean_probability={mean_probability:.6f}, max_probability={max_probability:.6f}"
        )
        if predicted_count == 0:
            print(f"WARNING: no sample is predicted as {tail_name}.")
        if max_probability < 0.01:
            print(f"WARNING: {tail_name} maximum probability is below 0.01.")

    print(f"Saved predictions: {args.output}")
    if "label" in df.columns and df["label"].notna().all():
        label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        y_true = np.array([label_to_idx[label] for label in df["label"]])
        metrics, _, _ = compute_metrics(y_true, y_prob, class_names)
        metrics_path = args.output.with_suffix(".metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            import json

            json.dump(json_safe(metrics), f, indent=2)
        print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
