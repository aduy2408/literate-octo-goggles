"""Inference for dermoscopic-only metadata checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .core import apply_class_bias, compute_metrics, json_safe
from .data import BASE_METADATA_COLUMNS, DermoscopicMetadataDataset, LABEL_COLUMNS, make_transforms, normalize_image_type, resolve_monet_columns
from .model import DermoscopicMetadataClassifier


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict with dermoscopic-only metadata checkpoints.")
    parser.add_argument("--checkpoint", type=Path, nargs="*", default=[])
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--groundtruth-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--tta-flips", action="store_true")
    parser.add_argument("--calibration-file", type=Path, default=None)
    parser.add_argument("--no-auto-calibration", action="store_true")
    parser.add_argument("--include-debug-columns", action="store_true")
    return parser.parse_args(argv)


def checkpoint_paths(args):
    paths = [path.expanduser().resolve() for path in args.checkpoint]
    if args.checkpoint_dir:
        directory = args.checkpoint_dir.expanduser().resolve()
        folds = sorted(directory.glob("fold_*/best.pt"))
        paths.extend(folds or ([directory / "best.pt"] if (directory / "best.pt").exists() else []))
    if not paths:
        raise ValueError("Pass --checkpoint or --checkpoint-dir containing best.pt.")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")
    return paths


def load_dataframe(input_dir: Path, metadata_csv: Path, groundtruth_csv: Path | None):
    input_dir = input_dir.expanduser().resolve()
    meta = pd.read_csv(metadata_csv.expanduser().resolve())
    required = {"lesion_id", "isic_id", "image_type", *BASE_METADATA_COLUMNS}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata CSV is missing columns: {sorted(missing)}")
    meta["image_type_norm"] = meta["image_type"].map(normalize_image_type)
    df = meta[meta["image_type_norm"] == "dermoscopic"].copy()
    duplicates = df.groupby("lesion_id").size()
    if (duplicates > 1).any():
        raise ValueError("Expected exactly one dermoscopic row per lesion.")
    df["image_path"] = df.apply(lambda row: input_dir / str(row["lesion_id"]) / f"{row['isic_id']}.jpg", axis=1)
    df = df[df["image_path"].map(Path.exists)].copy(); df["image_path"] = df["image_path"].map(str)
    if groundtruth_csv and groundtruth_csv.exists():
        gt = pd.read_csv(groundtruth_csv)
        columns = [name for name in LABEL_COLUMNS if name in gt.columns]
        gt["label"] = gt[columns].idxmax(axis=1)
        df = df.merge(gt[["lesion_id", "label"]], on="lesion_id", how="left", validate="one_to_one")
    if df.empty:
        raise ValueError("No existing dermoscopic image files were found.")
    return df.reset_index(drop=True)


def build_model(checkpoint, device):
    saved = checkpoint["args"]
    spec = checkpoint["metadata_spec"]
    input_dim = 2 + len(spec["sex_values"]) + len(spec["site_values"]) + len(spec.get("monet_columns", []))
    model = DermoscopicMetadataClassifier(
        len(checkpoint["class_names"]), input_dim, saved["metadata_mode"], saved.get("backbone", "efficientnet_b2"), False,
        int(saved.get("branch_dim", 512)), int(saved.get("metadata_dim", 64)), int(saved.get("classifier_hidden_dim", 512)),
        float(saved.get("dropout", 0.3)), checkpoint.get("backbone_backend_resolved", saved.get("backbone_backend", "timm")).replace("auto", "timm"),
        saved.get("metadata_gate_hidden_dim"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    return model


@torch.no_grad()
def predict(model, loader, device, tta):
    output = []
    for batch in tqdm(loader, leave=False):
        image = batch["image"].to(device); metadata = batch["metadata"].to(device)
        views = [image]
        if tta:
            views.extend([torch.flip(image, (-1,)), torch.flip(image, (-2,)), torch.flip(image, (-2, -1))])
        probs = sum(torch.softmax(model(view, metadata), 1) for view in views) / len(views)
        output.append(probs.cpu().numpy())
    return np.concatenate(output)


def run(args):
    paths = checkpoint_paths(args); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = [torch.load(path, map_location=device, weights_only=False) for path in paths]
    class_names = checkpoints[0]["class_names"]
    for checkpoint in checkpoints[1:]:
        if checkpoint["class_names"] != class_names:
            raise ValueError("Ensemble checkpoints have incompatible class names.")
    if args.data_dir is not None:
        data_dir=args.data_dir.expanduser().resolve()
        default_input=data_dir/"MILK10k_Training_Input"
        if not default_input.exists():default_input=data_dir.parent/"MILK10k_Training_Input"
        args.input_dir=args.input_dir or default_input
        args.metadata_csv=args.metadata_csv or data_dir/"MILK10k_Training_Metadata.csv"
        if args.groundtruth_csv is None and (data_dir/"MILK10k_Training_GroundTruth.csv").exists():args.groundtruth_csv=data_dir/"MILK10k_Training_GroundTruth.csv"
    if args.input_dir is None or args.metadata_csv is None:raise ValueError("Pass --data-dir or both --input-dir and --metadata-csv.")
    df = load_dataframe(args.input_dir, args.metadata_csv, args.groundtruth_csv)
    size = args.image_size or int(checkpoints[0]["args"].get("image_size", 260))
    _, transform = make_transforms(size)
    ensemble = []
    for path, checkpoint in zip(paths, checkpoints):
        dataset = DermoscopicMetadataDataset(df, None, checkpoint["metadata_spec"], transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
        probs = predict(build_model(checkpoint, device), loader, device, args.tta_flips)
        calibration_path = args.calibration_file.expanduser().resolve() if args.calibration_file else path.parent / "calibration.json"
        if not args.no_auto_calibration and calibration_path.exists():
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if calibration["class_names"] != class_names:
                raise ValueError(f"Calibration class mismatch: {calibration_path}")
            probs = apply_class_bias(probs, np.asarray(calibration["class_bias"], dtype=np.float32))
        ensemble.append(probs)
    y_prob = np.mean(ensemble, axis=0)
    output = pd.DataFrame(y_prob, columns=class_names)
    if args.include_debug_columns:
        output.insert(0,"confidence",y_prob.max(1));output.insert(0,"predicted_label",[class_names[i] for i in y_prob.argmax(1)])
        if "isic_id" in df:output.insert(0,"isic_id",df["isic_id"].tolist())
    output.insert(0, "lesion_id", df["lesion_id"].tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True); output.to_csv(args.output, index=False)
    if "label" in df and df["label"].notna().all():
        mapping = {name: index for index, name in enumerate(class_names)}
        y_true = df["label"].map(mapping).to_numpy()
        metrics, per_class, cm = compute_metrics(y_true, y_prob, class_names)
        args.output.with_suffix(".metrics.json").write_text(json.dumps(json_safe(metrics), indent=2), encoding="utf-8")
        per_class.to_csv(args.output.with_suffix(".per_class_metrics.csv"), index=False)
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(args.output.with_suffix(".confusion_matrix.csv"))
    print(f"Saved {len(df)} dermoscopic predictions to {args.output}")


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
