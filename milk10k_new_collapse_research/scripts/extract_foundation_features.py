"""Extract frozen paired clinical/dermoscopic features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from milk10k_new_collapse_research.compat import ensure_legacy_package_path
from milk10k_new_collapse_research.config import RESULTS_ROOT
from milk10k_new_collapse_research.features import (
    PairImageDataset,
    build_feature_model,
    extract_pair_features,
    save_feature_npz,
)

ensure_legacy_package_path()
from milk10k_effb2_metadata.data import lesion_split, load_paired_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "features")
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="Optional per-split row cap for smoke tests.")
    parser.add_argument("--no-l2-normalize", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true", help="Only for local smoke tests with timm models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    df = load_paired_dataframe(args.data_dir)
    train_df, val_df = lesion_split(df, args.val_size, args.seed)
    if args.limit > 0:
        train_df = train_df.head(args.limit).copy()
        val_df = val_df.head(args.limit).copy()

    model, transform, image_size = build_feature_model(args.model_name, device, pretrained=not args.no_pretrained)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_df in [("train", train_df), ("val", val_df)]:
        dataset = PairImageDataset(split_df, transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        batch = extract_pair_features(model, loader, device, l2_normalize=not args.no_l2_normalize)
        save_feature_npz(args.output_dir / f"{split}_features.npz", batch, args.model_name, split)

    manifest = {
        "model_name": args.model_name,
        "image_size": image_size,
        "data_dir": str(args.data_dir),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "val_size": args.val_size,
        "seed": args.seed,
    }
    (args.output_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
