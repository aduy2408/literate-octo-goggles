#!/usr/bin/env python3
"""
Prepare a one-class, one-modality MILK10k folder for Stable Diffusion training.

MILK10k has two images per lesion/sample:
  - clinical: close-up
  - dermoscopic

This script exports only one requested modality. For example, if you prepare
clinical_close_up MEL data, the paired dermoscopic MEL image from the same
lesion_id is not copied.

The output format is simple and works with common DreamBooth/LoRA trainers:
  <output-dir>/images/*.jpg
  <output-dir>/images/*.txt
  <output-dir>/training_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

LABEL_COLUMNS = [
    "AKIEC",
    "BCC",
    "BEN_OTH",
    "BKL",
    "DF",
    "INF",
    "MAL_OTH",
    "MEL",
    "NV",
    "SCCKA",
    "VASC",
]

CLASS_PROMPTS = {
    "AKIEC": "actinic keratosis or intraepithelial carcinoma skin lesion",
    "BCC": "basal cell carcinoma skin lesion",
    "BEN_OTH": "benign other skin lesion",
    "BKL": "benign keratosis-like skin lesion",
    "DF": "dermatofibroma skin lesion",
    "INF": "inflammatory skin lesion",
    "MAL_OTH": "other malignant skin lesion",
    "MEL": "melanoma skin lesion",
    "NV": "melanocytic nevus skin lesion",
    "SCCKA": "squamous cell carcinoma or keratoacanthoma skin lesion",
    "VASC": "vascular skin lesion",
}

IMAGE_TYPE_PROMPTS = {
    "clinical_close_up": "clinical close-up dermatology photograph",
    "dermoscopic": "dermoscopic dermatology image",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MILK10k images for one-class Stable Diffusion training.")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="MILK10k root folder.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output training folder.")
    parser.add_argument("--class-name", required=True, choices=LABEL_COLUMNS)
    parser.add_argument(
        "--image-type",
        required=True,
        choices=["clinical_close_up", "dermoscopic"],
        help="Export exactly one modality. clinical_close_up never copies dermoscopic images, and vice versa.",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--copy", action="store_true", help="Copy images instead of creating symlinks.")
    parser.add_argument("--caption", default=None, help="Override caption text written next to each image.")
    return parser.parse_args()


def normalize_image_type(image_type: str) -> str:
    if image_type == "clinical: close-up":
        return "clinical_close_up"
    return image_type.replace(" ", "_").replace(":", "").replace("-", "_")


def load_class_modality_rows(data_dir: Path, class_name: str, image_type: str) -> list[dict[str, str]]:
    input_dir = data_dir / "MILK10k_Training_Input"
    gt_path = data_dir / "MILK10k_Training_GroundTruth.csv"
    meta_path = data_dir / "MILK10k_Training_Metadata.csv"
    if not gt_path.exists() or not meta_path.exists() or not input_dir.exists():
        raise FileNotFoundError(
            "Expected MILK10k_Training_GroundTruth.csv, MILK10k_Training_Metadata.csv, "
            "and MILK10k_Training_Input under --data-dir."
        )

    selected_lesions = set()
    with gt_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if float(row[class_name]) == 1.0:
                selected_lesions.add(row["lesion_id"])

    rows = []
    with meta_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["lesion_id"] not in selected_lesions:
                continue
            image_type_norm = normalize_image_type(row["image_type"])
            if image_type_norm != image_type:
                continue
            source_path = input_dir / row["lesion_id"] / f"{row['isic_id']}.jpg"
            if not source_path.exists():
                continue
            rows.append(
                {
                    "lesion_id": row["lesion_id"],
                    "isic_id": row["isic_id"],
                    "image_type_norm": image_type_norm,
                    "source_path": str(source_path),
                }
            )

    if not rows:
        raise ValueError(f"No images found for class={class_name}, image_type={image_type}")
    return rows


def link_or_copy(src: Path, dst: Path, copy_file: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_file:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = load_class_modality_rows(data_dir, args.class_name, args.image_type)
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.max_images is not None:
        rows = rows[: args.max_images]

    caption = args.caption or f"{CLASS_PROMPTS[args.class_name]}, {IMAGE_TYPE_PROMPTS[args.image_type]}"
    manifest_path = output_dir / "training_manifest.csv"

    with manifest_path.open("w", newline="") as f:
        fieldnames = ["class_name", "image_type", "lesion_id", "isic_id", "source_path", "train_image_path", "caption"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            dst_image = images_dir / f"{row['lesion_id']}_{row['isic_id']}_{args.image_type}.jpg"
            dst_caption = dst_image.with_suffix(".txt")
            link_or_copy(Path(row["source_path"]), dst_image, args.copy)
            dst_caption.write_text(caption + "\n", encoding="utf-8")
            writer.writerow(
                {
                    "class_name": args.class_name,
                    "image_type": args.image_type,
                    "lesion_id": row["lesion_id"],
                    "isic_id": row["isic_id"],
                    "source_path": row["source_path"],
                    "train_image_path": str(dst_image),
                    "caption": caption,
                }
            )

    print(f"Exported {len(rows)} {args.image_type} images for class {args.class_name}")
    print(f"Images/captions: {images_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
