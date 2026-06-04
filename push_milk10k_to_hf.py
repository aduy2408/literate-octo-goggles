#!/usr/bin/env python3
"""
Stage and optionally upload MILK10k as a Hugging Face dataset repository.

The staged folder uses the Hub imagefolder convention:

  README.md
  metadata.csv
  images/<lesion_id>/<isic_id>.jpg
  original_csvs/*.csv

Examples:
  python push_milk10k_to_hf.py --repo-id your-username/milk10k
  python push_milk10k_to_hf.py --repo-id your-username/milk10k --private
  python push_milk10k_to_hf.py --stage-dir hf_milk10k --no-upload

Install upload dependency first:
  pip install huggingface_hub

Authenticate once with:
  hf auth login
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import Counter
from pathlib import Path


REQUIRED_DATA_FILES = (
    "MILK10k_Training_GroundTruth.csv",
    "MILK10k_Training_Metadata.csv",
    "MILK10k_Training_Input",
)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push MILK10k to a Hugging Face dataset repo.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Folder containing MILK10k CSVs and MILK10k_Training_Input.",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=Path("hf_milk10k_dataset"),
        help="Local folder to create before upload.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face dataset repo id, for example username/milk10k. If omitted, only stages locally.",
    )
    parser.add_argument("--private", action="store_true", help="Create the Hub dataset repo as private.")
    parser.add_argument("--token", default=None, help="Hugging Face token. Defaults to logged-in token or HF_TOKEN.")
    parser.add_argument(
        "--image-type",
        choices=["all", "clinical_close_up", "dermoscopic"],
        default="all",
        help="Optionally upload only one MILK10k modality.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into the staged folder instead of symlinking them.",
    )
    parser.add_argument(
        "--overwrite-stage",
        action="store_true",
        help="Delete and rebuild --stage-dir if it already exists.",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Only build the staged dataset folder.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload MILK10k dataset",
        help="Commit message used for Hugging Face upload.",
    )
    return parser.parse_args()


def normalize_image_type(image_type: str) -> str:
    if image_type == "clinical: close-up":
        return "clinical_close_up"
    return image_type.replace(" ", "_").replace(":", "").replace("-", "_")


def has_milk10k_files(path: Path) -> bool:
    return all((path / name).exists() for name in REQUIRED_DATA_FILES)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def label_for_groundtruth_row(row: dict[str, str]) -> str:
    values = []
    for label in LABEL_COLUMNS:
        try:
            value = float(row.get(label, "0") or 0)
        except ValueError:
            value = 0.0
        values.append((value, label))
    return max(values)[1]


def prepare_stage_dir(stage_dir: Path, overwrite: bool) -> None:
    if stage_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{stage_dir} already exists. Pass --overwrite-stage to rebuild it.")
        shutil.rmtree(stage_dir)
    (stage_dir / "images").mkdir(parents=True)
    (stage_dir / "original_csvs").mkdir(parents=True)


def link_or_copy_image(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_images:
        shutil.copy2(src, dst)
        return

    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def build_records(data_dir: Path, stage_dir: Path, image_type: str, copy_images: bool) -> list[dict[str, str]]:
    input_dir = data_dir / "MILK10k_Training_Input"
    gt_rows = read_csv(data_dir / "MILK10k_Training_GroundTruth.csv")
    meta_rows = read_csv(data_dir / "MILK10k_Training_Metadata.csv")
    supplement_rows = read_csv(data_dir / "MILK10k_Training_Supplement.csv")

    label_by_lesion = {row["lesion_id"]: label_for_groundtruth_row(row) for row in gt_rows}
    supplement_by_isic = {row["isic_id"]: row for row in supplement_rows}

    records: list[dict[str, str]] = []
    missing_images = 0
    for meta in meta_rows:
        image_type_norm = normalize_image_type(meta["image_type"])
        if image_type != "all" and image_type_norm != image_type:
            continue

        lesion_id = meta["lesion_id"]
        isic_id = meta["isic_id"]
        src = input_dir / lesion_id / f"{isic_id}.jpg"
        if not src.exists():
            missing_images += 1
            continue

        file_name = f"images/{lesion_id}/{isic_id}.jpg"
        link_or_copy_image(src, stage_dir / file_name, copy_images)

        supplement = supplement_by_isic.get(isic_id, {})
        record = {
            "file_name": file_name,
            "label": label_by_lesion.get(lesion_id, ""),
            "lesion_id": lesion_id,
            "isic_id": isic_id,
            "image_type_norm": image_type_norm,
        }
        for key, value in meta.items():
            if key not in record:
                record[key] = value
        for key, value in supplement.items():
            if key not in record:
                record[key] = value
            elif key != "isic_id":
                record[f"supplement_{key}"] = value
        records.append(record)

    if not records:
        raise ValueError(f"No images found for image_type={image_type!r} under {input_dir}")
    if missing_images:
        print(f"Skipped {missing_images} metadata rows because the image file was missing.")
    return records


def write_metadata(stage_dir: Path, records: list[dict[str, str]]) -> None:
    fieldnames: list[str] = []
    for preferred in ("file_name", "label", "lesion_id", "isic_id", "image_type", "image_type_norm"):
        if any(preferred in record for record in records):
            fieldnames.append(preferred)
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)

    with (stage_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_dataset_card(stage_dir: Path, records: list[dict[str, str]]) -> None:
    label_counts = Counter(record["label"] for record in records)
    image_type_counts = Counter(record["image_type_norm"] for record in records)
    labels = "\n".join(f"- {label}: {count}" for label, count in sorted(label_counts.items()))
    image_types = "\n".join(f"- {kind}: {count}" for kind, count in sorted(image_type_counts.items()))
    card = f"""---
task_categories:
- image-classification
pretty_name: MILK10k
---

# MILK10k

This repository contains a staged copy of the MILK10k training data prepared from the local files in this workspace.

## Files

- `metadata.csv`: Hugging Face imagefolder metadata with `file_name`, diagnosis label, lesion id, ISIC id, modality, MILK metadata, and supplement columns.
- `images/`: JPEG images organized by lesion id.
- `original_csvs/`: Original MILK10k CSV files used to build this upload.

## Counts

Total images: {len(records)}

Labels:

{labels}

Image types:

{image_types}

## License

The source CSV metadata lists `CC-BY-NC` in `copyright_license`. Confirm the exact upstream license and usage constraints before publishing publicly or using commercially.
"""
    (stage_dir / "README.md").write_text(card, encoding="utf-8")


def copy_original_csvs(data_dir: Path, stage_dir: Path) -> None:
    for name in (
        "MILK10k_Training_GroundTruth.csv",
        "MILK10k_Training_Metadata.csv",
        "MILK10k_Training_Supplement.csv",
    ):
        src = data_dir / name
        if src.exists():
            shutil.copy2(src, stage_dir / "original_csvs" / name)


def upload_to_hf(stage_dir: Path, repo_id: str, private: bool, token: str | None, commit_message: str) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install huggingface_hub") from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(stage_dir),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    stage_dir = args.stage_dir.expanduser().resolve()

    if not has_milk10k_files(data_dir):
        expected = ", ".join(REQUIRED_DATA_FILES)
        raise FileNotFoundError(f"{data_dir} does not contain required MILK10k files: {expected}")

    prepare_stage_dir(stage_dir, args.overwrite_stage)
    records = build_records(data_dir, stage_dir, args.image_type, args.copy_images)
    write_metadata(stage_dir, records)
    write_dataset_card(stage_dir, records)
    copy_original_csvs(data_dir, stage_dir)

    print(f"Staged {len(records)} images at {stage_dir}")
    print(f"Metadata: {stage_dir / 'metadata.csv'}")

    if args.no_upload or not args.repo_id:
        print("Upload skipped. Pass --repo-id username/dataset-name to upload.")
        return

    upload_to_hf(stage_dir, args.repo_id, args.private, args.token, args.commit_message)
    print(f"Uploaded dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
