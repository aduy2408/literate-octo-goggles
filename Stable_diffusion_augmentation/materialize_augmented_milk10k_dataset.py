#!/usr/bin/env python3
"""Create a MILK10k-style dataset folder with original + synthetic paired augmentations."""

from __future__ import annotations

import argparse
import csv
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

NEUTRAL_METADATA = {
    "age_approx": "",
    "sex": "unknown",
    "skin_tone_class": "",
    "site": "unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize original + synthetic data as a MILK10k-style dataset.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Original MILK10k_Training_Input folder.")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="Original MILK10k_Training_Metadata.csv.")
    parser.add_argument("--groundtruth-csv", type=Path, required=True, help="Original MILK10k_Training_GroundTruth.csv.")
    parser.add_argument("--augmentation-manifest", type=Path, required=True, help="Paired augmentation manifest to add.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output MILK10k-style dataset folder.")
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copying. Default is copy, which is easier to move/use later.",
    )
    parser.add_argument(
        "--synthetic-metadata",
        choices=["source", "neutral"],
        default="source",
        help="Metadata for synthetic rows. source copies source lesion metadata; neutral writes unknown/0 values.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output CSV files and existing image links.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def link_or_copy(src: Path, dst: Path, copy_file: bool, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source image not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if copy_file:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def original_image_path(input_dir: Path, row: dict[str, str]) -> Path:
    return input_dir / row["lesion_id"] / f"{row['isic_id']}.jpg"


def synthetic_metadata_row(
    columns: list[str],
    lesion_id: str,
    image_type: str,
    isic_id: str,
) -> dict[str, str]:
    row = {column: "" for column in columns}
    row["lesion_id"] = lesion_id
    row["image_type"] = image_type
    row["isic_id"] = isic_id
    if "attribution" in row:
        row["attribution"] = "Stable Diffusion synthetic augmentation"
    if "copyright_license" in row:
        row["copyright_license"] = "synthetic"
    if "image_manipulation" in row:
        row["image_manipulation"] = "synthetic"
    for key, value in NEUTRAL_METADATA.items():
        if key in row:
            row[key] = value
    for column in columns:
        if column.startswith("MONET_"):
            row[column] = "0"
    return row


def source_metadata_row(
    source_row: dict[str, str] | None,
    columns: list[str],
    lesion_id: str,
    image_type: str,
    isic_id: str,
) -> dict[str, str]:
    if source_row is None:
        return synthetic_metadata_row(columns, lesion_id, image_type, isic_id)
    row = {column: source_row.get(column, "") for column in columns}
    row["lesion_id"] = lesion_id
    row["image_type"] = image_type
    row["isic_id"] = isic_id
    if "attribution" in row:
        row["attribution"] = "Stable Diffusion synthetic augmentation"
    if "copyright_license" in row:
        row["copyright_license"] = "synthetic"
    if "image_manipulation" in row:
        row["image_manipulation"] = "synthetic_from_source"
    return row


def synthetic_groundtruth_row(lesion_id: str, class_name: str, columns: list[str]) -> dict[str, str]:
    row = {column: "0.0" for column in columns}
    row["lesion_id"] = lesion_id
    for class_column in LABEL_COLUMNS:
        if class_column in row:
            row[class_column] = "1.0" if class_column == class_name else "0.0"
    return row


def materialize_original_images(input_dir: Path, output_input_dir: Path, metadata_rows: list[dict[str, str]], copy_file: bool, overwrite: bool) -> None:
    for row in metadata_rows:
        src = original_image_path(input_dir, row)
        dst = output_input_dir / row["lesion_id"] / f"{row['isic_id']}.jpg"
        link_or_copy(src, dst, copy_file, overwrite)


def materialize_synthetic_rows(
    manifest_rows: list[dict[str, str]],
    metadata_columns: list[str],
    groundtruth_columns: list[str],
    source_metadata_by_key: dict[tuple[str, str], dict[str, str]],
    synthetic_metadata_mode: str,
    output_input_dir: Path,
    copy_file: bool,
    overwrite: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    metadata_rows = []
    groundtruth_rows = []
    seen_lesions = set()

    for row in manifest_rows:
        lesion_id = row["synthetic_lesion_id"]
        clinical_isic_id = row.get("clinical_synthetic_isic_id") or f"{lesion_id}__clinical"
        dermoscopic_isic_id = row.get("dermoscopic_synthetic_isic_id") or f"{lesion_id}__dermoscopic"

        clinical_dst = output_input_dir / lesion_id / f"{clinical_isic_id}.jpg"
        dermoscopic_dst = output_input_dir / lesion_id / f"{dermoscopic_isic_id}.jpg"
        link_or_copy(Path(row["clinical_generated_path"]), clinical_dst, copy_file, overwrite)
        link_or_copy(Path(row["dermoscopic_generated_path"]), dermoscopic_dst, copy_file, overwrite)

        if synthetic_metadata_mode == "source":
            source_lesion_id = row.get("source_lesion_id", "")
            clinical_source_isic_id = row.get("clinical_source_isic_id", "")
            dermoscopic_source_isic_id = row.get("dermoscopic_source_isic_id", "")
            metadata_rows.append(
                source_metadata_row(
                    source_metadata_by_key.get((source_lesion_id, clinical_source_isic_id)),
                    metadata_columns,
                    lesion_id,
                    "clinical: close-up",
                    clinical_isic_id,
                )
            )
            metadata_rows.append(
                source_metadata_row(
                    source_metadata_by_key.get((source_lesion_id, dermoscopic_source_isic_id)),
                    metadata_columns,
                    lesion_id,
                    "dermoscopic",
                    dermoscopic_isic_id,
                )
            )
        else:
            metadata_rows.append(synthetic_metadata_row(metadata_columns, lesion_id, "clinical: close-up", clinical_isic_id))
            metadata_rows.append(synthetic_metadata_row(metadata_columns, lesion_id, "dermoscopic", dermoscopic_isic_id))
        if lesion_id not in seen_lesions:
            groundtruth_rows.append(synthetic_groundtruth_row(lesion_id, row["class_name"], groundtruth_columns))
            seen_lesions.add(lesion_id)

    return metadata_rows, groundtruth_rows


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    metadata_csv = args.metadata_csv.expanduser().resolve()
    groundtruth_csv = args.groundtruth_csv.expanduser().resolve()
    augmentation_manifest = args.augmentation_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_input_dir = output_dir / "MILK10k_Training_Input"

    metadata_rows = read_rows(metadata_csv)
    groundtruth_rows = read_rows(groundtruth_csv)
    manifest_rows = read_rows(augmentation_manifest)
    if not manifest_rows:
        raise ValueError(f"No augmentation rows found: {augmentation_manifest}")

    metadata_columns = list(metadata_rows[0].keys())
    groundtruth_columns = list(groundtruth_rows[0].keys())
    source_metadata_by_key = {(row["lesion_id"], row["isic_id"]): row for row in metadata_rows}

    copy_file = not args.symlink
    materialize_original_images(input_dir, output_input_dir, metadata_rows, copy_file, args.overwrite)
    synthetic_metadata_rows, synthetic_groundtruth_rows = materialize_synthetic_rows(
        manifest_rows,
        metadata_columns,
        groundtruth_columns,
        source_metadata_by_key,
        args.synthetic_metadata,
        output_input_dir,
        copy_file,
        args.overwrite,
    )

    all_metadata_rows = metadata_rows + synthetic_metadata_rows
    all_groundtruth_rows = groundtruth_rows + synthetic_groundtruth_rows
    write_rows(output_dir / "MILK10k_Training_Metadata.csv", all_metadata_rows, metadata_columns)
    write_rows(output_dir / "MILK10k_Training_GroundTruth.csv", all_groundtruth_rows, groundtruth_columns)

    print("Materialized augmented MILK10k dataset")
    print(f"  output_dir: {output_dir}")
    print(f"  original metadata rows: {len(metadata_rows)}")
    print(f"  synthetic metadata rows: {len(synthetic_metadata_rows)}")
    print(f"  original groundtruth rows: {len(groundtruth_rows)}")
    print(f"  synthetic groundtruth rows: {len(synthetic_groundtruth_rows)}")
    print(f"  image mode: {'symlink' if args.symlink else 'copy'}")
    print(f"  synthetic metadata: {args.synthetic_metadata}")
    print("")
    print("Use this for training:")
    print(f"  --data-dir {output_dir}")


if __name__ == "__main__":
    main()
