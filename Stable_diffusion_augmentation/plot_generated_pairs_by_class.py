#!/usr/bin/env python3
"""Plot synthetic paired augmentation images grouped by class and modality."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a class/modality contact sheet from paired augmentation manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("Stable_diffusion_augmentation/out_minority_pairs/paired_augmentation_manifest.csv"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("Stable_diffusion_augmentation/out_minority_pairs/generated_pairs_by_class.png"),
    )
    parser.add_argument("--max-pairs-per-class", type=int, default=None)
    parser.add_argument("--thumb-size", type=int, default=180)
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--background", default="white")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def read_manifest(path: Path, max_pairs_per_class: int | None) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            grouped[row["class_name"]].append(row)

    rows = []
    for class_name in sorted(grouped):
        class_rows = grouped[class_name]
        if max_pairs_per_class is not None:
            class_rows = class_rows[:max_pairs_per_class]
        rows.extend(class_rows)
    if not rows:
        raise ValueError(f"No rows found in manifest: {path}")
    return rows


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def make_sheet(rows: list[dict[str, str]], thumb_size: int, padding: int, background: str) -> Image.Image:
    classes = sorted({row["class_name"] for row in rows})
    grouped = {class_name: [row for row in rows if row["class_name"] == class_name] for class_name in classes}
    max_pairs = max(len(items) for items in grouped.values())
    columns = len(classes) * 2
    header_height = 34
    label_height = 26
    cell_width = thumb_size
    cell_height = header_height + thumb_size + label_height
    width = columns * cell_width + (columns + 1) * padding
    height = max_pairs * cell_height + (max_pairs + 1) * padding

    sheet = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(sheet)
    header_font = load_font(14)
    label_font = load_font(11)

    for class_idx, class_name in enumerate(classes):
        for row_idx, row in enumerate(grouped[class_name]):
            for modality_idx, (modality, path_key) in enumerate(
                (("clinical", "clinical_generated_path"), ("derm", "dermoscopic_generated_path"))
            ):
                col = class_idx * 2 + modality_idx
                x = padding + col * (cell_width + padding)
                y = padding + row_idx * (cell_height + padding)
                header = f"{class_name} {modality}"
                draw.text((x, y), header, fill="black", font=header_font)

                image_y = y + header_height
                with Image.open(row[path_key]) as img:
                    thumb = ImageOps.fit(img.convert("RGB"), (thumb_size, thumb_size), method=Image.Resampling.LANCZOS)
                sheet.paste(thumb, (x, image_y))

                label = truncate(row["synthetic_lesion_id"], 24)
                draw.text((x, image_y + thumb_size + 4), label, fill="black", font=label_font)

    return sheet


def main() -> None:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if args.thumb_size < 32:
        raise ValueError("--thumb-size must be >= 32")
    if args.padding < 0:
        raise ValueError("--padding must be >= 0")

    rows = read_manifest(manifest, args.max_pairs_per_class)
    output_file = args.output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sheet = make_sheet(rows, args.thumb_size, args.padding, args.background)
    sheet.save(output_file)
    print(f"Plotted {len(rows)} synthetic pairs")
    print(f"Saved contact sheet: {output_file}")


if __name__ == "__main__":
    main()
