#!/usr/bin/env python3
"""
Create a contact-sheet image for generated MILK10k augmentation outputs.

The generator saves images under folders such as:
  <output-dir>/MEL/dermoscopic/*.jpg

This script scans a folder, lays all images into a grid, and saves one PNG so
it can be inspected from a notebook, terminal, or remote machine without GUI.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all images in a folder into one contact sheet.")
    parser.add_argument("input_dir", type=Path, help="Folder containing generated images.")
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="PNG/JPG path to save. Defaults to <input_dir>/contact_sheet.png.",
    )
    parser.add_argument("--columns", type=int, default=5, help="Number of columns in the grid.")
    parser.add_argument("--thumb-size", type=int, default=224, help="Square thumbnail size.")
    parser.add_argument("--padding", type=int, default=12, help="Padding between thumbnails.")
    parser.add_argument("--background", default="white", help="Background color.")
    parser.add_argument("--no-labels", action="store_true", help="Do not draw file names under images.")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-images", type=int, default=None, help="Optional limit for very large folders.")
    return parser.parse_args()


def find_images(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    paths = [path for path in input_dir.glob(pattern) if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths)


def load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        return ImageFont.load_default()


def truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars - 3
    left = keep // 2
    right = keep - left
    return f"{text[:left]}...{text[-right:]}"


def make_contact_sheet(
    image_paths: list[Path],
    input_dir: Path,
    columns: int,
    thumb_size: int,
    padding: int,
    background: str,
    draw_labels: bool,
) -> Image.Image:
    if columns < 1:
        raise ValueError("--columns must be >= 1")
    if thumb_size < 32:
        raise ValueError("--thumb-size must be >= 32")
    if padding < 0:
        raise ValueError("--padding must be >= 0")

    label_height = 34 if draw_labels else 0
    cell_width = thumb_size
    cell_height = thumb_size + label_height
    rows = math.ceil(len(image_paths) / columns)
    width = columns * cell_width + (columns + 1) * padding
    height = rows * cell_height + (rows + 1) * padding

    sheet = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(sheet)
    font = load_font()

    for idx, path in enumerate(image_paths):
        row = idx // columns
        col = idx % columns
        x = padding + col * (cell_width + padding)
        y = padding + row * (cell_height + padding)

        with Image.open(path) as img:
            thumb = ImageOps.fit(img.convert("RGB"), (thumb_size, thumb_size), method=Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y))

        if draw_labels:
            rel_name = str(path.relative_to(input_dir))
            label = truncate_middle(rel_name, max_chars=34)
            draw.multiline_text((x, y + thumb_size + 4), label, fill="black", font=font, spacing=2)

    return sheet


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    image_paths = find_images(input_dir, args.recursive)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise ValueError(f"No images found under: {input_dir}")

    output_file = args.output_file.expanduser().resolve() if args.output_file else input_dir / "contact_sheet.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    sheet = make_contact_sheet(
        image_paths=image_paths,
        input_dir=input_dir,
        columns=args.columns,
        thumb_size=args.thumb_size,
        padding=args.padding,
        background=args.background,
        draw_labels=not args.no_labels,
    )
    sheet.save(output_file)

    print(f"Plotted {len(image_paths)} images")
    print(f"Saved contact sheet: {output_file}")


if __name__ == "__main__":
    main()
